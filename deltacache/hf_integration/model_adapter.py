"""Base adapter for HuggingFace Transformers models.

This module provides the base class for adapting HuggingFace models to work
with DeltaCache's incremental computation engine.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable, Any
import torch
from torch import Tensor

from deltacache.utils.config import DeltaCacheConfig
from deltacache.hf_integration.kv_format import (
    hf_to_deltacache,
    deltacache_to_hf,
    KVFormatConverter,
)


class HFModelAdapter(ABC):
    """Base adapter for HuggingFace Transformers models.

    This adapter converts HuggingFace models to conform to DeltaCache's
    KVComputeFunc protocol, enabling incremental KV cache computation.

    Subclasses should implement model-specific loading and configuration.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[DeltaCacheConfig] = None,
        device: Optional[str] = None,
    ):
        """Initialize the adapter.

        Args:
            model: HuggingFace model instance
            tokenizer: HuggingFace tokenizer instance
            config: DeltaCache configuration. If None, extracted from model.
            device: Device to use. If None, uses model's device.
        """
        self.model = model
        self.tokenizer = tokenizer
        self._device = device or self._get_model_device()

        # Extract or use provided config
        self.config = config or self._extract_config()

        # Create format converter
        self.converter = KVFormatConverter(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            dtype=self._get_dtype(),
        )

        # Ensure model is in eval mode
        self.model.eval()

    def _get_model_device(self) -> str:
        """Get device from model parameters."""
        try:
            param = next(self.model.parameters())
            return str(param.device)
        except StopIteration:
            return "cpu"

    def _get_dtype(self) -> torch.dtype:
        """Get dtype from model parameters."""
        try:
            param = next(self.model.parameters())
            return param.dtype
        except StopIteration:
            return torch.float32

    @abstractmethod
    def _extract_config(self) -> DeltaCacheConfig:
        """Extract DeltaCacheConfig from model configuration.

        Subclasses must implement this to extract model-specific parameters.

        Returns:
            DeltaCacheConfig with model parameters
        """
        pass

    def compute_kv(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        past_key_values: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute KV cache following DeltaCache's KVComputeFunc protocol.

        This method is the main interface for DeltaCache's incremental engine.

        Args:
            input_ids: Input token IDs with shape [batch_size, seq_len]
            position_ids: Position IDs with shape [batch_size, seq_len]
            past_key_values: Optional cached KV from DeltaCache with shape
                [num_layers, seq_len, num_heads, head_dim]

        Returns:
            Tuple of (key_cache, value_cache) in DeltaCache format
        """
        # Convert DeltaCache KV to HuggingFace format if provided
        hf_past = None
        if past_key_values is not None:
            hf_past = deltacache_to_hf(
                past_key_values[0],
                past_key_values[1],
                add_batch_dim=True,
            )

        # Run model forward pass
        with torch.no_grad():
            outputs = self._forward(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=hf_past,
            )

        # Extract and convert new KV cache
        new_kv = self._extract_kv_from_outputs(outputs)

        # If we had past KV, we only get the new tokens' KV
        # HuggingFace returns the full concatenated cache
        key_cache, value_cache = hf_to_deltacache(new_kv, remove_batch_dim=True)

        return key_cache, value_cache

    def _forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
    ) -> Any:
        """Run model forward pass.

        Subclasses can override this for model-specific forward logic.

        Args:
            input_ids: Input token IDs
            position_ids: Position IDs
            past_key_values: Optional HuggingFace format KV cache

        Returns:
            Model outputs containing past_key_values
        """
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    def _extract_kv_from_outputs(self, outputs: Any) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """Extract KV cache from model outputs.

        Args:
            outputs: Model outputs

        Returns:
            HuggingFace format KV cache
        """
        if hasattr(outputs, "past_key_values"):
            return outputs.past_key_values
        raise ValueError("Model outputs do not contain past_key_values")

    def __call__(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        past_key_values: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Make adapter callable as KVComputeFunc."""
        return self.compute_kv(input_ids, position_ids, past_key_values)

    def tokenize(self, text: str, add_special_tokens: bool = True) -> Tensor:
        """Tokenize text and return tensor.

        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens

        Returns:
            Token IDs tensor with shape [1, seq_len]
        """
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        return tokens.to(self._device)

    def decode(self, token_ids: Tensor) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs tensor

        Returns:
            Decoded text
        """
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)
        return self.tokenizer.decode(token_ids)

    def get_position_ids(self, seq_len: int, offset: int = 0) -> Tensor:
        """Create position IDs tensor.

        Args:
            seq_len: Sequence length
            offset: Starting position offset

        Returns:
            Position IDs tensor with shape [1, seq_len]
        """
        positions = torch.arange(offset, offset + seq_len, device=self._device)
        return positions.unsqueeze(0)

    @property
    def device(self) -> str:
        """Get device string."""
        return self._device

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.model.config.vocab_size

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        use_deltacache: bool = False,
        delta_manager: Optional[Any] = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            use_deltacache: Whether to use DeltaCache for generation
            delta_manager: DeltaCacheManager instance (required if use_deltacache=True)

        Returns:
            Generated text including prompt
        """
        input_ids = self.tokenize(prompt)
        generated = input_ids.clone()

        if use_deltacache and delta_manager is not None:
            # Use DeltaCache for incremental generation
            return self._generate_with_deltacache(
                input_ids,
                max_new_tokens,
                temperature,
                top_k,
                delta_manager,
            )

        # Standard generation with HuggingFace cache
        past_kv = None
        for _ in range(max_new_tokens):
            position_ids = self.get_position_ids(
                generated.shape[1] if past_kv is None else 1,
                offset=0 if past_kv is None else generated.shape[1] - 1,
            )

            with torch.no_grad():
                if past_kv is None:
                    outputs = self.model(
                        input_ids=generated,
                        position_ids=self.get_position_ids(generated.shape[1]),
                        use_cache=True,
                        return_dict=True,
                    )
                else:
                    outputs = self.model(
                        input_ids=generated[:, -1:],
                        position_ids=position_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )

            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.decode(generated[0])

    def _generate_with_deltacache(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        delta_manager: Any,
    ) -> str:
        """Generate using DeltaCache for prefix caching.

        Args:
            input_ids: Initial input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            delta_manager: DeltaCacheManager instance

        Returns:
            Generated text
        """
        tokens = input_ids[0].tolist()

        # Get initial KV from DeltaCache
        result = delta_manager.compute_incremental(tokens, self)
        past_kv = deltacache_to_hf(
            result.full_key_cache,
            result.full_value_cache,
            add_batch_dim=True,
        )

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Get logits for last position
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated[:, -1:],
                    position_ids=self.get_position_ids(1, offset=generated.shape[1] - 1),
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )

            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.decode(generated[0])
