"""GPT-2 specific adapter for DeltaCache.

GPT-2 uses absolute position embeddings (not RoPE), which simplifies
KV cache handling as positions are baked into the embeddings.
"""

from typing import Any, Optional

import torch
from torch import Tensor

from deltacache.hf_integration.model_adapter import HFModelAdapter
from deltacache.utils.config import DeltaCacheConfig

# GPT-2 model configurations
GPT2_CONFIGS = {
    "gpt2": {
        "num_layers": 12,
        "num_heads": 12,
        "head_dim": 64,
        "hidden_size": 768,
        "vocab_size": 50257,
        "max_position": 1024,
    },
    "gpt2-medium": {
        "num_layers": 24,
        "num_heads": 16,
        "head_dim": 64,
        "hidden_size": 1024,
        "vocab_size": 50257,
        "max_position": 1024,
    },
    "gpt2-large": {
        "num_layers": 36,
        "num_heads": 20,
        "head_dim": 64,
        "hidden_size": 1280,
        "vocab_size": 50257,
        "max_position": 1024,
    },
    "gpt2-xl": {
        "num_layers": 48,
        "num_heads": 25,
        "head_dim": 64,
        "hidden_size": 1600,
        "vocab_size": 50257,
        "max_position": 1024,
    },
}


class GPT2Adapter(HFModelAdapter):
    """Adapter for GPT-2 models.

    GPT-2 uses learned absolute position embeddings, so KV cache can be
    directly reused without position adjustment.

    Example:
        >>> adapter = GPT2Adapter.from_pretrained("gpt2")
        >>> tokens = adapter.tokenize("Hello, world!")
        >>> kv = adapter.compute_kv(tokens, adapter.get_position_ids(tokens.shape[1]))
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        config: Optional[DeltaCacheConfig] = None,
    ) -> "GPT2Adapter":
        """Load GPT-2 model and tokenizer from HuggingFace.

        Args:
            model_name: Model name or path (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
            device: Device to load model on. If None, uses CUDA if available.
            dtype: Model dtype. If None, uses float32.
            config: Optional DeltaCacheConfig override.

        Returns:
            Initialized GPT2Adapter
        """
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for GPT2Adapter. Install with: pip install transformers"
            ) from exc

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Set pad token if not set (GPT-2 doesn't have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model, tokenizer, config=config, device=device)

    def _extract_config(self) -> DeltaCacheConfig:
        """Extract DeltaCacheConfig from GPT-2 model configuration."""
        hf_config = self.model.config

        return DeltaCacheConfig(
            num_layers=hf_config.n_layer,
            num_heads=hf_config.n_head,
            head_dim=hf_config.n_embd // hf_config.n_head,
            max_position=hf_config.n_positions,
            device=self._device,
            # GPT-2 doesn't use RoPE
            rope_base=0,  # Indicates no RoPE
        )

    def _forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        past_key_values: Optional[Any] = None,
    ) -> Any:
        """Run GPT-2 forward pass.

        Args:
            input_ids: Input token IDs [batch, seq]
            position_ids: Position IDs [batch, seq]
            past_key_values: Optional HuggingFace format KV cache

        Returns:
            Model outputs with past_key_values
        """
        # GPT-2 handles position embeddings internally
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    def compute_kv_for_tokens(
        self,
        tokens: list[int],
        position_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Compute KV cache for a list of tokens.

        Convenience method that handles tensor conversion.

        Args:
            tokens: List of token IDs
            position_offset: Starting position (for continuation)

        Returns:
            Tuple of (key_cache, value_cache) in DeltaCache format
        """
        input_ids = torch.tensor([tokens], device=self._device)
        position_ids = self.get_position_ids(len(tokens), offset=position_offset)
        return self.compute_kv(input_ids, position_ids)

    def get_embedding(self, token_ids: Tensor) -> Tensor:
        """Get token embeddings.

        Args:
            token_ids: Token IDs tensor

        Returns:
            Token embeddings
        """
        return self.model.transformer.wte(token_ids)

    def get_logits(
        self,
        input_ids: Tensor,
        past_key_values: Optional[Any] = None,
    ) -> Tensor:
        """Get logits for next token prediction.

        Args:
            input_ids: Input token IDs
            past_key_values: Optional HuggingFace format KV cache

        Returns:
            Logits tensor [batch, seq, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        return outputs.logits


def get_gpt2_config(model_name: str) -> dict:
    """Get GPT-2 model configuration.

    Args:
        model_name: Model name (gpt2, gpt2-medium, etc.)

    Returns:
        Configuration dictionary
    """
    base_name = model_name.split("/")[-1].lower()
    if base_name in GPT2_CONFIGS:
        return GPT2_CONFIGS[base_name]
    raise ValueError(f"Unknown GPT-2 model: {model_name}")
