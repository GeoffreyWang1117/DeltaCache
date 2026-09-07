"""Llama-style model adapter for DeltaCache.

Supports Llama, Llama-2, Llama-3, Mistral, TinyLlama, Qwen, and other
models using RoPE (Rotary Position Embedding).
"""

from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from deltacache.hf_integration.kv_format import deltacache_to_hf, hf_to_deltacache
from deltacache.hf_integration.model_adapter import HFModelAdapter
from deltacache.utils.config import DeltaCacheConfig

# Model configurations for common Llama-style models
LLAMA_STYLE_CONFIGS = {
    # TinyLlama - great for fast testing
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "num_layers": 22,
        "num_heads": 32,
        "num_kv_heads": 4,
        "head_dim": 64,
        "hidden_size": 2048,
        "max_position": 2048,
    },
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": {
        "num_layers": 22,
        "num_heads": 32,
        "num_kv_heads": 4,
        "head_dim": 64,
        "hidden_size": 2048,
        "max_position": 2048,
    },
    # Qwen2 - small efficient model
    "Qwen/Qwen2-0.5B": {
        "num_layers": 24,
        "num_heads": 14,
        "num_kv_heads": 2,
        "head_dim": 64,
        "hidden_size": 896,
        "max_position": 32768,
    },
    "Qwen/Qwen2-1.5B": {
        "num_layers": 28,
        "num_heads": 12,
        "num_kv_heads": 2,
        "head_dim": 128,
        "hidden_size": 1536,
        "max_position": 32768,
    },
    # Mistral - efficient 7B
    "mistralai/Mistral-7B-v0.1": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "max_position": 32768,
    },
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "max_position": 32768,
    },
    # Llama-2
    "meta-llama/Llama-2-7b-hf": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "hidden_size": 4096,
        "max_position": 4096,
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "hidden_size": 4096,
        "max_position": 4096,
    },
    "meta-llama/Llama-2-13b-hf": {
        "num_layers": 40,
        "num_heads": 40,
        "num_kv_heads": 40,
        "head_dim": 128,
        "hidden_size": 5120,
        "max_position": 4096,
    },
}


class LlamaStyleAdapter(HFModelAdapter):
    """Adapter for Llama-style models (Llama, Mistral, TinyLlama, Qwen, etc.).

    These models use Rotary Position Embeddings (RoPE) and have similar
    architectures with potentially different numbers of KV heads (GQA).

    Example:
        >>> adapter = LlamaStyleAdapter.from_pretrained(
        ...     "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ...     device="cuda"
        ... )
        >>> tokens = adapter.tokenize("Hello, world!")
        >>> kv = adapter.compute_kv(tokens, adapter.get_position_ids(tokens.shape[1]))
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[DeltaCacheConfig] = None,
        device: Optional[str] = None,
        num_kv_heads: Optional[int] = None,
    ):
        """Initialize the adapter.

        Args:
            model: HuggingFace model instance
            tokenizer: HuggingFace tokenizer instance
            config: Optional DeltaCacheConfig override
            device: Device string
            num_kv_heads: Number of KV heads (for GQA models)
        """
        self._num_kv_heads = num_kv_heads
        super().__init__(model, tokenizer, config, device)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        config: Optional[DeltaCacheConfig] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
    ) -> "LlamaStyleAdapter":
        """Load model and tokenizer from HuggingFace.

        Args:
            model_name: Model name or path
            device: Device to load on (None for auto)
            dtype: Model dtype (None for auto)
            config: Optional DeltaCacheConfig override
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            trust_remote_code: Execute Python shipped inside the model repository.
                Defaults to False. Turning it on runs arbitrary code from
                whatever `model_name` resolves to at load time, so it must be
                an explicit decision by the caller and never a default.
            revision: Pin the model to an exact commit. Without it the name
                resolves to whatever the repository's default branch points at
                today, which is neither reproducible nor a fixed trust anchor.

        Returns:
            Initialized adapter
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            ) from exc

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "revision": revision,
        }

        if load_in_8bit or load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                )
                model_kwargs["quantization_config"] = quantization_config
                # Use specific device to avoid multi-GPU distribution
                model_kwargs["device_map"] = {"": device} if device else "auto"
            except ImportError:
                print("Warning: bitsandbytes not available, loading without quantization")
                if dtype is None:
                    dtype = torch.float16
                model_kwargs["torch_dtype"] = dtype
        else:
            if dtype is None:
                dtype = torch.float16 if device != "cpu" else torch.float32
            model_kwargs["torch_dtype"] = dtype

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if not (load_in_8bit or load_in_4bit):
            model = model.to(device)

        model.eval()

        # Get num_kv_heads from model config
        hf_config = model.config
        num_kv_heads = getattr(hf_config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(hf_config, "num_attention_heads", 32)

        return cls(
            model,
            tokenizer,
            config=config,
            device=device,
            num_kv_heads=num_kv_heads,
        )

    def _extract_config(self) -> DeltaCacheConfig:
        """Extract DeltaCacheConfig from model configuration."""
        hf_config = self.model.config

        # Get number of KV heads (for GQA models)
        num_kv_heads = self._num_kv_heads
        if num_kv_heads is None:
            num_kv_heads = getattr(hf_config, "num_key_value_heads", None)
            if num_kv_heads is None:
                num_kv_heads = hf_config.num_attention_heads

        # Calculate head dimension
        hidden_size = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        head_dim = hidden_size // num_heads

        # Get RoPE base
        rope_base = getattr(hf_config, "rope_theta", 10000.0)

        return DeltaCacheConfig(
            num_layers=hf_config.num_hidden_layers,
            num_heads=num_kv_heads,  # Use KV heads for cache
            head_dim=head_dim,
            max_position=getattr(hf_config, "max_position_embeddings", 4096),
            rope_base=rope_base,
            device=self._device,
        )

    def _forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
    ) -> Any:
        """Run model forward pass."""
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    def compute_kv_for_tokens(
        self,
        tokens: List[int],
        position_offset: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute KV cache for a list of tokens.

        Args:
            tokens: List of token IDs
            position_offset: Starting position

        Returns:
            Tuple of (key_cache, value_cache) in DeltaCache format
        """
        input_ids = torch.tensor([tokens], device=self._device)
        position_ids = self.get_position_ids(len(tokens), offset=position_offset)
        return self.compute_kv(input_ids, position_ids)

    def generate_with_cache(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        delta_manager: Optional[Any] = None,
    ) -> Tuple[str, dict]:
        """Generate text with optional DeltaCache.

        Args:
            prompt: Input prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample
            delta_manager: Optional DeltaCacheManager for caching

        Returns:
            Tuple of (generated_text, stats_dict)
        """
        import time

        input_ids = self.tokenize(prompt)
        tokens = input_ids[0].tolist()
        stats = {
            "prompt_tokens": len(tokens),
            "generated_tokens": 0,
            "cached_tokens": 0,
            "computed_tokens": 0,
            "prompt_time_ms": 0,
            "generation_time_ms": 0,
        }

        # Process prompt
        start = time.perf_counter()

        if delta_manager is not None:
            # Use DeltaCache
            result = delta_manager.compute_incremental(tokens, self)
            past_kv = deltacache_to_hf(
                result.key_cache,
                result.value_cache,
                add_batch_dim=True,
            )
            stats["cached_tokens"] = result.matched_length
            stats["computed_tokens"] = result.computed_length
        else:
            # Standard forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                )
            past_kv = outputs.past_key_values
            stats["computed_tokens"] = len(tokens)

        stats["prompt_time_ms"] = (time.perf_counter() - start) * 1000

        # Generate tokens
        generated = input_ids.clone()
        start = time.perf_counter()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated[:, -1:],
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )

            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if do_sample and temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            stats["generated_tokens"] += 1

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        stats["generation_time_ms"] = (time.perf_counter() - start) * 1000

        return self.decode(generated[0]), stats

    def verify_cache_correctness(
        self,
        tokens: List[int],
        delta_manager: Any,
        tolerance: float = 1e-4,
    ) -> dict:
        """Verify that cached KV produces correct outputs.

        Args:
            tokens: Input token IDs
            delta_manager: DeltaCacheManager instance
            tolerance: Numerical tolerance for comparison

        Returns:
            Dict with correctness metrics
        """
        input_ids = torch.tensor([tokens], device=self._device)
        position_ids = self.get_position_ids(len(tokens))

        # Compute without cache (ground truth)
        with torch.no_grad():
            outputs_no_cache = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
        logits_no_cache = outputs_no_cache.logits

        # Convert ground truth KV to DeltaCache format for comparison
        kv_no_cache_dc = hf_to_deltacache(
            outputs_no_cache.past_key_values,
            remove_batch_dim=True,
        )

        # Compute with DeltaCache
        result = delta_manager.compute_incremental(tokens, self)

        # Compare KV values directly in DeltaCache format
        # Both should be [num_layers, seq_len, num_heads, head_dim]
        key_diff = (kv_no_cache_dc[0] - result.key_cache).abs()
        value_diff = (kv_no_cache_dc[1] - result.value_cache).abs()

        kv_max_diff = max(key_diff.max().item(), value_diff.max().item())
        kv_mean_diff = (key_diff.mean().item() + value_diff.mean().item()) / 2

        # Now test that the cached KV produces correct logits
        # We need to use KV for positions 0 to N-2, then compute position N-1
        # This tests that the cached prefix produces correct next-token prediction
        if len(tokens) > 1:
            prefix_key = result.key_cache[:, :-1, :, :]
            prefix_value = result.value_cache[:, :-1, :, :]
            kv_prefix = deltacache_to_hf(
                prefix_key,
                prefix_value,
                add_batch_dim=True,
            )

            # Compute logits for last token using prefix cache
            with torch.no_grad():
                outputs_cached = self.model(
                    input_ids=input_ids[:, -1:],
                    position_ids=position_ids[:, -1:],
                    past_key_values=kv_prefix,
                    use_cache=True,
                    return_dict=True,
                )

            logits_cached_last = outputs_cached.logits[:, -1, :]
        else:
            # Single token - just use the logits from ground truth
            logits_cached_last = logits_no_cache[:, -1, :]

        logits_no_cache_last = logits_no_cache[:, -1, :]
        logits_diff = (logits_cached_last - logits_no_cache_last).abs()

        # Check token predictions match
        pred_cached = logits_cached_last.argmax(dim=-1)
        pred_no_cache = logits_no_cache_last.argmax(dim=-1)
        tokens_match = (pred_cached == pred_no_cache).all().item()

        return {
            "kv_max_diff": kv_max_diff,
            "kv_mean_diff": kv_mean_diff,
            "logits_max_diff": logits_diff.max().item(),
            "logits_mean_diff": logits_diff.mean().item(),
            "tokens_match": tokens_match,
            "is_correct": kv_max_diff < tolerance,
            "cached_tokens": result.matched_length,
            "computed_tokens": result.computed_length,
        }
