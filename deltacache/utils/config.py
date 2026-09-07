"""Configuration management for DeltaCache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class DeltaCacheConfig:
    """
    Configuration for DeltaCache system.

    This configuration controls all aspects of the cache management system
    including memory limits, model parameters, and eviction behavior.
    """

    # Memory limits
    gpu_memory_limit: int = 0  # 0 = auto-detect (90% of available)
    cpu_memory_limit: int = 0  # 0 = unlimited

    # Eviction thresholds
    eviction_threshold: float = 0.9  # Trigger eviction when GPU usage exceeds this
    target_utilization: float = 0.7  # Target utilization after eviction

    # Model parameters
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    block_size: int = 16  # Tokens per block (matches vLLM default)

    # Data types
    dtype: str = "float16"  # "float16", "bfloat16", "float32"

    # Device
    device: str = "cuda"  # "cuda", "cuda:0", "cpu"

    # Eviction policy
    eviction_policy: str = "tiered"  # "lru", "lfu", "composite", "tiered", "adaptive"

    # RoPE parameters
    rope_base: float = 10000.0
    max_position: int = 8192

    # Memory monitor watermarks
    high_watermark: float = 0.85  # Trigger eviction above this GPU utilization
    low_watermark: float = 0.70  # Target utilization after eviction
    critical_threshold: float = 0.95  # Emergency eviction threshold
    enable_memory_monitor: bool = True  # Use real GPU memory monitoring

    # Tiered cache
    enable_offloading: bool = True  # Allow GPU -> CPU offloading
    cpu_cache_quantize: bool = False  # Quantize KV on CPU tier (INT8)

    # Observability
    enable_stats: bool = True  # Track statistics
    enable_metrics: bool = False  # Prometheus metrics (requires prometheus_client)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get PyTorch dtype from string."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float16)

    @property
    def torch_device(self) -> torch.device:
        """Get PyTorch device."""
        return torch.device(self.device)

    @property
    def kv_size_per_token(self) -> int:
        """Memory per token in bytes (for KV cache)."""
        element_size = 2 if self.dtype in ("float16", "bfloat16") else 4
        # key + value, all layers, all heads
        return 2 * self.num_layers * self.num_heads * self.head_dim * element_size

    @property
    def block_memory_size(self) -> int:
        """Memory per block in bytes."""
        return self.kv_size_per_token * self.block_size

    def validate(self) -> None:
        """Validate configuration values."""
        if self.eviction_threshold < 0 or self.eviction_threshold > 1:
            raise ValueError("eviction_threshold must be between 0 and 1")

        if self.target_utilization < 0 or self.target_utilization > 1:
            raise ValueError("target_utilization must be between 0 and 1")

        if self.target_utilization >= self.eviction_threshold:
            raise ValueError("target_utilization must be less than eviction_threshold")

        if self.high_watermark <= self.low_watermark:
            raise ValueError("high_watermark must be greater than low_watermark")

        if self.critical_threshold <= self.high_watermark:
            raise ValueError("critical_threshold must be greater than high_watermark")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")

        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"Invalid dtype: {self.dtype}")

        valid_policies = ("lru", "lfu", "composite", "tiered", "adaptive")
        if self.eviction_policy not in valid_policies:
            raise ValueError(f"Invalid eviction_policy: {self.eviction_policy}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_memory_limit": self.gpu_memory_limit,
            "cpu_memory_limit": self.cpu_memory_limit,
            "eviction_threshold": self.eviction_threshold,
            "target_utilization": self.target_utilization,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "block_size": self.block_size,
            "dtype": self.dtype,
            "device": self.device,
            "eviction_policy": self.eviction_policy,
            "rope_base": self.rope_base,
            "max_position": self.max_position,
            "enable_offloading": self.enable_offloading,
            "enable_stats": self.enable_stats,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DeltaCacheConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def for_model(
        cls,
        model_name: str,
        **overrides,
    ) -> DeltaCacheConfig:
        """
        Create config for known model architectures.

        Args:
            model_name: Model name or path.
            **overrides: Override specific config values.

        Returns:
            Configuration for the model.
        """
        # Common model configs
        model_configs = {
            # GPT-2 models (no RoPE, learned position embeddings)
            "gpt2": {
                "num_layers": 12,
                "num_heads": 12,
                "head_dim": 64,
                "max_position": 1024,
                "rope_base": 0,
            },
            "gpt2-medium": {
                "num_layers": 24,
                "num_heads": 16,
                "head_dim": 64,
                "max_position": 1024,
                "rope_base": 0,
            },
            "gpt2-large": {
                "num_layers": 36,
                "num_heads": 20,
                "head_dim": 64,
                "max_position": 1024,
                "rope_base": 0,
            },
            "gpt2-xl": {
                "num_layers": 48,
                "num_heads": 25,
                "head_dim": 64,
                "max_position": 1024,
                "rope_base": 0,
            },
            # Llama models
            "llama-7b": {"num_layers": 32, "num_heads": 32, "head_dim": 128},
            "llama-13b": {"num_layers": 40, "num_heads": 40, "head_dim": 128},
            "llama-70b": {"num_layers": 80, "num_heads": 64, "head_dim": 128},
            "llama-2-7b": {"num_layers": 32, "num_heads": 32, "head_dim": 128},
            "llama-2-13b": {"num_layers": 40, "num_heads": 40, "head_dim": 128},
            "llama-2-70b": {"num_layers": 80, "num_heads": 64, "head_dim": 128},
            "llama-3-8b": {"num_layers": 32, "num_heads": 32, "head_dim": 128},
            "llama-3-70b": {"num_layers": 80, "num_heads": 64, "head_dim": 128},
            "mistral-7b": {"num_layers": 32, "num_heads": 32, "head_dim": 128},
            "mixtral-8x7b": {"num_layers": 32, "num_heads": 32, "head_dim": 128},
            "qwen-7b": {"num_layers": 32, "num_heads": 32, "head_dim": 128},
            "qwen-14b": {"num_layers": 40, "num_heads": 40, "head_dim": 128},
            "qwen-72b": {"num_layers": 80, "num_heads": 64, "head_dim": 128},
        }

        # Find matching config (prefer longest/most specific match)
        model_key = model_name.lower().replace("_", "-")
        base_config = {}
        best_match_len = 0

        for key, config in model_configs.items():
            # Check if key is a substring of model_key or vice versa
            if key in model_key or model_key in key:
                # Prefer exact match or longest matching key
                match_len = len(key) if key in model_key else 0
                if match_len > best_match_len:
                    best_match_len = match_len
                    base_config = config

        # Merge with overrides
        final_config = {**base_config, **overrides}

        return cls(**final_config)

    def __post_init__(self) -> None:
        """Validate after initialization."""
        self.validate()


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    # Hit/miss tracking
    total_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Token-level stats
    total_tokens: int = 0
    cached_tokens: int = 0
    computed_tokens: int = 0

    # Memory stats
    peak_gpu_usage: int = 0
    peak_cpu_usage: int = 0

    # Eviction stats
    num_evictions: int = 0
    num_offloads: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        if self.total_lookups == 0:
            return 0.0
        return self.cache_hits / self.total_lookups

    @property
    def token_reuse_rate(self) -> float:
        """Token reuse rate."""
        if self.total_tokens == 0:
            return 0.0
        return self.cached_tokens / self.total_tokens

    @property
    def compute_savings(self) -> float:
        """Fraction of computation saved."""
        return self.token_reuse_rate

    def record_lookup(self, hit: bool, matched_tokens: int, computed_tokens: int) -> None:
        """Record a cache lookup."""
        self.total_lookups += 1
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        self.total_tokens += matched_tokens + computed_tokens
        self.cached_tokens += matched_tokens
        self.computed_tokens += computed_tokens

    def record_eviction(self, offloaded: bool) -> None:
        """Record an eviction."""
        self.num_evictions += 1
        if offloaded:
            self.num_offloads += 1

    def update_memory_peak(self, gpu_usage: int, cpu_usage: int) -> None:
        """Update peak memory usage."""
        self.peak_gpu_usage = max(self.peak_gpu_usage, gpu_usage)
        self.peak_cpu_usage = max(self.peak_cpu_usage, cpu_usage)

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_lookups = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_tokens = 0
        self.cached_tokens = 0
        self.computed_tokens = 0
        self.peak_gpu_usage = 0
        self.peak_cpu_usage = 0
        self.num_evictions = 0
        self.num_offloads = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_lookups": self.total_lookups,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.hit_rate,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "computed_tokens": self.computed_tokens,
            "token_reuse_rate": self.token_reuse_rate,
            "compute_savings": self.compute_savings,
            "peak_gpu_usage_mb": self.peak_gpu_usage / (1024 * 1024),
            "peak_cpu_usage_mb": self.peak_cpu_usage / (1024 * 1024),
            "num_evictions": self.num_evictions,
            "num_offloads": self.num_offloads,
        }
