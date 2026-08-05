"""Base class and registry for all KV cache compression baselines.

All baselines implement the same interface so they can be compared
under controlled conditions: same model, prompts, metrics, and hardware.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# Global registry of baseline methods
REGISTRY: Dict[str, type] = {}


def register_baseline(cls: type) -> type:
    """Decorator to register a baseline method."""
    REGISTRY[cls.name] = cls
    return cls


@dataclass
class CompressResult:
    """Standardized output from any baseline's compress() method."""

    layers: List[Tuple[Tensor, Tensor, Tensor]]
    # Each tuple: (keys_fp16, values_fp16, token_indices)
    #   keys/values: (1, n_tokens, num_heads, head_dim) in FP16
    #   token_indices: (n_tokens,) long tensor
    memory_bytes: int
    compress_time_ms: float


class BaselineMethod(ABC):
    """Abstract interface for KV cache compression baselines.

    Every baseline must implement compress() with a unified signature.
    The returned K/V are always dequantized to FP16 for fair quality
    comparison; actual memory is reported separately via memory_bytes().
    """

    # Subclass must set these class attributes:
    name: str = ""  # Short identifier (e.g., "snapkv")
    category: str = ""  # "eviction" | "quantization" | "joint"
    requires_attention: bool = False
    requires_hidden_states: bool = False
    is_per_layer: bool = False
    reference: str = ""  # Paper citation

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        **kwargs,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    @abstractmethod
    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        """Compress KV cache and return per-layer results.

        Args:
            full_keys: (num_layers, seq_len, num_heads, head_dim) FP16.
            full_values: Same shape.
            compression_ratio: Target compression (e.g., 3.0 = 33% memory).
            attention_weights: Per-layer attention tensors, each
                (batch, heads, seq_len, seq_len). Required if
                requires_attention is True.
            hidden_states: Per-layer hidden states, each
                (batch, seq_len, hidden_dim). Required if
                requires_hidden_states is True.

        Returns:
            List of (keys_fp16, values_fp16, token_indices) per layer.
            - keys/values are dequantized back to FP16 for comparison.
            - token_indices are the original positions retained.
            - For quant-only methods: indices = arange(seq_len).
        """
        ...

    def compress_timed(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> CompressResult:
        """Compress with timing and memory accounting."""
        t0 = time.perf_counter()
        layers = self.compress(
            full_keys, full_values, compression_ratio,
            attention_weights=attention_weights,
            hidden_states=hidden_states,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        mem = self.memory_bytes(layers)
        return CompressResult(
            layers=layers,
            memory_bytes=mem,
            compress_time_ms=elapsed,
        )

    def memory_bytes(self, compressed_layers: List[Tuple[Tensor, Tensor, Tensor]]) -> int:
        """Actual memory consumption.

        Default: FP16 memory of returned tensors. Subclasses that use
        quantization should override to report true compressed size.
        """
        total = 0
        for k, v, _ in compressed_layers:
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
        return total

    def _full_memory(self, seq_len: int) -> int:
        """Total FP16 memory for all layers at full sequence."""
        return 2 * self.num_layers * seq_len * self.num_heads * self.head_dim * 2

    def _layer_memory_fp16(self, n_tokens: int) -> int:
        """FP16 memory for one layer."""
        return 2 * n_tokens * self.num_heads * self.head_dim * 2


# =========================================================
#  Shared token selection utilities
# =========================================================


def h2o_token_selection(
    keys: Tensor,
    values: Tensor,
    n_tokens: int,
    seq_len: int,
    attention_weights: Optional[Tensor] = None,
    sink_tokens: int = 4,
    recent_tokens: int = 16,
) -> Tensor:
    """H2O-style token selection: sink + recent + heavy-hitter middle.

    If attention_weights is provided, uses cumulative attention mass.
    Otherwise, falls back to value-norm importance (as in LayerKVStore).

    Args:
        keys: (1, seq_len, H, D)
        values: (1, seq_len, H, D)
        n_tokens: Number of tokens to retain.
        seq_len: Total sequence length.
        attention_weights: Optional (1, heads, seq_len, seq_len) for this layer.
        sink_tokens: Always-retained initial tokens.
        recent_tokens: Always-retained final tokens.

    Returns:
        Sorted tensor of selected token indices (long).
    """
    if n_tokens >= seq_len:
        return torch.arange(seq_len)

    sink = min(sink_tokens, seq_len)
    recent = min(recent_tokens, max(0, seq_len - sink))
    middle_budget = max(0, n_tokens - sink - recent)

    sink_idx = torch.arange(sink)
    recent_start = max(sink, seq_len - recent)
    recent_idx = torch.arange(recent_start, seq_len)

    if middle_budget > 0 and recent_start > sink:
        if attention_weights is not None:
            # Cumulative attention received by each position
            scores = attention_weights[0].sum(dim=0).sum(dim=0).cpu()  # (seq_len,)
            middle_scores = scores[sink:recent_start]
        else:
            # Fallback: value norm as importance proxy
            v = values[0, sink:recent_start]  # (mid_len, H, D)
            middle_scores = v.float().norm(dim=-1).mean(dim=-1).cpu()  # (mid_len,)

        k = min(middle_budget, len(middle_scores))
        _, top_idx = middle_scores.topk(k)
        middle_idx = top_idx + sink
    else:
        middle_idx = torch.tensor([], dtype=torch.long)

    all_idx = torch.cat([sink_idx, middle_idx, recent_idx])
    all_idx = all_idx.unique()
    all_idx, _ = all_idx.sort()
    return all_idx[:n_tokens]
