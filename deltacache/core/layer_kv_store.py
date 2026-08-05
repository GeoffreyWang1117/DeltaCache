"""Per-layer KV storage with independent token sets and precision.

Each layer maintains its own KV cache with a potentially different:
  - Token subset (via index selection)
  - Quantization precision (FP16 / INT8 / INT4)

This enables the LayerBudget allocation strategy where high-sparsity
layers keep more tokens at lower precision while semantically important
layers keep fewer tokens at higher precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision, QuantizedKV


@dataclass
class LayerEntry:
    """KV cache entry for a single layer."""

    layer_idx: int
    token_indices: Tensor  # (n_tokens,) — which tokens are stored
    quant_bits: int  # 4, 8, or 16
    # Stored data (one of these is set)
    key_fp: Optional[Tensor] = None  # (1, n_tokens, num_heads, head_dim) if FP16
    value_fp: Optional[Tensor] = None
    quantized: Optional[QuantizedKV] = None  # If quantized

    @property
    def num_tokens(self) -> int:
        return self.token_indices.numel()

    @property
    def memory_bytes(self) -> int:
        if self.quantized is not None:
            return self.quantized.memory_size
        total = 0
        if self.key_fp is not None:
            total += self.key_fp.numel() * self.key_fp.element_size()
        if self.value_fp is not None:
            total += self.value_fp.numel() * self.value_fp.element_size()
        return total


class LayerKVStore:
    """Per-layer KV cache storage with independent budgets and precision.

    Stores KV cache for each layer independently, allowing different
    token subsets and quantization levels per layer.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of KV heads.
        head_dim: Head dimension.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._entries: Dict[int, LayerEntry] = {}
        self._quantizers: Dict[int, KVQuantizer] = {
            4: KVQuantizer(QuantPrecision.INT4),
            8: KVQuantizer(QuantPrecision.INT8),
        }

    def store_layer(
        self,
        layer_idx: int,
        keys: Tensor,
        values: Tensor,
        token_indices: Tensor,
        quant_bits: int = 16,
    ) -> LayerEntry:
        """Store KV cache for a single layer with optional quantization.

        Args:
            layer_idx: Layer index.
            keys: Key tensor, shape (seq_len, num_heads, head_dim) or
                  (1, seq_len, num_heads, head_dim).
            values: Value tensor, same shape as keys.
            token_indices: 1-D tensor of token position indices to retain.
            quant_bits: Precision (4, 8, or 16).

        Returns:
            LayerEntry with stored data.
        """
        # Ensure 4-D shape: (1, n_tokens, num_heads, head_dim)
        if keys.dim() == 3:
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)

        # Select tokens by index
        idx = token_indices.long()
        selected_keys = keys[:, idx, :, :]
        selected_values = values[:, idx, :, :]

        entry = LayerEntry(
            layer_idx=layer_idx,
            token_indices=token_indices.cpu(),
            quant_bits=quant_bits,
        )

        if quant_bits < 16 and quant_bits in self._quantizers:
            quantizer = self._quantizers[quant_bits]
            entry.quantized = quantizer.quantize(selected_keys, selected_values)
        else:
            entry.key_fp = selected_keys
            entry.value_fp = selected_values

        self._entries[layer_idx] = entry
        return entry

    def store_from_full_cache(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        allocations: List,
        token_selector: Optional[object] = None,
    ) -> None:
        """Store KV from full cache using LayerBudget allocations.

        Args:
            full_keys: Full key cache (num_layers, seq_len, num_heads, head_dim).
            full_values: Full value cache, same shape.
            allocations: List of LayerAllocation from LayerBudgetAllocator.
            token_selector: Optional callable(layer_idx, keys, values, n_tokens)
                           → token_indices. Defaults to H2O-style selection.
        """
        seq_len = full_keys.shape[1]

        for alloc in allocations:
            l = alloc.layer_idx
            n = alloc.token_budget
            bits = alloc.quant_bits

            layer_keys = full_keys[l:l+1]  # (1, seq_len, H, D)
            layer_values = full_values[l:l+1]

            if token_selector is not None:
                indices = token_selector(l, layer_keys, layer_values, n)
            else:
                indices = self._default_token_selection(
                    layer_keys, layer_values, n, seq_len,
                )

            self.store_layer(l, layer_keys, layer_values, indices, bits)

    def _default_token_selection(
        self,
        keys: Tensor,
        values: Tensor,
        n_tokens: int,
        seq_len: int,
        sink_tokens: int = 4,
        recent_tokens: int = 16,
    ) -> Tensor:
        """H2O-style token selection: sink + important + recent.

        Selects attention-sink tokens (first K), recent tokens (last W),
        and fills remaining budget with tokens having highest value norms
        (proxy for importance without explicit attention scores).

        Args:
            keys: (1, seq_len, H, D)
            values: (1, seq_len, H, D)
            n_tokens: Number of tokens to select.
            seq_len: Total sequence length.
            sink_tokens: Number of sink tokens (always first K).
            recent_tokens: Number of recent tokens (always last W).

        Returns:
            1-D tensor of selected token indices.
        """
        if n_tokens >= seq_len:
            return torch.arange(seq_len)

        sink = min(sink_tokens, seq_len)
        recent = min(recent_tokens, seq_len - sink)
        middle_budget = max(0, n_tokens - sink - recent)

        # Sink indices (first K)
        sink_idx = torch.arange(sink)

        # Recent indices (last W)
        recent_start = max(sink, seq_len - recent)
        recent_idx = torch.arange(recent_start, seq_len)

        if middle_budget > 0 and recent_start > sink:
            # Middle tokens: select by value norm (proxy for importance)
            middle_range = torch.arange(sink, recent_start)
            middle_values = values[0, sink:recent_start]  # (n_middle, H, D)
            norms = middle_values.float().norm(dim=-1).mean(dim=-1).cpu()  # (n_middle,)
            _, top_idx = norms.topk(min(middle_budget, len(middle_range)))
            middle_idx = middle_range[top_idx]
        else:
            middle_idx = torch.tensor([], dtype=torch.long)

        # Combine and sort
        all_idx = torch.cat([sink_idx, middle_idx, recent_idx])
        all_idx = all_idx.unique()
        all_idx, _ = all_idx.sort()

        # Truncate to budget
        return all_idx[:n_tokens]

    def get_layer(
        self,
        layer_idx: int,
        device: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Retrieve KV cache for a layer (dequantized if needed).

        Args:
            layer_idx: Layer index.
            device: Target device. If None, returns on original device.

        Returns:
            (keys, values, token_indices) where keys/values have shape
            (1, n_tokens, num_heads, head_dim) in FP16.
        """
        if layer_idx not in self._entries:
            raise KeyError(f"Layer {layer_idx} not stored")

        entry = self._entries[layer_idx]

        if entry.quantized is not None:
            quantizer = self._quantizers[entry.quant_bits]
            keys, values = quantizer.dequantize(entry.quantized)
        else:
            keys = entry.key_fp
            values = entry.value_fp

        if device is not None:
            keys = keys.to(device)
            values = values.to(device)

        return keys, values, entry.token_indices

    def get_all_layers(
        self,
        device: Optional[str] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        """Retrieve all layers in order.

        Returns:
            List of (keys, values, token_indices) per layer.
        """
        return [
            self.get_layer(l, device)
            for l in range(self.num_layers)
            if l in self._entries
        ]

    def memory_usage(self) -> int:
        """Total memory across all stored layers."""
        return sum(e.memory_bytes for e in self._entries.values())

    def layer_summary(self) -> List[Dict]:
        """Summary of per-layer storage."""
        summary = []
        for l in range(self.num_layers):
            if l in self._entries:
                e = self._entries[l]
                summary.append({
                    "layer": l,
                    "tokens": e.num_tokens,
                    "bits": e.quant_bits,
                    "memory_bytes": e.memory_bytes,
                })
            else:
                summary.append({
                    "layer": l,
                    "tokens": 0,
                    "bits": 0,
                    "memory_bytes": 0,
                })
        return summary

    def clear(self) -> None:
        """Remove all stored entries."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, layer_idx: int) -> bool:
        return layer_idx in self._entries
