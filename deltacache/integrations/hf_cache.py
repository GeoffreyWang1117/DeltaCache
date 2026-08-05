"""HuggingFace Transformers integration: LayerBudgetCache.

A DynamicCache subclass that applies per-layer joint token-precision
optimization when the cache exceeds a configurable memory budget.

Usage:
    from deltacache.integrations.hf_cache import LayerBudgetCache

    cache = LayerBudgetCache(max_memory_mb=256)
    output = model.generate(inputs, past_key_values=cache, max_new_tokens=100)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    raise ImportError("transformers>=4.36 required for DynamicCache support")


def _compute_gini(weights: Tensor) -> float:
    w = weights.float().flatten()
    w = w[w > 0]
    if w.numel() < 2 or w.sum() < 1e-10:
        return 0.0
    sorted_w, _ = w.sort()
    n = sorted_w.numel()
    idx = torch.arange(1, n + 1, dtype=torch.float32, device=w.device)
    return ((2 * (idx * sorted_w).sum() - (n + 1) * sorted_w.sum()) / (n * sorted_w.sum())).item()


def _get_kv(cache: DynamicCache, layer_idx: int):
    """Get (key, value) tensors for a layer, compatible with old and new API."""
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    return cache.layers[layer_idx].keys, cache.layers[layer_idx].values


def _set_kv(cache: DynamicCache, layer_idx: int, k: Tensor, v: Tensor):
    """Set (key, value) tensors for a layer."""
    if hasattr(cache, "key_cache"):
        cache.key_cache[layer_idx] = k
        cache.value_cache[layer_idx] = v
    else:
        cache.layers[layer_idx].keys = k
        cache.layers[layer_idx].values = v


def _num_layers(cache: DynamicCache) -> int:
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    return len(cache.layers)


class LayerBudgetCache(DynamicCache):
    """DynamicCache with automatic per-layer compression.

    Behaves like DynamicCache until total memory exceeds max_memory_mb,
    then triggers LayerBudget compression.
    """

    def __init__(
        self,
        max_memory_mb: float = 512.0,
        compression_ratio: float = 0.25,
        available_bits: Optional[List[int]] = None,
        sink_tokens: int = 4,
        recent_tokens: int = 16,
    ):
        super().__init__()
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.compression_ratio = compression_ratio
        self.available_bits = sorted(available_bits or [4, 8, 16])
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self._compressed = False
        self._sparsity: Dict[int, float] = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Profile sparsity from value norms
        v = value_states[0]
        if v.shape[1] > 1:
            norms = v.float().norm(dim=-1).mean(dim=0)
            self._sparsity[layer_idx] = _compute_gini(norms)

        # Compress after last layer if over budget
        nl = _num_layers(self)
        if layer_idx == nl - 1 and not self._compressed:
            mem = self._estimate_memory()
            if mem > self.max_memory_bytes:
                self._compress()

        return result

    def _estimate_memory(self) -> int:
        total = 0
        for l in range(_num_layers(self)):
            k, v = _get_kv(self, l)
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
        return total

    def _compress(self):
        """Compress KV cache in-place, preserving sequence length.

        Instead of removing tokens (which breaks generate()'s position tracking),
        we zero-fill evicted positions and apply quantize-dequantize to retained
        positions. This keeps the tensor shapes constant so attention masks and
        cache_position remain valid.
        """
        nl = _num_layers(self)
        if nl == 0:
            return

        k0, _ = _get_kv(self, 0)
        seq_len = k0.shape[2]
        num_heads = k0.shape[1]
        head_dim = k0.shape[3]
        device = k0.device

        if seq_len < self.sink_tokens + self.recent_tokens + 4:
            return

        # Importance weights (inverted: early layers get higher weight)
        importance = {}
        for l in range(nl):
            pos = 1.0 - l / max(1, nl - 1)  # inverted: early layers high
            x = 5.0 * (pos - 0.3)
            importance[l] = 1.0 / (1.0 + math.exp(-x))

        # Allocate
        fidelity = {16: 1.0, 8: 0.9999, 4: 0.9964}
        min_bits = self.available_bits[0]
        n_min = min(self.sink_tokens + self.recent_tokens, seq_len)
        full_mem = 2 * nl * seq_len * num_heads * head_dim * 2
        budget = int(full_mem * self.compression_ratio)
        token_step = 8

        alloc = [(n_min, min_bits)] * nl

        def mem_cost(n, b):
            return 2 * n * num_heads * head_dim * b // 8

        def quality(l, n, b):
            f = n / max(1, seq_len)
            g = self._sparsity.get(l, 0.5)
            return (f ** max(0.01, 1 - g)) * fidelity.get(b, 0.9) * importance.get(l, 0.5)

        used = sum(mem_cost(n, b) for n, b in alloc)
        for _ in range(nl * seq_len // token_step):
            best_gpb, best_act = 0, None
            for l in range(nl):
                n, b = alloc[l]
                if n + token_step <= seq_len:
                    nn = n + token_step
                    dq = quality(l, nn, b) - quality(l, n, b)
                    dm = mem_cost(nn, b) - mem_cost(n, b)
                    if dm > 0 and used + dm <= budget and dq / dm > best_gpb:
                        best_gpb = dq / dm
                        best_act = (l, nn, b, dm)
                bi = self.available_bits.index(b)
                if bi < len(self.available_bits) - 1:
                    nb = self.available_bits[bi + 1]
                    dq = quality(l, n, nb) - quality(l, n, b)
                    dm = mem_cost(n, nb) - mem_cost(n, b)
                    if dm > 0 and used + dm <= budget and dq / dm > best_gpb:
                        best_gpb = dq / dm
                        best_act = (l, n, nb, dm)
            if best_act is None:
                break
            l, n_new, b_new, dm = best_act
            alloc[l] = (n_new, b_new)
            used += dm

        # Apply in-place: zero-fill evicted positions, QDQ retained positions.
        # Tensor shapes are preserved so generate()'s position tracking stays valid.
        for l, (n_tokens, bits) in enumerate(alloc):
            k, v = _get_kv(self, l)
            if n_tokens < seq_len:
                # Determine which tokens to keep
                sink = min(self.sink_tokens, seq_len)
                recent = min(self.recent_tokens, max(0, seq_len - sink))
                mid_budget = max(0, n_tokens - sink - recent)
                rec_start = max(sink, seq_len - recent)

                # Build retain mask
                retain = torch.zeros(seq_len, dtype=torch.bool, device=device)
                retain[:sink] = True
                retain[rec_start:seq_len] = True

                if mid_budget > 0 and rec_start > sink:
                    norms = v[0, :, sink:rec_start, :].float().norm(dim=-1).mean(dim=0)
                    topk = min(mid_budget, len(norms))
                    _, top = norms.topk(topk)
                    retain[top + sink] = True

                # Zero-fill evicted positions in-place
                evict_mask = ~retain
                k[:, :, evict_mask, :] = 0
                v[:, :, evict_mask, :] = 0

            if bits < 16:
                k = self._qdq(k, bits)
                v = self._qdq(v, bits)

            _set_kv(self, l, k, v)

        self._compressed = True

    @staticmethod
    def _qdq(t, bits):
        tf = t.float()
        qmax = (1 << bits) - 1
        t_min = tf.amin(dim=-1, keepdim=True)
        t_max = tf.amax(dim=-1, keepdim=True)
        scale = ((t_max - t_min) / qmax).clamp(min=1e-8)
        q = ((tf - t_min) / scale).round().clamp(0, qmax)
        return (q * scale + t_min).to(t.dtype)
