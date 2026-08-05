"""NVIDIA KVPress integration for LayerBudget.

Provides LayerBudgetPress — a KVPress-compatible press that jointly
optimizes per-layer token eviction and quantization precision.

Usage with KVPress:
    from kvpress import PerLayerCompressionPress
    from deltacache.integrations.kvpress_integration import LayerBudgetPress

    press = LayerBudgetPress(compression_ratio=0.25)  # 4x compression
    with press(model):
        output = model.generate(inputs, max_new_tokens=100)

Usage standalone (no KVPress dependency):
    press = LayerBudgetPress(compression_ratio=0.25)
    press.compress_cache(model, past_key_values)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

# Try to import KVPress base class; fall back to standalone if unavailable
try:
    from kvpress import BasePress
    HAS_KVPRESS = True
except Exception:
    # kvpress may fail to import due to version incompatibilities
    HAS_KVPRESS = False
    # Define a minimal compatible base class
    class BasePress:
        """Minimal KVPress-compatible base when kvpress is not installed."""
        def __init__(self, compression_ratio: float = 0.5):
            self.compression_ratio = compression_ratio

        def __call__(self, model):
            return self  # Returns context manager

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass


def _compute_gini(weights: Tensor) -> float:
    """Gini coefficient of a 1D tensor."""
    w = weights.float().flatten()
    w = w[w > 0]
    if w.numel() < 2 or w.sum() < 1e-10:
        return 0.0
    sorted_w, _ = w.sort()
    n = sorted_w.numel()
    index = torch.arange(1, n + 1, dtype=torch.float32, device=w.device)
    return ((2 * (index * sorted_w).sum() - (n + 1) * sorted_w.sum()) / (n * sorted_w.sum())).item()


class LayerBudgetPress(BasePress):
    """KVPress-compatible press for joint per-layer token-precision optimization.

    Implements the LayerBudget algorithm:
    1. Profile attention sparsity (Gini coefficient) per layer
    2. Allocate per-layer (token_budget, quant_bits) via greedy marginal-gain
    3. Evict tokens and quantize per layer

    Args:
        compression_ratio: Fraction of memory to retain (e.g., 0.25 = 4x compression).
        available_bits: Quantization precision levels.
        sink_tokens: Always-retained initial tokens.
        recent_tokens: Always-retained final tokens.
        token_step: Granularity for token allocation.
        importance_k: Sigmoid steepness for layer importance.
        importance_tau: Sigmoid midpoint for layer importance.
        fidelity: Per-bit-width quality factors.
    """

    def __init__(
        self,
        compression_ratio: float = 0.5,
        available_bits: Optional[List[int]] = None,
        sink_tokens: int = 4,
        recent_tokens: int = 16,
        token_step: int = 8,
        importance_k: float = 5.0,
        importance_tau: float = 0.3,
        fidelity: Optional[Dict[int, float]] = None,
    ):
        super().__init__(compression_ratio=compression_ratio)
        self.available_bits = sorted(available_bits or [4, 8, 16])
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.token_step = token_step
        self.importance_k = importance_k
        self.importance_tau = importance_tau
        self.fidelity = fidelity or {16: 1.0, 8: 0.9999, 4: 0.9964}

    def compress_cache(
        self,
        past_key_values,
        attention_weights: Optional[List[Tensor]] = None,
    ):
        """Compress a HuggingFace DynamicCache in-place.

        Args:
            past_key_values: DynamicCache or list of (K, V) tuples.
                K shape: (batch, heads, seq_len, head_dim)
            attention_weights: Optional per-layer attention tensors.

        Returns:
            Compressed past_key_values (same type as input).
        """
        # Extract dimensions
        if hasattr(past_key_values, "key_cache"):
            # DynamicCache
            num_layers = len(past_key_values.key_cache)
            k0 = past_key_values.key_cache[0]
        else:
            num_layers = len(past_key_values)
            k0 = past_key_values[0][0]

        seq_len = k0.shape[2]
        num_heads = k0.shape[1]
        head_dim = k0.shape[3]
        device = k0.device
        dtype = k0.dtype

        # Step 1: Profile sparsity
        sparsity = self._profile_sparsity(past_key_values, attention_weights, num_layers)

        # Step 2: Compute importance weights
        importance = self._compute_importance(num_layers)

        # Step 3: Allocate
        full_mem = 2 * num_layers * seq_len * num_heads * head_dim * 2  # FP16
        budget = int(full_mem * self.compression_ratio)
        allocations = self._allocate(
            sparsity, importance, budget, seq_len, num_layers, num_heads, head_dim,
        )

        # Step 4: Apply compression per layer
        for l, (n_tokens, bits) in enumerate(allocations):
            if hasattr(past_key_values, "key_cache"):
                k = past_key_values.key_cache[l]  # (B, H, S, D)
                v = past_key_values.value_cache[l]
            else:
                k, v = past_key_values[l]

            # Token selection
            if n_tokens < seq_len:
                indices = self._select_tokens(k, v, n_tokens, seq_len, attention_weights, l)
                k = k[:, :, indices, :]
                v = v[:, :, indices, :]

            # Quantization (dequantize back for now — real impl would keep quantized)
            if bits < 16:
                k = self._quantize_dequantize(k, bits, axis="channel")
                v = self._quantize_dequantize(v, bits, axis="token")

            # Write back
            if hasattr(past_key_values, "key_cache"):
                past_key_values.key_cache[l] = k
                past_key_values.value_cache[l] = v
            else:
                past_key_values[l] = (k, v)

        return past_key_values

    def _profile_sparsity(self, past_kv, attn_weights, num_layers):
        """Extract per-layer Gini sparsity."""
        sparsity = {}
        for l in range(num_layers):
            if attn_weights and l < len(attn_weights):
                # Use last-token attention row
                last_row = attn_weights[l][0, :, -1, :].mean(dim=0)  # (S,)
                sparsity[l] = _compute_gini(last_row)
            else:
                # Fallback: estimate from value norms
                if hasattr(past_kv, "value_cache"):
                    v = past_kv.value_cache[l][0]  # (H, S, D)
                else:
                    v = past_kv[l][1][0]
                norms = v.float().norm(dim=-1).mean(dim=0)  # (S,)
                sparsity[l] = _compute_gini(norms)
        return sparsity

    def _compute_importance(self, num_layers):
        """Sigmoid importance weights."""
        importance = {}
        for l in range(num_layers):
            x = self.importance_k * (l / max(1, num_layers - 1) - self.importance_tau)
            importance[l] = 1.0 / (1.0 + math.exp(-x))
        return importance

    def _allocate(self, sparsity, importance, budget, seq_len, num_layers, num_heads, head_dim):
        """Greedy marginal-gain allocation."""
        min_bits = self.available_bits[0]
        n_min = min(self.sink_tokens + self.recent_tokens, seq_len)

        # Initialize at minimum
        alloc = [(n_min, min_bits) for _ in range(num_layers)]

        def mem_cost(n, b):
            return 2 * n * num_heads * head_dim * b // 8

        def quality(l, n, b):
            frac = n / max(1, seq_len)
            g = sparsity.get(l, 0.5)
            coverage = frac ** max(0.01, 1 - g)
            fidelity = self.fidelity.get(b, 0.9)
            return coverage * fidelity * importance.get(l, 0.5)

        used = sum(mem_cost(n, b) for n, b in alloc)

        while used < budget:
            best_gain = 0
            best_action = None

            for l in range(num_layers):
                n, b = alloc[l]

                # Try adding tokens
                if n + self.token_step <= seq_len:
                    nn = n + self.token_step
                    dq = quality(l, nn, b) - quality(l, n, b)
                    dm = mem_cost(nn, b) - mem_cost(n, b)
                    if dm > 0 and used + dm <= budget:
                        gpb = dq / dm
                        if gpb > best_gain:
                            best_gain = gpb
                            best_action = (l, nn, b, dm)

                # Try upgrading bits
                bi = self.available_bits.index(b)
                if bi < len(self.available_bits) - 1:
                    nb = self.available_bits[bi + 1]
                    dq = quality(l, n, nb) - quality(l, n, b)
                    dm = mem_cost(n, nb) - mem_cost(n, b)
                    if dm > 0 and used + dm <= budget:
                        gpb = dq / dm
                        if gpb > best_gain:
                            best_gain = gpb
                            best_action = (l, n, nb, dm)

            if best_action is None:
                break

            l, n_new, b_new, dm = best_action
            alloc[l] = (n_new, b_new)
            used += dm

        return alloc

    def _select_tokens(self, k, v, n_tokens, seq_len, attn_weights, layer_idx):
        """H2O-style token selection."""
        sink = min(self.sink_tokens, seq_len)
        recent = min(self.recent_tokens, max(0, seq_len - sink))
        middle_budget = max(0, n_tokens - sink - recent)

        sink_idx = torch.arange(sink, device=k.device)
        recent_start = max(sink, seq_len - recent)
        recent_idx = torch.arange(recent_start, seq_len, device=k.device)

        if middle_budget > 0 and recent_start > sink:
            if attn_weights and layer_idx < len(attn_weights):
                scores = attn_weights[layer_idx][0].sum(dim=0).sum(dim=0)  # (S,)
            else:
                scores = v[0].float().norm(dim=-1).mean(dim=0)  # (S,)
            mid_scores = scores[sink:recent_start]
            topk = min(middle_budget, len(mid_scores))
            _, top_idx = mid_scores.topk(topk)
            middle_idx = top_idx + sink
        else:
            middle_idx = torch.tensor([], dtype=torch.long, device=k.device)

        indices = torch.cat([sink_idx, middle_idx, recent_idx])
        indices = indices.unique()
        indices, _ = indices.sort()
        return indices[:n_tokens]

    @staticmethod
    def _quantize_dequantize(tensor: Tensor, bits: int, axis: str) -> Tensor:
        """Asymmetric quantization + dequantization."""
        if axis == "channel":
            # Per-channel (reduce over seq_len dim=2)
            reduce_dim = 2
        else:
            # Per-token (reduce over head_dim dim=3)
            reduce_dim = 3

        t = tensor.float()
        t_min = t.amin(dim=reduce_dim, keepdim=True)
        t_max = t.amax(dim=reduce_dim, keepdim=True)
        qmax = (1 << bits) - 1
        scale = (t_max - t_min) / max(qmax, 1)
        scale = scale.clamp(min=1e-8)
        quantized = ((t - t_min) / scale).round().clamp(0, qmax)
        dequantized = quantized * scale + t_min
        return dequantized.to(tensor.dtype)
