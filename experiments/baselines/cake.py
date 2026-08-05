"""CAKE baseline reproduction (ICLR 2025, arXiv:2503.12491).

CAKE (Cache-Aware KV Eviction) uses per-layer token budget allocation
based on a preference score combining attention entropy and temporal
variance, with cascading prefill management.

Key ideas:
  1. Preference score: P = H^(1/τ1) × V^(1/τ2) where H = attention entropy,
     V = temporal variance of attention scores.
  2. Higher preference → layer gets larger token budget.
  3. Cascading prefill: process prefill in chunks, applying eviction between chunks.
  4. Token selection: heavy-hitter (H2O-style) within each layer.

This is a simplified reproduction for comparison purposes — we implement
the core per-layer budget allocation and token selection, without the
full cascading prefill pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class CAKELayerProfile:
    """CAKE preference metrics for a single layer."""

    layer_idx: int
    entropy: float  # Attention entropy H
    temporal_variance: float  # Variance of attention across positions V
    preference: float  # P = H^(1/τ1) × V^(1/τ2)


@dataclass
class CAKEAllocation:
    """Per-layer allocation from CAKE."""

    layer_idx: int
    token_budget: int
    memory_bytes: int


class CAKEBaseline:
    """CAKE per-layer eviction baseline.

    Implements per-layer token budget allocation based on the CAKE
    preference score, with H2O-style token selection within each layer.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of KV heads.
        head_dim: Head dimension.
        tau1: Temperature for entropy component.
        tau2: Temperature for variance component.
        sink_tokens: Number of sink tokens to always retain.
        recent_tokens: Number of recent tokens to always retain.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        tau1: float = 1.0,
        tau2: float = 1.0,
        sink_tokens: int = 4,
        recent_tokens: int = 16,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.tau1 = tau1
        self.tau2 = tau2
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens

    def compute_preference(
        self,
        attention_weights: List[Tensor],
    ) -> List[CAKELayerProfile]:
        """Compute CAKE preference score for each layer.

        Args:
            attention_weights: Per-layer attention tensors,
                each shape (batch, heads, seq_len, seq_len).

        Returns:
            List of CAKELayerProfile with preference scores.
        """
        profiles = []

        for layer_idx, attn in enumerate(attention_weights):
            # Use last-token attention row: (heads, seq_len)
            last_row = attn[0, :, -1, :]  # (heads, seq_len)

            # Average over heads
            avg_attn = last_row.mean(dim=0)  # (seq_len,)

            # Attention entropy H
            clamped = avg_attn.clamp(min=1e-10)
            entropy = -(clamped * clamped.log()).sum().item()
            seq_len = avg_attn.numel()
            max_entropy = math.log(seq_len) if seq_len > 1 else 1.0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Temporal variance V: variance of per-head attention distributions
            # Higher variance = more diverse attention patterns across heads
            per_head_entropy = []
            for h in range(last_row.shape[0]):
                head_attn = last_row[h].clamp(min=1e-10)
                h_entropy = -(head_attn * head_attn.log()).sum().item()
                per_head_entropy.append(h_entropy / max_entropy if max_entropy > 0 else 0.0)

            if len(per_head_entropy) > 1:
                mean_h = sum(per_head_entropy) / len(per_head_entropy)
                variance = sum((x - mean_h) ** 2 for x in per_head_entropy) / len(per_head_entropy)
            else:
                variance = 0.0

            # Preference score: P = H^(1/τ1) × V^(1/τ2)
            h_component = max(1e-10, norm_entropy) ** (1.0 / self.tau1)
            v_component = max(1e-10, variance) ** (1.0 / self.tau2)
            preference = h_component * v_component

            profiles.append(CAKELayerProfile(
                layer_idx=layer_idx,
                entropy=norm_entropy,
                temporal_variance=variance,
                preference=preference,
            ))

        return profiles

    def allocate_budgets(
        self,
        profiles: List[CAKELayerProfile],
        total_token_budget: int,
        seq_len: int,
    ) -> List[CAKEAllocation]:
        """Allocate per-layer token budgets proportional to preference.

        Args:
            profiles: CAKE preference profiles per layer.
            total_token_budget: Total tokens to distribute across layers.
            seq_len: Sequence length.

        Returns:
            Per-layer allocations.
        """
        min_tokens = self.sink_tokens + self.recent_tokens
        min_tokens = min(min_tokens, seq_len)

        # Normalize preferences
        total_pref = sum(p.preference for p in profiles)
        if total_pref < 1e-10:
            # Fallback: uniform
            per_layer = max(min_tokens, total_token_budget // self.num_layers)
            return [
                CAKEAllocation(
                    layer_idx=l,
                    token_budget=min(per_layer, seq_len),
                    memory_bytes=self._memory_cost(min(per_layer, seq_len)),
                )
                for l in range(self.num_layers)
            ]

        allocations = []
        allocated = 0

        for profile in profiles:
            frac = profile.preference / total_pref
            n_tokens = int(frac * total_token_budget)
            n_tokens = max(min_tokens, min(n_tokens, seq_len))
            allocated += n_tokens

            allocations.append(CAKEAllocation(
                layer_idx=profile.layer_idx,
                token_budget=n_tokens,
                memory_bytes=self._memory_cost(n_tokens),
            ))

        # Redistribute remaining budget
        remaining = total_token_budget - allocated
        if remaining > 0:
            # Give to highest-preference layers
            sorted_idx = sorted(
                range(len(profiles)),
                key=lambda i: profiles[i].preference,
                reverse=True,
            )
            for i in sorted_idx:
                if remaining <= 0:
                    break
                add = min(remaining, seq_len - allocations[i].token_budget)
                allocations[i].token_budget += add
                allocations[i].memory_bytes = self._memory_cost(allocations[i].token_budget)
                remaining -= add

        return allocations

    def select_tokens(
        self,
        keys: Tensor,
        values: Tensor,
        attention_weights: Tensor,
        n_tokens: int,
        seq_len: int,
    ) -> Tensor:
        """H2O-style token selection within a layer.

        Args:
            keys: (1, seq_len, H, D)
            values: (1, seq_len, H, D)
            attention_weights: (1, heads, seq_len, seq_len)
            n_tokens: Number of tokens to select.
            seq_len: Total sequence length.

        Returns:
            Sorted tensor of selected token indices.
        """
        if n_tokens >= seq_len:
            return torch.arange(seq_len)

        # Cumulative attention (H2O): sum of attention received by each position
        # across all query positions
        cumulative = attention_weights[0].sum(dim=0).sum(dim=0).cpu()  # (seq_len,)

        sink = min(self.sink_tokens, seq_len)
        recent = min(self.recent_tokens, seq_len - sink)
        middle_budget = max(0, n_tokens - sink - recent)

        sink_idx = torch.arange(sink)
        recent_start = max(sink, seq_len - recent)
        recent_idx = torch.arange(recent_start, seq_len)

        if middle_budget > 0 and recent_start > sink:
            middle_scores = cumulative[sink:recent_start]
            _, top_idx = middle_scores.topk(min(middle_budget, len(middle_scores)))
            middle_idx = top_idx + sink
        else:
            middle_idx = torch.tensor([], dtype=torch.long)

        all_idx = torch.cat([sink_idx, middle_idx, recent_idx])
        all_idx = all_idx.unique()
        all_idx, _ = all_idx.sort()
        return all_idx[:n_tokens]

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        attention_weights: List[Tensor],
        compression_ratio: float,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        """Full CAKE compression pipeline.

        Args:
            full_keys: (num_layers, seq_len, H, D)
            full_values: Same shape.
            attention_weights: Per-layer attention tensors.
            compression_ratio: Target compression (e.g., 3.0 = 33% memory).

        Returns:
            List of (keys, values, indices) per layer.
        """
        seq_len = full_keys.shape[1]

        # Compute preferences
        profiles = self.compute_preference(attention_weights)

        # Total token budget across all layers
        total_tokens_full = seq_len * self.num_layers
        total_budget = int(total_tokens_full / compression_ratio)

        # Allocate per-layer
        allocations = self.allocate_budgets(profiles, total_budget, seq_len)

        # Select tokens per layer
        results = []
        for alloc in allocations:
            l = alloc.layer_idx
            layer_keys = full_keys[l:l+1]
            layer_values = full_values[l:l+1]

            if l < len(attention_weights):
                layer_attn = attention_weights[l]
            else:
                layer_attn = None

            if layer_attn is not None:
                indices = self.select_tokens(
                    layer_keys, layer_values, layer_attn,
                    alloc.token_budget, seq_len,
                )
            else:
                indices = torch.arange(min(alloc.token_budget, seq_len))

            selected_k = layer_keys[:, indices.long(), :, :]
            selected_v = layer_values[:, indices.long(), :, :]
            results.append((selected_k, selected_v, indices))

        return results

    def _memory_cost(self, n_tokens: int) -> int:
        """FP16 memory for one layer."""
        return 2 * n_tokens * self.num_heads * self.head_dim * 2  # FP16 = 2 bytes
