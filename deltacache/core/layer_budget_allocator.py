"""Per-layer (token_budget, quant_bits) allocator for KV cache.

Given per-layer sparsity profiles (Gini) and importance weights (sigmoid),
allocates a total memory budget across layers by solving:

    max  Σ_l  quality(n_l, b_l)
    s.t. Σ_l  memory(n_l, b_l) ≤ B

Using a greedy marginal-gain algorithm: repeatedly pick the (layer, action)
with the highest quality-gain-per-byte, where actions are either
"add more tokens" or "upgrade precision".

Key insight: high-sparsity layers benefit from more tokens at lower bits
(attention is concentrated, so keeping the right tokens matters more than
precision), while semantically important layers benefit from fewer tokens
at higher bits (fidelity matters more than coverage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class LayerAllocation:
    """Per-layer allocation result."""

    layer_idx: int
    token_budget: int  # Number of tokens to retain (n_l)
    quant_bits: int  # Quantization precision (b_l: 4, 8, or 16)
    memory_bytes: int  # Memory consumed by this layer

    @property
    def is_full_precision(self) -> bool:
        return self.quant_bits == 16

    @property
    def is_full_budget(self) -> bool:
        """True if this layer retains all tokens."""
        return self.token_budget >= self._max_tokens

    def __post_init__(self) -> None:
        self._max_tokens = self.token_budget  # Set by allocator


@dataclass
class AllocationResult:
    """Complete allocation across all layers."""

    allocations: List[LayerAllocation]
    total_memory_bytes: int
    budget_bytes: int
    budget_utilization: float  # fraction of budget used
    compression_ratio: float  # vs full FP16

    def get_allocation(self, layer_idx: int) -> LayerAllocation:
        return self.allocations[layer_idx]


def sigmoid_importance(
    layer_idx: int,
    num_layers: int,
    k: float = 5.0,
    tau: float = 0.3,
    invert: bool = True,
) -> float:
    """Compute sigmoid importance weight for a layer.

    By default (invert=True), early layers get higher weight. This reflects
    the empirical finding that early layers are the quality bottleneck under
    KV cache eviction (see Section 5.1 of the paper).

    With invert=False, later layers get higher weight (conventional).

    Args:
        layer_idx: Layer index (0-based).
        num_layers: Total number of layers.
        k: Steepness parameter.
        tau: Midpoint parameter (fraction of total layers).
        invert: If True (default), early layers get higher importance.

    Returns:
        Weight in (0, 1).
    """
    if num_layers <= 1:
        return 0.5
    pos = layer_idx / (num_layers - 1)
    if invert:
        pos = 1.0 - pos
    return 1.0 / (1.0 + math.exp(-k * (pos - tau)))


class LayerBudgetAllocator:
    """Greedy allocator for per-layer (token_budget, quant_bits).

    Solves the joint token-precision allocation problem using a
    greedy marginal-gain approach.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of KV heads per layer.
        head_dim: Dimension per head.
        available_bits: Allowed quantization levels.
        sink_tokens: Number of attention-sink tokens to always retain.
        recent_tokens: Number of recent tokens to always retain.
        token_step: Granularity for token budget increments.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        available_bits: Optional[List[int]] = None,
        sink_tokens: int = 4,
        recent_tokens: int = 16,
        token_step: int = 8,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.available_bits = sorted(available_bits or [4, 8, 16])
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.token_step = token_step

    def memory_cost(self, n_tokens: int, bits: int) -> int:
        """Memory in bytes for one layer with n tokens at given precision.

        KV cache per layer: 2 (K+V) × n_tokens × num_heads × head_dim × bits/8
        Plus quantization metadata overhead (~2% for INT8, ~4% for INT4).
        """
        base = 2 * n_tokens * self.num_heads * self.head_dim * bits // 8

        # Metadata overhead (scale + zero-point tensors)
        if bits < 16:
            # Per-channel key metadata + per-token value metadata
            key_meta = self.num_heads * self.head_dim * 4  # FP16 scale + zero
            value_meta = n_tokens * 4  # FP16 scale + zero
            base += key_meta + value_meta

        return base

    def full_memory(self, seq_len: int) -> int:
        """Total memory for full FP16 cache across all layers."""
        per_layer = self.memory_cost(seq_len, 16)
        return per_layer * self.num_layers

    def _quality_score(
        self,
        layer_idx: int,
        n_tokens: int,
        bits: int,
        seq_len: int,
        sparsity: Dict[int, float],
        importance: Dict[int, float],
    ) -> float:
        """Estimate quality contribution of a layer's allocation.

        Quality = coverage × fidelity × importance

        - Coverage: fraction of attention mass captured (approximated by
          token fraction, weighted by sparsity — sparse layers capture
          more mass with fewer tokens).
        - Fidelity: precision quality factor (16-bit=1.0, 8-bit=0.98, 4-bit=0.92).
        - Importance: sigmoid weight for this layer.
        """
        gini = sparsity.get(layer_idx, 0.5)
        imp = importance.get(layer_idx, 0.5)

        # Coverage model: retaining fraction f of tokens.
        # With zero-fill evaluation, evicted positions create zeros that
        # distort the attention softmax. The damage is worse than just
        # losing attention mass — we use a steeper penalty.
        # f^(1-g) models attention mass captured; we square it to penalize
        # aggressive eviction more heavily.
        frac = min(1.0, n_tokens / max(1, seq_len))
        exponent = max(0.01, 1.0 - gini)
        attention_mass = frac**exponent
        # Eviction penalty: squared coverage to discourage aggressive eviction
        coverage = attention_mass**2

        # Fidelity model: calibrated from cosine similarity measurements
        # on Mistral-7B (INT8: 0.9999, INT4: 0.9964).
        fidelity_map = {16: 1.0, 8: 0.9999, 4: 0.9964}
        fidelity = fidelity_map.get(bits, 0.99)

        return coverage * fidelity * imp

    def _marginal_quality_gain(
        self,
        layer_idx: int,
        current_tokens: int,
        current_bits: int,
        action: str,
        seq_len: int,
        sparsity: Dict[int, float],
        importance: Dict[int, float],
    ) -> Tuple[float, int]:
        """Compute quality-gain-per-byte for an action.

        Args:
            action: "add_tokens" or "upgrade_bits"

        Returns:
            (gain_per_byte, additional_bytes)
        """
        current_quality = self._quality_score(
            layer_idx,
            current_tokens,
            current_bits,
            seq_len,
            sparsity,
            importance,
        )
        current_memory = self.memory_cost(current_tokens, current_bits)

        if action == "add_tokens":
            new_tokens = min(seq_len, current_tokens + self.token_step)
            if new_tokens == current_tokens:
                return -1.0, 0
            new_quality = self._quality_score(
                layer_idx,
                new_tokens,
                current_bits,
                seq_len,
                sparsity,
                importance,
            )
            new_memory = self.memory_cost(new_tokens, current_bits)

        elif action == "upgrade_bits":
            bits_idx = self.available_bits.index(current_bits)
            if bits_idx >= len(self.available_bits) - 1:
                return -1.0, 0  # Already at max precision
            new_bits = self.available_bits[bits_idx + 1]
            new_quality = self._quality_score(
                layer_idx,
                current_tokens,
                new_bits,
                seq_len,
                sparsity,
                importance,
            )
            new_memory = self.memory_cost(current_tokens, new_bits)

        else:
            return -1.0, 0

        delta_quality = new_quality - current_quality
        delta_memory = new_memory - current_memory

        if delta_memory <= 0 or delta_quality <= 0:
            return -1.0, 0

        return delta_quality / delta_memory, delta_memory

    def allocate(
        self,
        sparsity: Dict[int, float],
        importance: Dict[int, float],
        budget_bytes: int,
        seq_len: int,
    ) -> AllocationResult:
        """Allocate per-layer (token_budget, quant_bits) within budget.

        Args:
            sparsity: {layer_idx: gini_coefficient} from LayerAttentionProfiler.
            importance: {layer_idx: importance_weight} (sigmoid or custom).
            budget_bytes: Total memory budget in bytes.
            seq_len: Sequence length.

        Returns:
            AllocationResult with per-layer allocations.
        """
        # Minimum tokens: sink + recent (always retained at all layers)
        min_tokens = self.sink_tokens + self.recent_tokens
        min_tokens = min(min_tokens, seq_len)
        min_bits = self.available_bits[0]  # Lowest precision

        # Initialize: every layer gets minimum allocation
        current_tokens = [min_tokens] * self.num_layers
        current_bits = [min_bits] * self.num_layers

        # Check if even minimum allocation exceeds budget
        total_used = sum(
            self.memory_cost(current_tokens[layer_i], current_bits[layer_i])
            for layer_i in range(self.num_layers)
        )

        if total_used > budget_bytes:
            # Budget is extremely tight — reduce tokens proportionally
            scale = budget_bytes / max(1, total_used)
            for layer_i in range(self.num_layers):
                current_tokens[layer_i] = max(1, int(min_tokens * scale))
            total_used = sum(
                self.memory_cost(current_tokens[layer_i], current_bits[layer_i])
                for layer_i in range(self.num_layers)
            )

        # Greedy allocation: repeatedly pick best (layer, action)
        max_iterations = self.num_layers * (
            (seq_len // self.token_step + 1) + len(self.available_bits)
        )

        for _ in range(max_iterations):
            remaining = budget_bytes - total_used
            if remaining <= 0:
                break

            best_gain = -1.0
            best_layer = -1
            best_action = ""
            best_cost = 0

            for layer_i in range(self.num_layers):
                for action in ("add_tokens", "upgrade_bits"):
                    gain, cost = self._marginal_quality_gain(
                        layer_i,
                        current_tokens[layer_i],
                        current_bits[layer_i],
                        action,
                        seq_len,
                        sparsity,
                        importance,
                    )
                    if gain > best_gain and cost <= remaining and cost > 0:
                        best_gain = gain
                        best_layer = layer_i
                        best_action = action
                        best_cost = cost

            if best_layer < 0 or best_gain <= 0:
                break  # No beneficial action fits

            # Apply best action
            if best_action == "add_tokens":
                current_tokens[best_layer] = min(
                    seq_len,
                    current_tokens[best_layer] + self.token_step,
                )
            elif best_action == "upgrade_bits":
                bits_idx = self.available_bits.index(current_bits[best_layer])
                current_bits[best_layer] = self.available_bits[bits_idx + 1]

            total_used += best_cost

        # Build result
        allocations = []
        actual_total = 0
        for layer_i in range(self.num_layers):
            mem = self.memory_cost(current_tokens[layer_i], current_bits[layer_i])
            alloc = LayerAllocation(
                layer_idx=layer_i,
                token_budget=current_tokens[layer_i],
                quant_bits=current_bits[layer_i],
                memory_bytes=mem,
            )
            alloc._max_tokens = seq_len
            allocations.append(alloc)
            actual_total += mem

        full_mem = self.full_memory(seq_len)

        return AllocationResult(
            allocations=allocations,
            total_memory_bytes=actual_total,
            budget_bytes=budget_bytes,
            budget_utilization=actual_total / max(1, budget_bytes),
            compression_ratio=full_mem / max(1, actual_total),
        )

    def allocate_uniform(
        self,
        token_fraction: float,
        bits: int,
        seq_len: int,
    ) -> AllocationResult:
        """Uniform allocation baseline (same budget/bits for all layers).

        Args:
            token_fraction: Fraction of tokens to keep (0-1).
            bits: Quantization precision for all layers.
            seq_len: Sequence length.

        Returns:
            AllocationResult.
        """
        n_tokens = max(1, int(seq_len * token_fraction))
        allocations = []
        total = 0

        for layer_i in range(self.num_layers):
            mem = self.memory_cost(n_tokens, bits)
            alloc = LayerAllocation(
                layer_idx=layer_i,
                token_budget=n_tokens,
                quant_bits=bits,
                memory_bytes=mem,
            )
            alloc._max_tokens = seq_len
            allocations.append(alloc)
            total += mem

        full_mem = self.full_memory(seq_len)

        return AllocationResult(
            allocations=allocations,
            total_memory_bytes=total,
            budget_bytes=total,
            budget_utilization=1.0,
            compression_ratio=full_mem / max(1, total),
        )

    @staticmethod
    def compute_importance_weights(
        num_layers: int,
        k: float = 5.0,
        tau: float = 0.3,
        invert: bool = True,
    ) -> Dict[int, float]:
        """Compute importance weights for all layers.

        By default (invert=True), early layers get higher importance,
        reflecting that they are the quality bottleneck under eviction.

        Args:
            num_layers: Total number of layers.
            k: Sigmoid steepness.
            tau: Sigmoid midpoint.
            invert: If True (default), early layers get higher importance.
        """
        return {
            layer_i: sigmoid_importance(layer_i, num_layers, k, tau, invert=invert)
            for layer_i in range(num_layers)
        }
