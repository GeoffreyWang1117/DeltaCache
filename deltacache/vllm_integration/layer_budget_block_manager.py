"""Per-layer block-level KV cache compression for vLLM PagedAttention.

Bridges LayerBudget's per-layer (token_budget, quant_bits) allocation
into vLLM's fixed-size block structure. Each layer independently decides
which blocks to retain, evict, or compress.

Key insight: vLLM uses fixed block sizes (typically 16 tokens). LayerBudget
token budgets map to block retention — round to block boundaries and evict
entire blocks per-layer. Quantize-dequantize is applied in-place to retained
blocks at the target precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from deltacache.core.layer_budget_allocator import (
    LayerBudgetAllocator,
    AllocationResult,
)
from deltacache.core.layer_profiler import LayerAttentionProfiler


@dataclass
class LayerBlockAllocation:
    """Per-layer block retention decisions."""

    # layer_idx -> (retained_block_indices, quant_bits)
    allocations: Dict[int, Tuple[List[int], int]]
    total_memory_bytes: int
    budget_bytes: int
    freed_block_indices: Dict[int, List[int]]  # layer_idx -> evicted block indices

    @property
    def compression_ratio(self) -> float:
        if self.total_memory_bytes == 0:
            return 1.0
        return self.budget_bytes / self.total_memory_bytes


class LayerBudgetBlockManager:
    """Manages per-layer block allocation with LayerBudget optimization.

    Bridges LayerBudget's per-layer (token_budget, quant_bits) decisions
    into vLLM's fixed-size block structure. Each layer independently
    decides which blocks to retain/evict/compress.

    Args:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV heads per layer.
        head_dim: Dimension per head.
        block_size: vLLM block size (tokens per block, typically 16).
        available_bits: Allowed quantization levels.
        sink_blocks: Always retain first N blocks (attention sinks).
        recent_blocks: Always retain last N blocks (recent context).
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        available_bits: Optional[List[int]] = None,
        sink_blocks: int = 1,
        recent_blocks: int = 2,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.available_bits = sorted(available_bits or [4, 8, 16])
        self.sink_blocks = sink_blocks
        self.recent_blocks = recent_blocks

        # Underlying token-level allocator (reuse existing logic)
        self._allocator = LayerBudgetAllocator(
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            available_bits=self.available_bits,
            sink_tokens=sink_blocks * block_size,
            recent_tokens=recent_blocks * block_size,
            token_step=block_size,  # Align steps to block boundaries
        )
        self._profiler = LayerAttentionProfiler()

    def block_memory_cost(self, bits: int = 16) -> int:
        """Memory in bytes for one block at one layer at given precision."""
        return 2 * self.block_size * self.num_kv_heads * self.head_dim * bits // 8

    def full_memory(self, num_blocks: int) -> int:
        """Total FP16 memory for all layers at given block count."""
        return num_blocks * self.num_layers * self.block_memory_cost(16)

    def profile_and_allocate(
        self,
        attention_weights: Optional[List[Tensor]],
        seq_len: int,
        budget_bytes: int,
    ) -> LayerBlockAllocation:
        """Profile attention sparsity and allocate per-layer block budgets.

        Args:
            attention_weights: Per-layer attention tensors from prefill.
                Each shape (batch, heads, seq_len, seq_len).
                If None, uses uniform sparsity estimate.
            seq_len: Current sequence length.
            budget_bytes: Total memory budget in bytes.

        Returns:
            LayerBlockAllocation with per-layer block decisions.
        """
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Profile sparsity
        if attention_weights is not None:
            profile = self._profiler.profile_from_attention_weights(attention_weights)
            sparsity = profile.gini_scores()
        else:
            sparsity = {l: 0.5 for l in range(self.num_layers)}

        importance = self._allocator.compute_importance_weights(self.num_layers)

        # Allocate at token level (step size = block_size ensures alignment)
        token_alloc = self._allocator.allocate(sparsity, importance, budget_bytes, seq_len)

        # Convert token allocations to block allocations
        allocations: Dict[int, Tuple[List[int], int]] = {}
        freed: Dict[int, List[int]] = {}
        total_mem = 0

        for la in token_alloc.allocations:
            layer_idx = la.layer_idx
            token_budget = la.token_budget
            bits = la.quant_bits

            # Round token budget to block boundary
            block_budget = (token_budget + self.block_size - 1) // self.block_size
            block_budget = min(block_budget, num_blocks)

            if block_budget >= num_blocks:
                # Keep all blocks
                retained = list(range(num_blocks))
                evicted = []
            else:
                # Select which blocks to retain using importance
                retained, evicted = self._select_blocks(
                    layer_idx, num_blocks, block_budget,
                    attention_weights, seq_len,
                )

            allocations[layer_idx] = (retained, bits)
            freed[layer_idx] = evicted
            total_mem += len(retained) * self.block_memory_cost(bits)

        return LayerBlockAllocation(
            allocations=allocations,
            total_memory_bytes=total_mem,
            budget_bytes=budget_bytes,
            freed_block_indices=freed,
        )

    def _select_blocks(
        self,
        layer_idx: int,
        num_blocks: int,
        block_budget: int,
        attention_weights: Optional[List[Tensor]],
        seq_len: int,
    ) -> Tuple[List[int], List[int]]:
        """Select which blocks to retain for a layer.

        Always retains sink blocks (first N) and recent blocks (last N).
        Remaining budget filled by block importance (attention mass or uniform).

        Returns:
            (retained_block_indices, evicted_block_indices)
        """
        # Protected blocks
        sink_end = min(self.sink_blocks, num_blocks)
        recent_start = max(sink_end, num_blocks - self.recent_blocks)

        sink_indices = set(range(sink_end))
        recent_indices = set(range(recent_start, num_blocks))
        protected = sink_indices | recent_indices

        mid_budget = max(0, block_budget - len(protected))
        mid_candidates = [i for i in range(sink_end, recent_start)]

        if mid_budget >= len(mid_candidates):
            # Can keep all middle blocks
            retained = sorted(protected | set(mid_candidates))
        elif mid_budget > 0 and mid_candidates:
            # Select top-importance middle blocks
            if attention_weights is not None and layer_idx < len(attention_weights):
                importance = self.compute_block_importance(
                    layer_idx, attention_weights[layer_idx],
                    self.block_size, num_blocks,
                )
                mid_scores = [(i, importance[i].item()) for i in mid_candidates]
            else:
                # Uniform fallback — prefer blocks closer to recent context
                mid_scores = [(i, i / num_blocks) for i in mid_candidates]

            mid_scores.sort(key=lambda x: x[1], reverse=True)
            selected_mid = set(i for i, _ in mid_scores[:mid_budget])
            retained = sorted(protected | selected_mid)
        else:
            retained = sorted(protected)

        retained = retained[:block_budget]
        evicted = [i for i in range(num_blocks) if i not in set(retained)]
        return retained, evicted

    def compute_block_importance(
        self,
        layer_idx: int,
        attention_weights: Tensor,
        block_size: int,
        num_blocks: int,
    ) -> Tensor:
        """Compute per-block importance scores from attention weights.

        Aggregates token-level attention into block-level scores by
        summing attention mass from the last-token row within each block.

        Args:
            layer_idx: Layer index (unused, for future per-layer logic).
            attention_weights: Attention tensor, shape (batch, heads, seq, seq).
            block_size: Tokens per block.
            num_blocks: Total number of blocks.

        Returns:
            (num_blocks,) tensor of importance scores.
        """
        # Extract last-token attention row, average over heads
        # Shape: (seq_len,)
        last_row = attention_weights[0, :, -1, :].mean(dim=0).detach()
        seq_len = last_row.shape[0]

        scores = torch.zeros(num_blocks, device=last_row.device)
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, seq_len)
            if start < seq_len:
                scores[b] = last_row[start:end].sum()

        return scores

    def apply_compression(
        self,
        gpu_cache: List[Tensor],
        allocation: LayerBlockAllocation,
    ) -> None:
        """Apply LayerBudget compression to vLLM block cache in-place.

        For each layer:
        1. Zero-fill blocks not in the retained set
        2. Apply quantize-dequantize to retained blocks at target precision

        Args:
            gpu_cache: Per-layer cache tensors.
                vLLM format: [2, num_blocks, block_size, num_kv_heads, head_dim]
            allocation: Block allocation from profile_and_allocate().
        """
        for layer_idx, (retained, bits) in allocation.allocations.items():
            if layer_idx >= len(gpu_cache):
                continue

            cache = gpu_cache[layer_idx]  # [2, num_blocks, block_size, H, D]
            num_blocks = cache.shape[1]
            retained_set = set(retained)

            # Zero-fill evicted blocks
            for b in range(num_blocks):
                if b not in retained_set:
                    cache[:, b, :, :, :] = 0

            # QDQ retained blocks at target precision
            if bits < 16:
                for b in retained:
                    if b < num_blocks:
                        cache[0, b] = self._qdq(cache[0, b], bits)  # keys
                        cache[1, b] = self._qdq(cache[1, b], bits)  # values

    @staticmethod
    def _qdq(t: Tensor, bits: int) -> Tensor:
        """Quantize-dequantize a tensor (simulated, preserves shape/dtype)."""
        tf = t.float()
        qmax = (1 << bits) - 1
        t_min = tf.amin(dim=-1, keepdim=True)
        t_max = tf.amax(dim=-1, keepdim=True)
        scale = ((t_max - t_min) / qmax).clamp(min=1e-8)
        q = ((tf - t_min) / scale).round().clamp(0, qmax)
        return (q * scale + t_min).to(t.dtype)
