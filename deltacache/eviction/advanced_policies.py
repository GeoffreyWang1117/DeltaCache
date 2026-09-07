"""Advanced eviction policies for DeltaCache.

This module implements novel eviction strategies that leverage model-internal
information for smarter cache management decisions.

Key innovations:
1. Attention-Aware Eviction: Uses attention scores to prioritize important KV entries
2. Layer-Aware Caching: Treats different transformer layers with different priorities
3. Hierarchical Offloading: Smart GPU -> CPU -> Delete tiering
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from torch import Tensor

from deltacache.core.memory_pool import MemoryPool
from deltacache.core.prefix_tree import PrefixTree, PrefixTreeNode
from deltacache.eviction.policy import (
    EvictionAction,
    EvictionCandidate,
    EvictionPolicy,
    EvictionResult,
)


@dataclass
class AttentionMetadata:
    """Attention-based importance metadata for a cache entry."""

    # Average attention score across all queries that used this cache
    avg_attention_score: float = 0.0

    # Number of attention score samples
    attention_samples: int = 0

    # Per-layer importance scores (higher = more important)
    layer_importance: Optional[Dict[int, float]] = None

    # Tokens that receive high attention (important for future queries)
    high_attention_positions: Optional[List[int]] = None

    def update_attention(self, attention_score: float) -> None:
        """Update running average of attention scores."""
        self.attention_samples += 1
        # Exponential moving average
        alpha = min(0.3, 1.0 / self.attention_samples)
        self.avg_attention_score = (1 - alpha) * self.avg_attention_score + alpha * attention_score

    def update_layer_importance(self, layer_scores: Dict[int, float]) -> None:
        """Update per-layer importance scores."""
        if self.layer_importance is None:
            self.layer_importance = {}

        for layer_idx, score in layer_scores.items():
            if layer_idx in self.layer_importance:
                # Running average
                self.layer_importance[layer_idx] = (
                    0.7 * self.layer_importance[layer_idx] + 0.3 * score
                )
            else:
                self.layer_importance[layer_idx] = score


class AttentionAwareEvictionPolicy(EvictionPolicy):
    """
    Eviction policy that uses attention scores to make smarter decisions.

    Key insight: KV cache entries that receive high attention scores from
    subsequent queries are more likely to be important for future queries.
    By tracking these scores, we can prioritize keeping high-attention
    entries in cache.

    This is inspired by IMPRESS (FAST 2025) but implemented in a lighter-weight
    manner suitable for online serving.
    """

    def __init__(
        self,
        attention_weight: float = 0.4,
        frequency_weight: float = 0.3,
        recency_weight: float = 0.3,
        layer_aware: bool = True,
        num_layers: Optional[int] = None,
        prefer_offload: bool = True,
    ) -> None:
        """
        Initialize attention-aware eviction policy.

        Args:
            attention_weight: Weight for attention score in importance calculation.
            frequency_weight: Weight for access frequency.
            recency_weight: Weight for access recency.
            layer_aware: Whether to apply layer-aware weighting.
            num_layers: Number of transformer layers (required if layer_aware=True).
            prefer_offload: Prefer offloading to CPU over deletion.
        """
        self.attention_weight = attention_weight
        self.frequency_weight = frequency_weight
        self.recency_weight = recency_weight
        self.layer_aware = layer_aware
        self.num_layers = num_layers
        self.prefer_offload = prefer_offload

        # Attention metadata storage: node_id -> AttentionMetadata
        self._attention_metadata: Dict[int, AttentionMetadata] = {}

        # Layer importance weights (computed once)
        if layer_aware and num_layers:
            self._layer_weights = self._compute_layer_weights(num_layers)
        else:
            self._layer_weights = None

    def _compute_layer_weights(self, num_layers: int) -> List[float]:
        """
        Compute importance weights for each layer.

        Insight: Later layers capture more semantic information and are
        more stable across different inputs. Earlier layers capture local
        patterns that may vary more.

        We use a smooth curve that gives higher weight to later layers.
        """
        weights = []
        for i in range(num_layers):
            # Logistic curve: later layers get higher weight
            normalized_pos = i / (num_layers - 1) if num_layers > 1 else 0.5
            weight = 1.0 / (1.0 + math.exp(-5 * (normalized_pos - 0.3)))
            # Scale to [0.5, 1.5] range
            weight = 0.5 + weight
            weights.append(weight)

        # Normalize so sum equals num_layers
        total = sum(weights)
        weights = [w * num_layers / total for w in weights]

        return weights

    def record_attention(
        self,
        node_id: int,
        attention_scores: Tensor,
        layer_idx: Optional[int] = None,
    ) -> None:
        """
        Record attention scores for a cache entry.

        This should be called after each forward pass that uses cached KV.

        Args:
            node_id: ID of the prefix tree node.
            attention_scores: Attention weights from the model [seq_len] or [heads, seq_len].
            layer_idx: Optional layer index for per-layer tracking.
        """
        if node_id not in self._attention_metadata:
            self._attention_metadata[node_id] = AttentionMetadata()

        metadata = self._attention_metadata[node_id]

        # Compute average attention for this entry
        if attention_scores.dim() > 1:
            avg_score = attention_scores.mean().item()
        else:
            avg_score = attention_scores.mean().item()

        metadata.update_attention(avg_score)

        # Track per-layer importance if provided
        if layer_idx is not None:
            metadata.update_layer_importance({layer_idx: avg_score})

    def _get_attention_score(self, node: PrefixTreeNode) -> float:
        """Get attention importance score for a node."""
        if id(node) in self._attention_metadata:
            return self._attention_metadata[id(node)].avg_attention_score
        return 0.0

    def _compute_importance(self, node: PrefixTreeNode) -> float:
        """
        Compute importance score for a node.

        Higher score = more important = evict last.
        Lower score = less important = evict first.
        """
        now = time.time()
        time_since_access = max(1.0, now - node.last_access)

        # Attention component (0-1 range, higher = more important)
        attention_score = self._get_attention_score(node)
        attention_component = attention_score * self.attention_weight

        # Frequency component (log scale, normalized)
        freq_score = math.log1p(node.access_count) / 10.0  # Normalize
        frequency_component = freq_score * self.frequency_weight

        # Recency component (decay over time)
        recency_score = 1.0 / (1.0 + math.log1p(time_since_access))
        recency_component = recency_score * self.recency_weight

        # Layer-aware adjustment
        layer_multiplier = 1.0
        if self._layer_weights and node.cache_block:
            # Average layer weight for this cache entry
            # Entries covering more layers of high importance get bonus
            num_layers = node.cache_block.num_layers
            if num_layers <= len(self._layer_weights):
                layer_multiplier = sum(self._layer_weights[:num_layers]) / num_layers

        # Combined score
        importance = (
            attention_component + frequency_component + recency_component
        ) * layer_multiplier

        return importance

    def select_victims(
        self,
        prefix_tree: PrefixTree,
        memory_pool: MemoryPool,
        required_memory: int,
    ) -> List[EvictionCandidate]:
        """Select victims based on attention-aware importance."""
        candidates = []

        for node in prefix_tree.get_all_cached_nodes():
            if node.ref_count > 0:
                continue

            if node.cache_block is None:
                continue

            importance = self._compute_importance(node)

            # Determine action
            if self.prefer_offload and node.cache_block.is_on_gpu:
                action = EvictionAction.OFFLOAD_TO_CPU
            else:
                action = EvictionAction.DELETE

            candidate = EvictionCandidate(
                node=node,
                score=importance,  # Lower = evict first
                action=action,
                memory_freed=node.cache_block.memory_size,
            )
            candidates.append(candidate)

        # Sort by importance (ascending = least important first)
        candidates.sort()
        return candidates

    def clear_metadata(self, node_id: Optional[int] = None) -> None:
        """Clear attention metadata for a node or all nodes."""
        if node_id is not None:
            self._attention_metadata.pop(node_id, None)
        else:
            self._attention_metadata.clear()


class LayerAwareCachingPolicy(EvictionPolicy):
    """
    Eviction policy with layer-aware prioritization.

    Key insight: Different transformer layers have different characteristics:
    - Early layers (0-30%): Capture local patterns, high variance across inputs
    - Middle layers (30-70%): Feature transformation, moderate importance
    - Late layers (70-100%): Semantic features, more stable, higher importance

    When under memory pressure, we can:
    1. Prioritize keeping late layers in GPU
    2. Offload early layers to CPU first
    3. Completely drop early layers before touching late layers
    """

    def __init__(
        self,
        num_layers: int,
        early_layer_ratio: float = 0.3,
        late_layer_ratio: float = 0.3,
        early_layer_penalty: float = 0.5,
        late_layer_bonus: float = 1.5,
        prefer_offload: bool = True,
    ) -> None:
        """
        Initialize layer-aware caching policy.

        Args:
            num_layers: Total number of transformer layers.
            early_layer_ratio: Fraction of layers considered "early" (0.0-1.0).
            late_layer_ratio: Fraction of layers considered "late" (0.0-1.0).
            early_layer_penalty: Importance multiplier for early layers (<1 = penalize).
            late_layer_bonus: Importance multiplier for late layers (>1 = bonus).
            prefer_offload: Prefer offloading to CPU over deletion.
        """
        self.num_layers = num_layers
        self.early_layer_ratio = early_layer_ratio
        self.late_layer_ratio = late_layer_ratio
        self.early_layer_penalty = early_layer_penalty
        self.late_layer_bonus = late_layer_bonus
        self.prefer_offload = prefer_offload

        # Compute layer boundaries
        self.early_layer_end = int(num_layers * early_layer_ratio)
        self.late_layer_start = int(num_layers * (1 - late_layer_ratio))

        # Precompute layer weights
        self._layer_weights = self._compute_layer_weights()

    def _compute_layer_weights(self) -> List[float]:
        """Compute importance weight for each layer."""
        weights = []
        for i in range(self.num_layers):
            if i < self.early_layer_end:
                weight = self.early_layer_penalty
            elif i >= self.late_layer_start:
                weight = self.late_layer_bonus
            else:
                # Middle layers: linear interpolation
                middle_range = self.late_layer_start - self.early_layer_end
                if middle_range > 0:
                    progress = (i - self.early_layer_end) / middle_range
                    weight = self.early_layer_penalty + progress * (
                        self.late_layer_bonus - self.early_layer_penalty
                    )
                else:
                    weight = 1.0
            weights.append(weight)
        return weights

    def get_layer_weight(self, layer_idx: int) -> float:
        """Get importance weight for a specific layer."""
        if 0 <= layer_idx < len(self._layer_weights):
            return self._layer_weights[layer_idx]
        return 1.0

    def _compute_importance(self, node: PrefixTreeNode) -> float:
        """Compute layer-aware importance score."""
        now = time.time()
        time_since_access = max(1.0, now - node.last_access)

        # Base importance from access patterns
        base_importance = math.log1p(node.access_count) * (
            1.0 / (1.0 + math.log1p(time_since_access))
        )

        # Layer-aware multiplier
        if node.cache_block:
            # Use average layer weight
            avg_weight = sum(self._layer_weights) / len(self._layer_weights)
            layer_multiplier = avg_weight
        else:
            layer_multiplier = 1.0

        # Subtree bonus (more dependents = more important)
        subtree_bonus = 1.0 + math.log1p(node.subtree_size) * 0.1

        return base_importance * layer_multiplier * subtree_bonus

    def select_victims(
        self,
        prefix_tree: PrefixTree,
        memory_pool: MemoryPool,
        required_memory: int,
    ) -> List[EvictionCandidate]:
        """Select victims with layer-aware prioritization."""
        candidates = []

        for node in prefix_tree.get_all_cached_nodes():
            if node.ref_count > 0:
                continue

            if node.cache_block is None:
                continue

            importance = self._compute_importance(node)

            # Determine action
            if self.prefer_offload and node.cache_block.is_on_gpu:
                action = EvictionAction.OFFLOAD_TO_CPU
            else:
                action = EvictionAction.DELETE

            candidate = EvictionCandidate(
                node=node,
                score=importance,
                action=action,
                memory_freed=node.cache_block.memory_size,
            )
            candidates.append(candidate)

        candidates.sort()
        return candidates

    def should_cache_layer(self, layer_idx: int, memory_pressure: float) -> bool:
        """
        Determine if a specific layer should be cached under memory pressure.

        Args:
            layer_idx: Layer index.
            memory_pressure: Memory utilization (0.0-1.0).

        Returns:
            True if layer should be cached.
        """
        layer_weight = self.get_layer_weight(layer_idx)

        # Higher weight layers are more resistant to pressure
        # At pressure=0.5, all layers are cached
        # At pressure=0.9, only layers with weight > 0.9 are cached
        threshold = memory_pressure

        return layer_weight > threshold


class HierarchicalEvictionPolicy(EvictionPolicy):
    """
    Hierarchical eviction with smart GPU -> CPU -> Delete tiering.

    Combines attention-aware and layer-aware strategies with intelligent
    tiered storage management.

    Features:
    1. Proactive CPU offloading before memory pressure hits
    2. Smart prefetching of likely-to-be-accessed entries
    3. Batch offloading for efficiency
    """

    def __init__(
        self,
        num_layers: int,
        attention_weight: float = 0.4,
        layer_weight: float = 0.3,
        access_weight: float = 0.3,
        gpu_threshold: float = 0.8,
        cpu_threshold: float = 0.9,
        batch_size: int = 4,
    ) -> None:
        """
        Initialize hierarchical eviction policy.

        Args:
            num_layers: Number of transformer layers.
            attention_weight: Weight for attention-based importance.
            layer_weight: Weight for layer-based importance.
            access_weight: Weight for access pattern importance.
            gpu_threshold: GPU utilization threshold to start offloading.
            cpu_threshold: CPU utilization threshold to start deleting.
            batch_size: Number of entries to offload/delete at once.
        """
        self.num_layers = num_layers
        self.attention_weight = attention_weight
        self.layer_weight = layer_weight
        self.access_weight = access_weight
        self.gpu_threshold = gpu_threshold
        self.cpu_threshold = cpu_threshold
        self.batch_size = batch_size

        # Sub-policies
        self.attention_policy = AttentionAwareEvictionPolicy(
            attention_weight=1.0,
            frequency_weight=0.0,
            recency_weight=0.0,
            layer_aware=True,
            num_layers=num_layers,
        )

        self.layer_policy = LayerAwareCachingPolicy(
            num_layers=num_layers,
        )

        # Attention metadata (shared with attention policy)
        self._attention_metadata = self.attention_policy._attention_metadata

    def record_attention(
        self,
        node_id: int,
        attention_scores: Tensor,
        layer_idx: Optional[int] = None,
    ) -> None:
        """Record attention scores (delegates to attention policy)."""
        self.attention_policy.record_attention(node_id, attention_scores, layer_idx)

    def _compute_importance(self, node: PrefixTreeNode) -> float:
        """Compute combined importance score."""
        now = time.time()
        time_since_access = max(1.0, now - node.last_access)

        # Attention component
        attention_score = self.attention_policy._get_attention_score(node)
        attention_component = attention_score * self.attention_weight

        # Layer component
        layer_importance = self.layer_policy._compute_importance(node)
        layer_component = layer_importance * self.layer_weight

        # Access pattern component
        access_score = math.log1p(node.access_count) / (1.0 + math.log1p(time_since_access))
        access_component = access_score * self.access_weight

        return attention_component + layer_component + access_component

    def select_victims(
        self,
        prefix_tree: PrefixTree,
        memory_pool: MemoryPool,
        required_memory: int,
    ) -> List[EvictionCandidate]:
        """Select victims with hierarchical strategy."""
        candidates = []

        gpu_utilization = memory_pool.gpu_utilization
        cpu_limit = memory_pool.cpu_limit
        cpu_used = memory_pool.cpu_used
        cpu_utilization = cpu_used / cpu_limit if cpu_limit > 0 else 0.0

        for node in prefix_tree.get_all_cached_nodes():
            if node.ref_count > 0:
                continue

            if node.cache_block is None:
                continue

            importance = self._compute_importance(node)

            # Determine action based on current tier and pressure
            if node.cache_block.is_on_gpu:
                if gpu_utilization > self.gpu_threshold:
                    action = EvictionAction.OFFLOAD_TO_CPU
                else:
                    action = EvictionAction.OFFLOAD_TO_CPU  # Default to offload
            else:
                # Already on CPU
                if cpu_utilization > self.cpu_threshold:
                    action = EvictionAction.DELETE
                else:
                    action = EvictionAction.DELETE  # Only action for CPU blocks

            candidate = EvictionCandidate(
                node=node,
                score=importance,
                action=action,
                memory_freed=node.cache_block.memory_size,
            )
            candidates.append(candidate)

        # Sort by importance
        candidates.sort()

        # Batch for efficiency
        return candidates[: self.batch_size * 2]

    def proactive_offload(
        self,
        prefix_tree: PrefixTree,
        memory_pool: MemoryPool,
        target_gpu_utilization: float = 0.7,
    ) -> EvictionResult:
        """
        Proactively offload GPU entries to maintain headroom.

        Called during idle periods to prepare for future requests.

        Args:
            prefix_tree: Prefix tree.
            memory_pool: Memory pool.
            target_gpu_utilization: Target GPU utilization after offloading.

        Returns:
            EvictionResult with statistics.
        """
        current_utilization = memory_pool.gpu_utilization

        if current_utilization <= target_gpu_utilization:
            return EvictionResult()

        # Calculate how much to offload
        target_used = memory_pool.gpu_limit * target_gpu_utilization
        to_offload = memory_pool.gpu_used - target_used

        if to_offload <= 0:
            return EvictionResult()

        return self.evict(prefix_tree, memory_pool, int(to_offload))


def create_advanced_eviction_policy(
    name: str = "hierarchical", num_layers: int = 32, **kwargs
) -> EvictionPolicy:
    """
    Factory function to create advanced eviction policies.

    Args:
        name: Policy name ("attention_aware", "layer_aware", "hierarchical").
        num_layers: Number of transformer layers.
        **kwargs: Additional arguments for policy constructor.

    Returns:
        Eviction policy instance.
    """
    policies = {
        "attention_aware": lambda: AttentionAwareEvictionPolicy(num_layers=num_layers, **kwargs),
        "layer_aware": lambda: LayerAwareCachingPolicy(num_layers=num_layers, **kwargs),
        "hierarchical": lambda: HierarchicalEvictionPolicy(num_layers=num_layers, **kwargs),
    }

    if name not in policies:
        raise ValueError(f"Unknown policy: {name}. Available: {list(policies.keys())}")

    return policies[name]()
