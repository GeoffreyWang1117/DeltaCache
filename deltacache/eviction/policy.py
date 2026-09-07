"""Cache eviction policies for DeltaCache."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from deltacache.core.memory_pool import MemoryPool
from deltacache.core.prefix_tree import PrefixTree, PrefixTreeNode


class EvictionAction(Enum):
    """Type of eviction action."""

    OFFLOAD_TO_CPU = "offload"  # Move from GPU to CPU
    DELETE = "delete"  # Remove completely


@dataclass
class EvictionCandidate:
    """A candidate for eviction with computed priority."""

    node: PrefixTreeNode
    score: float
    action: EvictionAction = EvictionAction.OFFLOAD_TO_CPU
    memory_freed: int = 0

    def __lt__(self, other: EvictionCandidate) -> bool:
        """Lower score = higher eviction priority."""
        return self.score < other.score


@dataclass
class EvictionResult:
    """Result of an eviction operation."""

    num_offloaded: int = 0
    num_deleted: int = 0
    memory_freed_gpu: int = 0
    memory_freed_cpu: int = 0


class EvictionPolicy(ABC):
    """Abstract base class for eviction policies."""

    @abstractmethod
    def select_victims(
        self,
        prefix_tree: PrefixTree,
        memory_pool: MemoryPool,
        required_memory: int,
    ) -> List[EvictionCandidate]:
        """
        Select nodes to evict.

        Args:
            prefix_tree: Prefix tree with cached nodes.
            memory_pool: Memory pool with usage stats.
            required_memory: Amount of memory to free.

        Returns:
            List of eviction candidates.
        """
        pass

    def evict(
        self,
        prefix_tree: PrefixTree,
        memory_pool: MemoryPool,
        required_memory: int,
    ) -> EvictionResult:
        """
        Execute eviction to free required memory.

        Args:
            prefix_tree: Prefix tree.
            memory_pool: Memory pool.
            required_memory: Memory to free.

        Returns:
            EvictionResult with statistics.
        """
        candidates = self.select_victims(prefix_tree, memory_pool, required_memory)
        result = EvictionResult()

        freed = 0
        for candidate in candidates:
            if freed >= required_memory:
                break

            if candidate.node.cache_block is None:
                continue

            block = candidate.node.cache_block
            block_id = block.block_id
            memory_size = block.memory_size

            if candidate.action == EvictionAction.OFFLOAD_TO_CPU:
                if memory_pool.move_to_cpu(block_id):
                    result.num_offloaded += 1
                    result.memory_freed_gpu += memory_size
                    freed += memory_size
            else:  # DELETE
                if block.is_on_gpu:
                    result.memory_freed_gpu += memory_size
                else:
                    result.memory_freed_cpu += memory_size

                memory_pool.free(block_id)
                prefix_tree.remove_cache(block_id)
                result.num_deleted += 1
                freed += memory_size

        return result


class LRUEvictionPolicy(EvictionPolicy):
    """
    Least Recently Used eviction policy.

    Simple policy that evicts nodes with oldest access times first.
    """

    def select_victims(
        self,
        prefix_tree: PrefixTree,
        memory_pool: MemoryPool,
        required_memory: int,
    ) -> List[EvictionCandidate]:
        candidates = []

        for node in prefix_tree.get_all_cached_nodes():
            if node.ref_count > 0:
                continue  # Don't evict referenced nodes

            if node.cache_block is None:
                continue

            candidate = EvictionCandidate(
                node=node,
                score=node.last_access,  # Lower = older = evict first
                action=EvictionAction.OFFLOAD_TO_CPU
                if node.cache_block.is_on_gpu
                else EvictionAction.DELETE,
                memory_freed=node.cache_block.memory_size,
            )
            candidates.append(candidate)

        candidates.sort()
        return candidates


class LFUEvictionPolicy(EvictionPolicy):
    """
    Least Frequently Used eviction policy.

    Evicts nodes with lowest access counts first.
    """

    def select_victims(
        self,
        prefix_tree: PrefixTree,
        memory_pool: MemoryPool,
        required_memory: int,
    ) -> List[EvictionCandidate]:
        candidates = []

        for node in prefix_tree.get_all_cached_nodes():
            if node.ref_count > 0:
                continue

            if node.cache_block is None:
                continue

            candidate = EvictionCandidate(
                node=node,
                score=node.access_count,  # Lower = less used = evict first
                action=EvictionAction.OFFLOAD_TO_CPU
                if node.cache_block.is_on_gpu
                else EvictionAction.DELETE,
                memory_freed=node.cache_block.memory_size,
            )
            candidates.append(candidate)

        candidates.sort()
        return candidates


class CompositeEvictionPolicy(EvictionPolicy):
    """
    Composite eviction policy combining multiple factors.

    Considers:
    - Access frequency (higher = more valuable)
    - Subtree size (larger = more children depend on it)
    - Recomputation cost (deeper = more expensive to recompute)
    - Recency (more recent = more valuable)

    Score formula:
    score = (access_count * subtree_size * depth) / time_since_access

    Higher score = more valuable = evict last
    Lower score = less valuable = evict first
    """

    def __init__(
        self,
        frequency_weight: float = 1.0,
        subtree_weight: float = 0.5,
        depth_weight: float = 0.3,
        recency_weight: float = 1.0,
        prefer_offload: bool = True,
    ) -> None:
        """
        Initialize composite policy.

        Args:
            frequency_weight: Weight for access frequency.
            subtree_weight: Weight for subtree size.
            depth_weight: Weight for node depth (recomputation cost).
            recency_weight: Weight for access recency.
            prefer_offload: Prefer offloading to CPU over deletion.
        """
        self.frequency_weight = frequency_weight
        self.subtree_weight = subtree_weight
        self.depth_weight = depth_weight
        self.recency_weight = recency_weight
        self.prefer_offload = prefer_offload

    def _compute_score(self, node: PrefixTreeNode) -> float:
        """Compute eviction score for a node."""
        now = time.time()
        time_since_access = max(1.0, now - node.last_access)

        # Higher frequency = more valuable
        frequency_score = (node.access_count + 1) * self.frequency_weight

        # Larger subtree = more dependents
        subtree_score = node.subtree_size * self.subtree_weight

        # Deeper node = more expensive to recompute
        depth_score = (node.depth + 1) * self.depth_weight

        # More recent = more valuable
        recency_score = self.recency_weight / time_since_access

        # Combined score: higher = more valuable
        total_score = frequency_score * subtree_score * depth_score * recency_score

        return total_score

    def select_victims(
        self,
        prefix_tree: PrefixTree,
        memory_pool: MemoryPool,
        required_memory: int,
    ) -> List[EvictionCandidate]:
        candidates = []

        for node in prefix_tree.get_all_cached_nodes():
            if node.ref_count > 0:
                continue

            if node.cache_block is None:
                continue

            score = self._compute_score(node)

            # Determine action
            if self.prefer_offload and node.cache_block.is_on_gpu:
                action = EvictionAction.OFFLOAD_TO_CPU
            else:
                action = EvictionAction.DELETE

            candidate = EvictionCandidate(
                node=node,
                score=score,
                action=action,
                memory_freed=node.cache_block.memory_size,
            )
            candidates.append(candidate)

        candidates.sort()
        return candidates


class TieredEvictionPolicy(EvictionPolicy):
    """
    Tiered eviction policy: GPU -> CPU -> Delete.

    First offloads GPU blocks to CPU, then deletes CPU blocks.
    Uses a composite score for ordering within each tier.
    """

    def __init__(self, base_policy: Optional[EvictionPolicy] = None) -> None:
        """
        Initialize tiered policy.

        Args:
            base_policy: Policy for scoring within tiers.
        """
        self.base_policy = base_policy or CompositeEvictionPolicy()

    def select_victims(
        self,
        prefix_tree: PrefixTree,
        memory_pool: MemoryPool,
        required_memory: int,
    ) -> List[EvictionCandidate]:
        # Get all candidates from base policy
        all_candidates = self.base_policy.select_victims(
            prefix_tree,
            memory_pool,
            required_memory * 2,  # Get more candidates
        )

        # Separate into GPU and CPU blocks
        gpu_candidates = []
        cpu_candidates = []

        for c in all_candidates:
            if c.node.cache_block and c.node.cache_block.is_on_gpu:
                c.action = EvictionAction.OFFLOAD_TO_CPU
                gpu_candidates.append(c)
            else:
                c.action = EvictionAction.DELETE
                cpu_candidates.append(c)

        # GPU offloads first, then CPU deletes
        return gpu_candidates + cpu_candidates


class AdaptiveEvictionPolicy(EvictionPolicy):
    """
    Adaptive eviction using online learning.

    Tracks which eviction decisions led to cache misses and adjusts
    scoring weights accordingly.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        history_size: int = 1000,
    ) -> None:
        """
        Initialize adaptive policy.

        Args:
            learning_rate: Rate for weight updates.
            history_size: Number of evictions to track.
        """
        self.learning_rate = learning_rate
        self.history_size = history_size

        # Learnable weights
        self.weights = {
            "frequency": 1.0,
            "subtree": 0.5,
            "depth": 0.3,
            "recency": 1.0,
        }

        # Eviction history: (node_tokens, was_reaccessed)
        self._history: List[tuple] = []
        self._evicted_tokens: set = set()

    def record_access(self, tokens: tuple) -> None:
        """Record an access to update learning."""
        if tokens in self._evicted_tokens:
            # We evicted something that was reaccessed - bad decision
            self._update_weights(was_miss=True)
            self._evicted_tokens.discard(tokens)

    def _update_weights(self, was_miss: bool) -> None:
        """Update weights based on eviction outcome."""
        # Simple update: increase weights when we had misses
        if was_miss:
            # Make eviction more conservative
            for key in self.weights:
                self.weights[key] *= 1 + self.learning_rate
        else:
            # Make eviction more aggressive
            for key in self.weights:
                self.weights[key] *= 1 - self.learning_rate * 0.1

    def select_victims(
        self,
        prefix_tree: PrefixTree,
        memory_pool: MemoryPool,
        required_memory: int,
    ) -> List[EvictionCandidate]:
        candidates = []

        for node in prefix_tree.get_all_cached_nodes():
            if node.ref_count > 0:
                continue

            if node.cache_block is None:
                continue

            score = self._compute_adaptive_score(node)

            candidate = EvictionCandidate(
                node=node,
                score=score,
                action=EvictionAction.OFFLOAD_TO_CPU
                if node.cache_block.is_on_gpu
                else EvictionAction.DELETE,
                memory_freed=node.cache_block.memory_size,
            )
            candidates.append(candidate)

        candidates.sort()

        # Track evictions
        for c in candidates[:10]:  # Track top candidates
            tokens = tuple(c.node.get_path_to_root())
            self._evicted_tokens.add(tokens)

        return candidates

    def _compute_adaptive_score(self, node: PrefixTreeNode) -> float:
        """Compute score using learned weights."""
        now = time.time()
        time_since_access = max(1.0, now - node.last_access)

        score = (
            (node.access_count + 1)
            * self.weights["frequency"]
            * node.subtree_size
            * self.weights["subtree"]
            * (node.depth + 1)
            * self.weights["depth"]
            * self.weights["recency"]
            / time_since_access
        )

        return score


def create_eviction_policy(name: str = "composite", **kwargs) -> EvictionPolicy:
    """
    Factory function to create eviction policies.

    Args:
        name: Policy name ("lru", "lfu", "composite", "tiered", "adaptive").
        **kwargs: Additional arguments for policy constructor.

    Returns:
        Eviction policy instance.
    """
    policies = {
        "lru": LRUEvictionPolicy,
        "lfu": LFUEvictionPolicy,
        "composite": CompositeEvictionPolicy,
        "tiered": TieredEvictionPolicy,
        "adaptive": AdaptiveEvictionPolicy,
    }

    if name not in policies:
        raise ValueError(f"Unknown policy: {name}. Available: {list(policies.keys())}")

    return policies[name](**kwargs)
