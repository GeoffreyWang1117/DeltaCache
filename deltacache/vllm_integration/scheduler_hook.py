"""vLLM scheduler hook for DeltaCache prefix-aware scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from deltacache.api import DeltaCacheManager
from deltacache.core.prefix_tree import PrefixTree

if TYPE_CHECKING:
    try:
        from vllm.sequence import SequenceGroup
    except ImportError:
        SequenceGroup = None


@dataclass
class SchedulingHint:
    """Hint for scheduler about a sequence's cache status."""
    seq_id: int
    prefix_length: int
    has_cache: bool
    shared_prefix_group: Optional[int] = None  # Group ID for shared prefix


class SchedulerHook:
    """
    Hook into vLLM scheduler for prefix-aware scheduling.

    This hook provides information about prefix sharing to the scheduler,
    allowing it to batch sequences with shared prefixes together for
    better GPU utilization.

    Features:
    - Identify sequences sharing common prefixes
    - Prioritize sequences with cached prefixes
    - Group sequences for batch processing

    Usage:
        ```python
        from vllm import LLM
        from deltacache.vllm_integration import SchedulerHook

        # Create hook with shared cache manager
        hook = SchedulerHook(cache_manager)

        # Integrate with vLLM scheduler
        # (requires custom scheduler implementation)
        ```
    """

    def __init__(
        self,
        cache_manager: Optional[DeltaCacheManager] = None,
        prefix_tree: Optional[PrefixTree] = None,
    ) -> None:
        """
        Initialize scheduler hook.

        Args:
            cache_manager: DeltaCacheManager instance.
            prefix_tree: Or just the prefix tree for lookups.
        """
        if cache_manager:
            self.prefix_tree = cache_manager.prefix_tree
            self.cache_manager = cache_manager
        elif prefix_tree:
            self.prefix_tree = prefix_tree
            self.cache_manager = None
        else:
            self.prefix_tree = PrefixTree()
            self.cache_manager = None

        # Track sequence groups by prefix
        self._prefix_groups: Dict[Tuple[int, ...], List[int]] = {}
        self._seq_prefix_map: Dict[int, Tuple[int, ...]] = {}

    def on_new_sequence(
        self,
        seq_id: int,
        token_ids: List[int],
    ) -> SchedulingHint:
        """
        Called when a new sequence arrives.

        Args:
            seq_id: Sequence identifier.
            token_ids: Token IDs for the prompt.

        Returns:
            SchedulingHint with cache information.
        """
        # Look up existing cache
        result = self.prefix_tree.lookup(token_ids)

        # Find matching prefix group
        prefix_key = tuple(token_ids[:result.matched_length]) if result.matched_length > 0 else ()

        # Register sequence in prefix group
        if prefix_key not in self._prefix_groups:
            self._prefix_groups[prefix_key] = []
        self._prefix_groups[prefix_key].append(seq_id)
        self._seq_prefix_map[seq_id] = prefix_key

        # Compute group ID (hash of prefix)
        group_id = hash(prefix_key) if prefix_key else None

        return SchedulingHint(
            seq_id=seq_id,
            prefix_length=result.matched_length,
            has_cache=result.has_match and result.kv_cache is not None,
            shared_prefix_group=group_id,
        )

    def on_sequence_finished(self, seq_id: int) -> None:
        """
        Called when a sequence completes.

        Args:
            seq_id: Completed sequence ID.
        """
        if seq_id in self._seq_prefix_map:
            prefix_key = self._seq_prefix_map.pop(seq_id)
            if prefix_key in self._prefix_groups:
                self._prefix_groups[prefix_key] = [
                    s for s in self._prefix_groups[prefix_key] if s != seq_id
                ]
                if not self._prefix_groups[prefix_key]:
                    del self._prefix_groups[prefix_key]

    def get_shared_prefix_groups(self) -> Dict[int, List[int]]:
        """
        Get current prefix sharing groups.

        Returns:
            Dict mapping group ID to list of sequence IDs.
        """
        result = {}
        for prefix_key, seq_ids in self._prefix_groups.items():
            if len(seq_ids) > 1:  # Only groups with sharing
                group_id = hash(prefix_key)
                result[group_id] = seq_ids.copy()
        return result

    def suggest_batch_order(
        self,
        sequence_ids: List[int],
    ) -> List[int]:
        """
        Suggest optimal batch ordering for sequences.

        Orders sequences to maximize prefix sharing within batch.

        Args:
            sequence_ids: Sequences to order.

        Returns:
            Reordered sequence IDs.
        """
        # Group by prefix
        groups: Dict[Tuple[int, ...], List[int]] = {}
        ungrouped = []

        for seq_id in sequence_ids:
            if seq_id in self._seq_prefix_map:
                prefix = self._seq_prefix_map[seq_id]
                if prefix:
                    if prefix not in groups:
                        groups[prefix] = []
                    groups[prefix].append(seq_id)
                else:
                    ungrouped.append(seq_id)
            else:
                ungrouped.append(seq_id)

        # Order: larger groups first (more sharing benefit)
        sorted_groups = sorted(groups.values(), key=len, reverse=True)

        result = []
        for group in sorted_groups:
            result.extend(group)
        result.extend(ungrouped)

        return result

    def get_priority_boost(self, seq_id: int) -> float:
        """
        Get scheduling priority boost for a sequence.

        Sequences with cached prefixes get higher priority.

        Args:
            seq_id: Sequence ID.

        Returns:
            Priority boost factor (1.0 = no boost).
        """
        if seq_id not in self._seq_prefix_map:
            return 1.0

        prefix = self._seq_prefix_map[seq_id]
        if not prefix:
            return 1.0

        # Longer cached prefix = higher priority
        prefix_length = len(prefix)

        # Check if prefix is actually cached
        result = self.prefix_tree.lookup(list(prefix))
        if result.has_match and result.kv_cache is not None:
            # Boost based on prefix length (more computation saved)
            return 1.0 + (prefix_length / 100.0)

        return 1.0

    def find_prefetch_candidates(
        self,
        pending_sequences: List[Tuple[int, List[int]]],
        gpu_memory_available: int,
    ) -> List[int]:
        """
        Find sequences whose cache should be prefetched to GPU.

        Args:
            pending_sequences: List of (seq_id, token_ids) tuples.
            gpu_memory_available: Available GPU memory in bytes.

        Returns:
            List of sequence IDs to prefetch.
        """
        candidates = []

        for seq_id, token_ids in pending_sequences:
            result = self.prefix_tree.lookup(token_ids)
            if result.matched_node and result.matched_node.cache_block:
                block = result.matched_node.cache_block
                if not block.is_on_gpu:
                    if block.memory_size <= gpu_memory_available:
                        candidates.append(seq_id)
                        gpu_memory_available -= block.memory_size

        return candidates


class PrefixAwareScheduler:
    """
    Wrapper to make vLLM scheduler prefix-aware.

    This is a higher-level integration that wraps vLLM's scheduler
    to incorporate prefix sharing information.
    """

    def __init__(
        self,
        hook: SchedulerHook,
        base_scheduler: Optional[object] = None,
    ) -> None:
        """
        Initialize prefix-aware scheduler.

        Args:
            hook: Scheduler hook for prefix information.
            base_scheduler: Underlying vLLM scheduler (optional).
        """
        self.hook = hook
        self.base_scheduler = base_scheduler

    def add_sequence(self, seq_id: int, token_ids: List[int]) -> SchedulingHint:
        """Add a new sequence for scheduling."""
        return self.hook.on_new_sequence(seq_id, token_ids)

    def remove_sequence(self, seq_id: int) -> None:
        """Remove a completed sequence."""
        self.hook.on_sequence_finished(seq_id)

    def select_batch(
        self,
        ready_sequences: List[int],
        max_batch_size: int,
    ) -> List[int]:
        """
        Select sequences for the next batch.

        Prioritizes sequences that share prefixes.

        Args:
            ready_sequences: Sequences ready for processing.
            max_batch_size: Maximum batch size.

        Returns:
            Selected sequence IDs.
        """
        # Get optimal ordering
        ordered = self.hook.suggest_batch_order(ready_sequences)

        # Apply priority boosts
        with_priority = [
            (seq_id, self.hook.get_priority_boost(seq_id))
            for seq_id in ordered
        ]

        # Sort by priority (higher first)
        with_priority.sort(key=lambda x: x[1], reverse=True)

        # Select top sequences
        selected = [seq_id for seq_id, _ in with_priority[:max_batch_size]]

        # Re-order selected for prefix sharing
        return self.hook.suggest_batch_order(selected)
