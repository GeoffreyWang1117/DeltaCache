"""Incremental computation engine for DeltaCache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Protocol

import torch
from torch import Tensor

from deltacache.core.prefix_tree import PrefixTree, LookupResult
from deltacache.core.cache_block import CacheBlock
from deltacache.engine.rope_handler import RoPEHandler, create_position_ids


class KVComputeFunc(Protocol):
    """Protocol for KV computation function."""

    def __call__(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        past_key_values: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute KV cache for given input.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            position_ids: Position indices [batch_size, seq_len].
            past_key_values: Optional existing KV cache.

        Returns:
            Tuple of (key_cache, value_cache).
        """
        ...


@dataclass
class IncrementalResult:
    """Result of incremental computation."""
    key_cache: Tensor
    value_cache: Tensor
    matched_length: int
    computed_length: int
    cache_hit: bool

    @property
    def total_length(self) -> int:
        """Total sequence length."""
        return self.matched_length + self.computed_length


class IncrementalEngine:
    """
    Engine for incremental KV cache computation.

    Given a token sequence, the engine:
    1. Queries the prefix tree for the longest matching prefix
    2. Retrieves cached KV for the matched prefix
    3. Computes KV only for the unmatched suffix
    4. Merges cached and computed KV

    This reduces redundant computation when requests share common prefixes.
    """

    def __init__(
        self,
        prefix_tree: PrefixTree,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        rope_handler: Optional[RoPEHandler] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Initialize incremental engine.

        Args:
            prefix_tree: Prefix tree for cache lookup.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Dimension per attention head.
            rope_handler: Optional RoPE handler for position adjustments.
            device: Computation device.
            dtype: Data type for tensors.
        """
        self.prefix_tree = prefix_tree
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_handler = rope_handler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Statistics
        self._total_tokens = 0
        self._cached_tokens = 0
        self._computed_tokens = 0

    @property
    def cache_hit_rate(self) -> float:
        """Ratio of tokens served from cache."""
        if self._total_tokens == 0:
            return 0.0
        return self._cached_tokens / self._total_tokens

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_tokens = 0
        self._cached_tokens = 0
        self._computed_tokens = 0

    def compute(
        self,
        tokens: List[int],
        compute_fn: KVComputeFunc,
        store_result: bool = True,
    ) -> IncrementalResult:
        """
        Compute KV cache incrementally.

        Args:
            tokens: Token sequence.
            compute_fn: Function to compute KV for new tokens.
            store_result: Whether to store result in prefix tree.

        Returns:
            IncrementalResult with full KV cache.
        """
        # Look up existing cache
        lookup_result = self.prefix_tree.lookup(tokens)

        # Only count as a match if we have actual cached KV values
        # (intermediate tree nodes may match but have no cached KV)
        has_usable_cache = lookup_result.has_match and lookup_result.kv_cache is not None
        matched_length = lookup_result.matched_length if has_usable_cache else 0
        suffix_tokens = tokens[matched_length:]

        self._total_tokens += len(tokens)

        # Full cache hit
        if matched_length == len(tokens) and lookup_result.kv_cache is not None:
            self._cached_tokens += len(tokens)
            key_cache, value_cache = lookup_result.kv_cache
            return IncrementalResult(
                key_cache=key_cache,
                value_cache=value_cache,
                matched_length=matched_length,
                computed_length=0,
                cache_hit=True,
            )

        # Compute KV for suffix
        if suffix_tokens:
            suffix_result = self._compute_suffix(
                suffix_tokens,
                matched_length,
                compute_fn,
                lookup_result.kv_cache,
            )
        else:
            suffix_result = None

        # Merge cached and computed KV
        if lookup_result.kv_cache is not None and suffix_result is not None:
            # Merge prefix cache with suffix computation
            cached_key, cached_value = lookup_result.kv_cache
            key_cache = torch.cat([cached_key, suffix_result[0]], dim=1)
            value_cache = torch.cat([cached_value, suffix_result[1]], dim=1)
            self._cached_tokens += matched_length
            self._computed_tokens += len(suffix_tokens)
        elif lookup_result.kv_cache is not None:
            # Only cache, no suffix
            key_cache, value_cache = lookup_result.kv_cache
            self._cached_tokens += matched_length
        elif suffix_result is not None:
            # Only suffix, no cache
            key_cache, value_cache = suffix_result
            self._computed_tokens += len(suffix_tokens)
        else:
            raise RuntimeError("No cache and no computation result")

        # Store in prefix tree
        if store_result:
            cache_block = CacheBlock.from_kv(key_cache, value_cache)
            self.prefix_tree.insert(tokens, cache_block)

        return IncrementalResult(
            key_cache=key_cache,
            value_cache=value_cache,
            matched_length=matched_length,
            computed_length=len(suffix_tokens),
            cache_hit=matched_length > 0,
        )

    def _compute_suffix(
        self,
        suffix_tokens: List[int],
        start_position: int,
        compute_fn: KVComputeFunc,
        past_kv: Optional[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute KV for suffix tokens.

        Args:
            suffix_tokens: Tokens to compute.
            start_position: Starting position in full sequence.
            compute_fn: KV computation function.
            past_kv: Optional past KV cache to attend to.

        Returns:
            Tuple of (key_cache, value_cache) for suffix.
        """
        # Prepare input
        input_ids = torch.tensor(
            [suffix_tokens],
            dtype=torch.long,
            device=self.device,
        )

        position_ids = create_position_ids(
            len(suffix_tokens),
            start_pos=start_position,
            device=self.device,
        ).unsqueeze(0)

        # Compute KV
        key_cache, value_cache = compute_fn(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_kv,
        )

        return key_cache, value_cache

    def compute_batch(
        self,
        batch_tokens: List[List[int]],
        compute_fn: KVComputeFunc,
        store_results: bool = True,
    ) -> List[IncrementalResult]:
        """
        Compute KV cache for a batch of sequences.

        Sequences sharing common prefixes are grouped for efficient computation.

        Args:
            batch_tokens: List of token sequences.
            compute_fn: KV computation function.
            store_results: Whether to store results.

        Returns:
            List of IncrementalResults.
        """
        results = []

        # Group by shared prefix
        groups = self._group_by_prefix(batch_tokens)

        for prefix_tokens, suffix_list, indices in groups:
            group_results = self._compute_group(
                prefix_tokens,
                suffix_list,
                compute_fn,
                store_results,
            )

            # Map results back to original order
            for idx, result in zip(indices, group_results):
                results.append((idx, result))

        # Sort by original order
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]

    def _group_by_prefix(
        self,
        batch_tokens: List[List[int]],
    ) -> List[Tuple[List[int], List[List[int]], List[int]]]:
        """
        Group sequences by their longest common cached prefix.

        Returns:
            List of (prefix, suffixes, original_indices) tuples.
        """
        # For simplicity, compute individually but track prefixes
        groups = {}

        for i, tokens in enumerate(batch_tokens):
            lookup = self.prefix_tree.lookup(tokens)
            matched = lookup.matched_length if lookup.has_match else 0
            prefix = tuple(tokens[:matched])
            suffix = tokens[matched:]

            if prefix not in groups:
                groups[prefix] = (list(prefix), [], [])

            groups[prefix][1].append(suffix)
            groups[prefix][2].append(i)

        return list(groups.values())

    def _compute_group(
        self,
        prefix_tokens: List[int],
        suffix_list: List[List[int]],
        compute_fn: KVComputeFunc,
        store_results: bool,
    ) -> List[IncrementalResult]:
        """Compute KV for a group sharing the same prefix."""
        results = []

        # Get prefix cache
        prefix_kv = None
        if prefix_tokens:
            lookup = self.prefix_tree.lookup(prefix_tokens)
            if lookup.kv_cache is not None:
                prefix_kv = lookup.kv_cache

        # Compute each suffix
        for suffix in suffix_list:
            full_tokens = prefix_tokens + suffix

            if not suffix and prefix_kv is not None:
                # Full prefix match
                key_cache, value_cache = prefix_kv
                result = IncrementalResult(
                    key_cache=key_cache,
                    value_cache=value_cache,
                    matched_length=len(prefix_tokens),
                    computed_length=0,
                    cache_hit=True,
                )
            else:
                # Need to compute suffix
                result = self.compute(full_tokens, compute_fn, store_results)

            results.append(result)

        return results

    def prefetch(self, tokens_list: List[List[int]]) -> List[bool]:
        """
        Ensure cache blocks are on GPU for upcoming requests.

        Args:
            tokens_list: List of token sequences to prefetch.

        Returns:
            List of booleans indicating if each was found in cache.
        """
        results = []

        for tokens in tokens_list:
            lookup = self.prefix_tree.lookup(tokens)
            if lookup.matched_node and lookup.matched_node.cache_block:
                block = lookup.matched_node.cache_block
                if not block.is_on_gpu:
                    block.to_gpu(self.device)
                results.append(True)
            else:
                results.append(False)

        return results
