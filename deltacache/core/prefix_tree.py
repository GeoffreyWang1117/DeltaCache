"""Prefix tree (trie) index for KV cache management."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator, Callable

import torch
from torch import Tensor

from deltacache.core.cache_block import CacheBlock


@dataclass
class PrefixTreeNode:
    """
    A node in the prefix tree.

    Each node represents a token position and may hold KV cache data
    for that position.
    """
    token: int
    parent: Optional[PrefixTreeNode] = None
    children: Dict[int, PrefixTreeNode] = field(default_factory=dict)
    cache_block: Optional[CacheBlock] = None

    # Metadata for eviction decisions
    ref_count: int = 0
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    depth: int = 0

    def touch(self) -> None:
        """Update access metadata."""
        self.last_access = time.time()
        self.access_count += 1

    def add_ref(self) -> None:
        """Increment reference count."""
        self.ref_count += 1
        if self.cache_block:
            self.cache_block.add_ref()

    def release_ref(self) -> None:
        """Decrement reference count."""
        self.ref_count = max(0, self.ref_count - 1)
        if self.cache_block:
            self.cache_block.release_ref()

    @property
    def is_leaf(self) -> bool:
        """Check if node has no children."""
        return len(self.children) == 0

    @property
    def has_cache(self) -> bool:
        """Check if node has associated cache block."""
        return self.cache_block is not None

    @property
    def subtree_size(self) -> int:
        """Count total nodes in subtree."""
        count = 1
        for child in self.children.values():
            count += child.subtree_size
        return count

    def get_path_to_root(self) -> List[int]:
        """Get token sequence from root to this node."""
        tokens = []
        node = self
        while node.parent is not None:
            tokens.append(node.token)
            node = node.parent
        return list(reversed(tokens))


@dataclass
class LookupResult:
    """Result of a prefix lookup operation."""
    matched_length: int
    matched_node: Optional[PrefixTreeNode]
    kv_cache: Optional[Tuple[Tensor, Tensor]] = None

    @property
    def has_match(self) -> bool:
        """Check if any prefix was matched."""
        return self.matched_length > 0 and self.matched_node is not None


class PrefixTree:
    """
    Prefix tree (trie) for organizing KV cache by token sequences.

    The tree allows O(n) lookup where n is the length of the query prefix,
    independent of the total number of cached sequences.

    Features:
    - Longest prefix matching for cache lookup
    - Efficient insertion and deletion
    - Reference counting for safe cache sharing
    - Subtree traversal for eviction candidates
    """

    def __init__(self) -> None:
        """Initialize empty prefix tree."""
        self._lock = threading.RLock()
        self._root = PrefixTreeNode(token=-1)  # Sentinel root
        self._size = 0
        self._cache_nodes: Dict[int, PrefixTreeNode] = {}  # block_id -> node

    @property
    def size(self) -> int:
        """Number of nodes in tree (excluding root)."""
        return self._size

    @property
    def num_cached(self) -> int:
        """Number of nodes with cache blocks."""
        return len(self._cache_nodes)

    def lookup(self, tokens: List[int]) -> LookupResult:
        """
        Find the longest matching prefix in the tree.

        Args:
            tokens: Token sequence to look up.

        Returns:
            LookupResult with matched length, node, and KV cache if available.
        """
        with self._lock:
            node = self._root
            matched_length = 0
            last_cached_node = None
            last_cached_length = 0

            for i, token in enumerate(tokens):
                if token not in node.children:
                    break

                node = node.children[token]
                node.touch()
                matched_length = i + 1

                if node.has_cache:
                    last_cached_node = node
                    last_cached_length = matched_length

            if last_cached_node is not None:
                kv = last_cached_node.cache_block.get_kv()
                return LookupResult(
                    matched_length=last_cached_length,
                    matched_node=last_cached_node,
                    kv_cache=kv,
                )

            return LookupResult(
                matched_length=matched_length,
                matched_node=node if matched_length > 0 else None,
            )

    def insert(
        self,
        tokens: List[int],
        cache_block: Optional[CacheBlock] = None,
    ) -> PrefixTreeNode:
        """
        Insert a token sequence into the tree.

        Args:
            tokens: Token sequence to insert.
            cache_block: Optional cache block to attach at the end.

        Returns:
            The node at the end of the inserted path.
        """
        with self._lock:
            node = self._root

            for i, token in enumerate(tokens):
                if token not in node.children:
                    new_node = PrefixTreeNode(
                        token=token,
                        parent=node,
                        depth=node.depth + 1,
                    )
                    node.children[token] = new_node
                    self._size += 1

                node = node.children[token]
                node.touch()

            if cache_block is not None:
                node.cache_block = cache_block
                self._cache_nodes[cache_block.block_id] = node

            return node

    def insert_with_kv(
        self,
        tokens: List[int],
        key_cache: Tensor,
        value_cache: Tensor,
        normalized: bool = False,
    ) -> PrefixTreeNode:
        """
        Insert a token sequence with KV cache tensors.

        Args:
            tokens: Token sequence.
            key_cache: Key tensor.
            value_cache: Value tensor.
            normalized: Whether KV are position-normalized.

        Returns:
            The node at the end of the path.
        """
        cache_block = CacheBlock.from_kv(key_cache, value_cache, normalized)
        return self.insert(tokens, cache_block)

    def remove(self, tokens: List[int]) -> bool:
        """
        Remove a token sequence from the tree.

        Only removes leaf nodes; internal nodes with children are kept.

        Args:
            tokens: Token sequence to remove.

        Returns:
            True if sequence was removed.
        """
        with self._lock:
            # Find the node
            node = self._root
            for token in tokens:
                if token not in node.children:
                    return False
                node = node.children[token]

            # Can only remove leaves
            if not node.is_leaf:
                return False

            # Remove cache block if present
            if node.cache_block is not None:
                self._cache_nodes.pop(node.cache_block.block_id, None)
                node.cache_block = None

            # Remove nodes going up until we hit a non-leaf
            while node.parent is not None and node.is_leaf and not node.has_cache:
                parent = node.parent
                del parent.children[node.token]
                self._size -= 1
                node = parent

            return True

    def remove_cache(self, block_id: int) -> bool:
        """
        Remove cache block association from a node.

        Args:
            block_id: ID of cache block to remove.

        Returns:
            True if cache was removed.
        """
        with self._lock:
            if block_id not in self._cache_nodes:
                return False

            node = self._cache_nodes.pop(block_id)
            node.cache_block = None
            return True

    def get_node(self, tokens: List[int]) -> Optional[PrefixTreeNode]:
        """
        Get the node for an exact token sequence.

        Args:
            tokens: Token sequence.

        Returns:
            Node if found, None otherwise.
        """
        with self._lock:
            node = self._root
            for token in tokens:
                if token not in node.children:
                    return None
                node = node.children[token]
            return node

    def get_all_cached_nodes(self) -> List[PrefixTreeNode]:
        """Get all nodes with cache blocks."""
        with self._lock:
            return list(self._cache_nodes.values())

    def iter_nodes(self) -> Iterator[PrefixTreeNode]:
        """Iterate over all nodes in the tree (BFS order)."""
        with self._lock:
            queue = list(self._root.children.values())
            while queue:
                node = queue.pop(0)
                yield node
                queue.extend(node.children.values())

    def iter_leaves(self) -> Iterator[PrefixTreeNode]:
        """Iterate over all leaf nodes."""
        for node in self.iter_nodes():
            if node.is_leaf:
                yield node

    def get_eviction_candidates(
        self,
        scorer: Optional[Callable[[PrefixTreeNode], float]] = None,
        limit: int = 10,
    ) -> List[PrefixTreeNode]:
        """
        Get candidate nodes for eviction, sorted by score.

        Args:
            scorer: Function to compute eviction priority (lower = evict first).
                    If None, uses default based on access recency.
            limit: Maximum number of candidates to return.

        Returns:
            List of nodes sorted by eviction priority.
        """
        if scorer is None:
            scorer = lambda n: n.last_access

        with self._lock:
            candidates = [n for n in self._cache_nodes.values() if n.ref_count == 0]
            candidates.sort(key=scorer)
            return candidates[:limit]

    def find_shared_prefix(self, tokens_list: List[List[int]]) -> Tuple[List[int], List[PrefixTreeNode]]:
        """
        Find the common prefix shared by multiple token sequences.

        Args:
            tokens_list: List of token sequences.

        Returns:
            Tuple of (common prefix tokens, list of end nodes for each sequence).
        """
        if not tokens_list:
            return [], []

        # Find minimum length and common prefix
        min_len = min(len(t) for t in tokens_list)
        common_prefix = []

        for i in range(min_len):
            tokens_at_pos = set(t[i] for t in tokens_list)
            if len(tokens_at_pos) == 1:
                common_prefix.append(tokens_list[0][i])
            else:
                break

        # Get nodes for each sequence
        nodes = []
        for tokens in tokens_list:
            result = self.lookup(tokens)
            nodes.append(result.matched_node)

        return common_prefix, nodes

    def clear(self) -> None:
        """Clear all nodes from the tree."""
        with self._lock:
            self._root = PrefixTreeNode(token=-1)
            self._size = 0
            self._cache_nodes.clear()

    def __len__(self) -> int:
        return self._size

    def __contains__(self, tokens: List[int]) -> bool:
        return self.get_node(tokens) is not None
