"""Core data structures for DeltaCache."""

from deltacache.core.cache_block import CacheBlock
from deltacache.core.prefix_tree import PrefixTree, PrefixTreeNode, LookupResult
from deltacache.core.memory_pool import MemoryPool
from deltacache.core.hierarchical_memory import (
    HierarchicalMemoryManager,
    HierarchicalMemoryStats,
)

__all__ = [
    "CacheBlock",
    "PrefixTree",
    "PrefixTreeNode",
    "LookupResult",
    "MemoryPool",
    "HierarchicalMemoryManager",
    "HierarchicalMemoryStats",
]
