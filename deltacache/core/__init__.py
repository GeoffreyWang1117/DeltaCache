"""Core data structures for DeltaCache."""

from deltacache.core.cache_block import CacheBlock
from deltacache.core.hierarchical_memory import (
    HierarchicalMemoryManager,
    HierarchicalMemoryStats,
)
from deltacache.core.memory_pool import MemoryPool
from deltacache.core.prefix_tree import LookupResult, PrefixTree, PrefixTreeNode

__all__ = [
    "CacheBlock",
    "HierarchicalMemoryManager",
    "HierarchicalMemoryStats",
    "LookupResult",
    "MemoryPool",
    "PrefixTree",
    "PrefixTreeNode",
]
