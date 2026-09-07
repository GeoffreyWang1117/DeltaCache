"""DeltaCache - Incremental computation-aware KV cache management system."""

from deltacache.api import DeltaCacheManager
from deltacache.core.cache_block import CacheBlock
from deltacache.core.kv_quantizer import KVQuantizer, QuantizedKV, QuantPrecision
from deltacache.core.layer_budget_allocator import (
    AllocationResult,
    LayerAllocation,
    LayerBudgetAllocator,
)
from deltacache.core.layer_kv_store import LayerEntry, LayerKVStore
from deltacache.core.layer_profiler import LayerAttentionProfiler, LayerProfile, ProfileResult
from deltacache.core.memory_monitor import GPUMemoryMonitor, MemoryPressure, MemoryState
from deltacache.core.memory_pool import MemoryPool
from deltacache.core.prefix_tree import LookupResult, PrefixTree, PrefixTreeNode
from deltacache.metrics import DeltaCacheMetrics
from deltacache.utils.config import DeltaCacheConfig

__version__ = "0.2.0"

__all__ = [
    "AllocationResult",
    "CacheBlock",
    "DeltaCacheConfig",
    "DeltaCacheManager",
    "DeltaCacheMetrics",
    "GPUMemoryMonitor",
    "KVQuantizer",
    "LayerAllocation",
    "LayerAttentionProfiler",
    "LayerBudgetAllocator",
    "LayerEntry",
    "LayerKVStore",
    "LayerProfile",
    "LookupResult",
    "MemoryPool",
    "MemoryPressure",
    "MemoryState",
    "PrefixTree",
    "PrefixTreeNode",
    "ProfileResult",
    "QuantPrecision",
    "QuantizedKV",
]
