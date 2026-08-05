"""DeltaCache - Incremental computation-aware KV cache management system."""

from deltacache.api import DeltaCacheManager
from deltacache.core.cache_block import CacheBlock
from deltacache.core.prefix_tree import PrefixTree, PrefixTreeNode, LookupResult
from deltacache.core.memory_pool import MemoryPool
from deltacache.core.memory_monitor import GPUMemoryMonitor, MemoryPressure, MemoryState
from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision, QuantizedKV
from deltacache.core.layer_profiler import LayerAttentionProfiler, LayerProfile, ProfileResult
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator, LayerAllocation, AllocationResult
from deltacache.core.layer_kv_store import LayerKVStore, LayerEntry
from deltacache.metrics import DeltaCacheMetrics
from deltacache.utils.config import DeltaCacheConfig

__version__ = "0.2.0"

__all__ = [
    "DeltaCacheManager",
    "CacheBlock",
    "PrefixTree",
    "PrefixTreeNode",
    "LookupResult",
    "MemoryPool",
    "GPUMemoryMonitor",
    "MemoryPressure",
    "MemoryState",
    "KVQuantizer",
    "QuantPrecision",
    "QuantizedKV",
    "LayerAttentionProfiler",
    "LayerProfile",
    "ProfileResult",
    "LayerBudgetAllocator",
    "LayerAllocation",
    "AllocationResult",
    "LayerKVStore",
    "LayerEntry",
    "DeltaCacheMetrics",
    "DeltaCacheConfig",
]
