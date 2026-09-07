"""Cache eviction policies for DeltaCache."""

from deltacache.eviction.advanced_policies import (
    AttentionAwareEvictionPolicy,
    AttentionMetadata,
    HierarchicalEvictionPolicy,
    LayerAwareCachingPolicy,
    create_advanced_eviction_policy,
)
from deltacache.eviction.policy import (
    AdaptiveEvictionPolicy,
    CompositeEvictionPolicy,
    EvictionAction,
    EvictionCandidate,
    EvictionPolicy,
    EvictionResult,
    LFUEvictionPolicy,
    LRUEvictionPolicy,
    TieredEvictionPolicy,
    create_eviction_policy,
)

__all__ = [
    "AdaptiveEvictionPolicy",
    # Advanced policies
    "AttentionAwareEvictionPolicy",
    "AttentionMetadata",
    "CompositeEvictionPolicy",
    "EvictionAction",
    "EvictionCandidate",
    # Base policies
    "EvictionPolicy",
    "EvictionResult",
    "HierarchicalEvictionPolicy",
    "LFUEvictionPolicy",
    "LRUEvictionPolicy",
    "LayerAwareCachingPolicy",
    "TieredEvictionPolicy",
    "create_advanced_eviction_policy",
    "create_eviction_policy",
]
