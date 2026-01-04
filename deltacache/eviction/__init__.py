"""Cache eviction policies for DeltaCache."""

from deltacache.eviction.policy import (
    EvictionPolicy,
    EvictionCandidate,
    EvictionResult,
    EvictionAction,
    LRUEvictionPolicy,
    LFUEvictionPolicy,
    CompositeEvictionPolicy,
    TieredEvictionPolicy,
    AdaptiveEvictionPolicy,
    create_eviction_policy,
)

from deltacache.eviction.advanced_policies import (
    AttentionAwareEvictionPolicy,
    LayerAwareCachingPolicy,
    HierarchicalEvictionPolicy,
    AttentionMetadata,
    create_advanced_eviction_policy,
)

__all__ = [
    # Base policies
    "EvictionPolicy",
    "EvictionCandidate",
    "EvictionResult",
    "EvictionAction",
    "LRUEvictionPolicy",
    "LFUEvictionPolicy",
    "CompositeEvictionPolicy",
    "TieredEvictionPolicy",
    "AdaptiveEvictionPolicy",
    "create_eviction_policy",
    # Advanced policies
    "AttentionAwareEvictionPolicy",
    "LayerAwareCachingPolicy",
    "HierarchicalEvictionPolicy",
    "AttentionMetadata",
    "create_advanced_eviction_policy",
]
