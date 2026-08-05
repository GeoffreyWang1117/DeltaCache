"""Unified API for DeltaCache system."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from torch import Tensor

from deltacache.core.cache_block import CacheBlock
from deltacache.core.memory_monitor import GPUMemoryMonitor, MemoryPressure
from deltacache.core.memory_pool import MemoryPool, MemoryStats
from deltacache.core.prefix_tree import LookupResult, PrefixTree
from deltacache.engine.incremental import IncrementalEngine, IncrementalResult, KVComputeFunc
from deltacache.engine.rope_handler import RoPEHandler
from deltacache.eviction.policy import EvictionPolicy, EvictionResult, create_eviction_policy
from deltacache.metrics import DeltaCacheMetrics
from deltacache.utils.config import CacheStats, DeltaCacheConfig

logger = logging.getLogger(__name__)


class DeltaCacheManager:
    """
    Main interface for the DeltaCache system.

    Provides a unified API for:
    - Cache lookup and insertion
    - Incremental KV computation
    - Memory management and eviction
    - Statistics tracking

    Example:
        ```python
        config = DeltaCacheConfig.for_model("llama-7b")
        manager = DeltaCacheManager(config)

        # Lookup existing cache
        result = manager.lookup(tokens)
        if result.has_match:
            # Use cached KV
            key, value = result.kv_cache

        # Or compute incrementally
        result = manager.compute_incremental(tokens, model.compute_kv)
        ```
    """

    def __init__(
        self,
        config: Optional[DeltaCacheConfig] = None,
        prefix_tree: Optional[PrefixTree] = None,
        memory_pool: Optional[MemoryPool] = None,
        eviction_policy: Optional[EvictionPolicy] = None,
    ) -> None:
        """
        Initialize DeltaCacheManager.

        Args:
            config: Configuration. If None, uses defaults.
            prefix_tree: Custom prefix tree. If None, creates new.
            memory_pool: Custom memory pool. If None, creates new.
            eviction_policy: Custom eviction policy. If None, uses config.
        """
        self.config = config or DeltaCacheConfig()

        # Core components
        self.prefix_tree = prefix_tree or PrefixTree()
        self.memory_pool = memory_pool or MemoryPool(
            gpu_limit=self.config.gpu_memory_limit,
            cpu_limit=self.config.cpu_memory_limit,
            device=self.config.torch_device,
        )

        # Eviction
        self.eviction_policy = eviction_policy or create_eviction_policy(
            self.config.eviction_policy
        )

        # RoPE handler
        self.rope_handler = RoPEHandler(
            head_dim=self.config.head_dim,
            max_position=self.config.max_position,
            base=self.config.rope_base,
            device=self.config.torch_device,
        )

        # Incremental engine
        self.engine = IncrementalEngine(
            prefix_tree=self.prefix_tree,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            rope_handler=self.rope_handler,
            device=self.config.torch_device,
            dtype=self.config.torch_dtype,
        )

        # Statistics
        self.stats = CacheStats()

        # Prometheus metrics (no-op if prometheus_client not installed)
        self.metrics = DeltaCacheMetrics(enabled=self.config.enable_metrics)

        # GPU memory monitor
        self.memory_monitor: Optional[GPUMemoryMonitor] = None
        if self.config.enable_memory_monitor and self.config.device.startswith("cuda"):
            self.memory_monitor = GPUMemoryMonitor(
                device=self.config.torch_device,
                high_watermark=self.config.high_watermark,
                low_watermark=self.config.low_watermark,
                critical_threshold=self.config.critical_threshold,
            )
            self.memory_monitor.set_eviction_callback(self._on_monitor_pressure)

        # Set up eviction callback
        self.memory_pool.set_eviction_callback(self._on_memory_pressure)

    def lookup(self, tokens: List[int]) -> LookupResult:
        """
        Look up cached KV for a token sequence.

        Args:
            tokens: Token sequence to look up.

        Returns:
            LookupResult with matched length and KV cache if available.
        """
        with self.metrics.lookup_timer():
            result = self.prefix_tree.lookup(tokens)

        # Update stats
        if self.config.enable_stats:
            self.stats.record_lookup(
                hit=result.has_match,
                matched_tokens=result.matched_length,
                computed_tokens=len(tokens) - result.matched_length,
            )
        self.metrics.record_lookup(hit=result.has_match, matched_tokens=result.matched_length)

        return result

    def insert(
        self,
        tokens: List[int],
        key_cache: Tensor,
        value_cache: Tensor,
        normalized: bool = False,
    ) -> CacheBlock:
        """
        Insert KV cache for a token sequence.

        Args:
            tokens: Token sequence.
            key_cache: Key tensor [num_layers, seq_len, num_heads, head_dim].
            value_cache: Value tensor [num_layers, seq_len, num_heads, head_dim].
            normalized: Whether KV are position-normalized.

        Returns:
            Created CacheBlock.
        """
        # Check memory and evict if needed
        required_memory = key_cache.numel() * key_cache.element_size() * 2
        self._ensure_memory(required_memory)

        # Create and register block
        cache_block = CacheBlock.from_kv(key_cache, value_cache, normalized)
        self.memory_pool.register(cache_block)

        # Insert into prefix tree
        self.prefix_tree.insert(tokens, cache_block)

        # Update memory stats
        if self.config.enable_stats:
            self.stats.update_memory_peak(
                self.memory_pool.gpu_used,
                self.memory_pool.cpu_used,
            )

        return cache_block

    def compute_incremental(
        self,
        tokens: List[int],
        compute_fn: KVComputeFunc,
        store_result: bool = True,
    ) -> IncrementalResult:
        """
        Compute KV cache incrementally, reusing cached prefixes.

        Args:
            tokens: Token sequence.
            compute_fn: Function to compute KV for new tokens.
            store_result: Whether to store the result in cache.

        Returns:
            IncrementalResult with full KV cache.
        """
        # Check memory before computation
        estimated_memory = len(tokens) * self.config.kv_size_per_token
        self._ensure_memory(estimated_memory)

        result = self.engine.compute(tokens, compute_fn, store_result)

        # Update stats
        if self.config.enable_stats:
            self.stats.record_lookup(
                hit=result.cache_hit,
                matched_tokens=result.matched_length,
                computed_tokens=result.computed_length,
            )
            self.stats.update_memory_peak(
                self.memory_pool.gpu_used,
                self.memory_pool.cpu_used,
            )

        return result

    def compute_batch(
        self,
        batch_tokens: List[List[int]],
        compute_fn: KVComputeFunc,
        store_results: bool = True,
    ) -> List[IncrementalResult]:
        """
        Compute KV cache for a batch of sequences.

        Sequences with shared prefixes are grouped for efficiency.

        Args:
            batch_tokens: List of token sequences.
            compute_fn: KV computation function.
            store_results: Whether to store results.

        Returns:
            List of IncrementalResults.
        """
        return self.engine.compute_batch(batch_tokens, compute_fn, store_results)

    def evict_if_needed(self, required_memory: int = 0) -> EvictionResult:
        """
        Trigger eviction if memory usage exceeds threshold.

        Args:
            required_memory: Additional memory needed.

        Returns:
            EvictionResult with eviction statistics.
        """
        result = EvictionResult()

        # Check if eviction is needed
        current_util = self.memory_pool.gpu_utilization
        if current_util < self.config.eviction_threshold and required_memory == 0:
            return result

        # Calculate how much to free
        target_util = self.config.target_utilization
        current_usage = self.memory_pool.gpu_used
        target_usage = int(self.memory_pool.gpu_limit * target_util)
        to_free = max(current_usage - target_usage, required_memory)

        if to_free <= 0:
            return result

        # Execute eviction
        result = self.eviction_policy.evict(
            self.prefix_tree,
            self.memory_pool,
            to_free,
        )

        # Update stats
        if self.config.enable_stats:
            for _ in range(result.num_offloaded):
                self.stats.record_eviction(offloaded=True)
            for _ in range(result.num_deleted):
                self.stats.record_eviction(offloaded=False)

        return result

    def prefetch(self, tokens_list: List[List[int]]) -> List[bool]:
        """
        Prefetch cache blocks to GPU for upcoming requests.

        Args:
            tokens_list: List of token sequences to prefetch.

        Returns:
            List of booleans indicating which sequences were found.
        """
        return self.engine.prefetch(tokens_list)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.stats.to_dict()

    def get_memory_stats(self) -> MemoryStats:
        """Get memory usage statistics."""
        return self.memory_pool.get_stats()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats.reset()
        self.engine.reset_stats()

    def clear(self) -> None:
        """Clear all cached data."""
        self.prefix_tree.clear()
        self.memory_pool.clear()
        self.reset_stats()

    def _ensure_memory(self, required: int) -> None:
        """Ensure sufficient GPU memory is available."""
        # Check real GPU memory via monitor if available
        if self.memory_monitor is not None and not self.memory_monitor.can_allocate(required):
            to_free = self.memory_monitor.bytes_to_free()
            to_free = max(to_free, required)
            self.evict_if_needed(to_free)
            return

        if required > self.memory_pool.gpu_free:
            self.evict_if_needed(required)

    def _on_memory_pressure(self, block_ids: List[int]) -> None:
        """Callback when memory pool detects pressure."""
        self.evict_if_needed()

    def _on_monitor_pressure(self, bytes_to_free: int, pressure: MemoryPressure) -> None:
        """Callback from GPU memory monitor when watermarks are exceeded."""
        logger.info(
            "GPU memory pressure: %s, need to free %d MB",
            pressure.value,
            bytes_to_free / (1024 * 1024),
        )
        with self.metrics.eviction_timer():
            self.evict_if_needed(bytes_to_free)

    @property
    def num_cached_sequences(self) -> int:
        """Number of cached sequences."""
        return self.prefix_tree.num_cached

    @property
    def cache_size_tokens(self) -> int:
        """Total tokens in cache."""
        return self.prefix_tree.size

    def __repr__(self) -> str:
        return (
            f"DeltaCacheManager("
            f"sequences={self.num_cached_sequences}, "
            f"tokens={self.cache_size_tokens}, "
            f"gpu_util={self.memory_pool.gpu_utilization:.1%}, "
            f"hit_rate={self.stats.hit_rate:.1%})"
        )


def create_delta_cache(
    model_name: Optional[str] = None,
    **config_overrides,
) -> DeltaCacheManager:
    """
    Convenience function to create a DeltaCacheManager.

    Args:
        model_name: Optional model name for auto-configuration.
        **config_overrides: Configuration overrides.

    Returns:
        Configured DeltaCacheManager.
    """
    if model_name:
        config = DeltaCacheConfig.for_model(model_name, **config_overrides)
    else:
        config = DeltaCacheConfig(**config_overrides)

    return DeltaCacheManager(config)
