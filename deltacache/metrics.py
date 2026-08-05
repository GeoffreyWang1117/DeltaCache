"""Prometheus metrics for DeltaCache.

Optional module - requires `prometheus_client` package.
Install via: pip install deltacache[monitoring]

When prometheus_client is not installed, all metric operations are no-ops.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


class _NoOpMetric:
    """No-op metric stub when prometheus_client is not installed."""

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, amount: float) -> None:
        pass

    def labels(self, **kwargs: str) -> "_NoOpMetric":
        return self


_NOOP = _NoOpMetric()


class DeltaCacheMetrics:
    """Prometheus metrics for cache monitoring.

    All metrics are prefixed with `deltacache_`. When prometheus_client
    is not installed, all operations are no-ops with zero overhead.

    Usage:
        metrics = DeltaCacheMetrics()
        metrics.record_lookup(hit=True)
        metrics.record_eviction(offloaded=True)

        # Or use as context manager for latency tracking:
        with metrics.lookup_timer():
            result = cache.lookup(tokens)
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled and _PROMETHEUS_AVAILABLE

        if self._enabled:
            self.lookups_total = Counter(
                "deltacache_lookups_total",
                "Total cache lookups",
            )
            self.hits_total = Counter(
                "deltacache_hits_total",
                "Cache hits",
            )
            self.misses_total = Counter(
                "deltacache_misses_total",
                "Cache misses",
            )
            self.hit_rate = Gauge(
                "deltacache_hit_rate",
                "Current cache hit rate",
            )
            self.gpu_memory_bytes = Gauge(
                "deltacache_gpu_memory_bytes",
                "GPU memory used by cache",
            )
            self.cpu_memory_bytes = Gauge(
                "deltacache_cpu_memory_bytes",
                "CPU memory used by cache",
            )
            self.gpu_utilization = Gauge(
                "deltacache_gpu_utilization",
                "GPU memory utilization ratio",
            )
            self.evictions_total = Counter(
                "deltacache_evictions_total",
                "Total evictions",
                ["action"],  # "offload" or "delete"
            )
            self.prefetches_total = Counter(
                "deltacache_prefetches_total",
                "CPU-to-GPU prefetches",
                ["result"],  # "hit" or "miss"
            )
            self.cached_sequences = Gauge(
                "deltacache_cached_sequences",
                "Number of cached sequences",
            )
            self.cached_tokens = Gauge(
                "deltacache_cached_tokens",
                "Total cached tokens",
            )
            self.lookup_duration = Histogram(
                "deltacache_lookup_duration_seconds",
                "Lookup latency",
                buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
            )
            self.eviction_duration = Histogram(
                "deltacache_eviction_duration_seconds",
                "Eviction latency",
                buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
            )
            self.transfer_duration = Histogram(
                "deltacache_transfer_duration_seconds",
                "GPU-CPU transfer latency",
                buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
            )

            # Internal counters for hit rate calculation
            self._total_lookups = 0
            self._total_hits = 0
        else:
            # No-op stubs
            self.lookups_total = _NOOP
            self.hits_total = _NOOP
            self.misses_total = _NOOP
            self.hit_rate = _NOOP
            self.gpu_memory_bytes = _NOOP
            self.cpu_memory_bytes = _NOOP
            self.gpu_utilization = _NOOP
            self.evictions_total = _NOOP
            self.prefetches_total = _NOOP
            self.cached_sequences = _NOOP
            self.cached_tokens = _NOOP
            self.lookup_duration = _NOOP
            self.eviction_duration = _NOOP
            self.transfer_duration = _NOOP
            self._total_lookups = 0
            self._total_hits = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record_lookup(self, hit: bool, matched_tokens: int = 0) -> None:
        """Record a cache lookup."""
        self.lookups_total.inc()
        self._total_lookups += 1
        if hit:
            self.hits_total.inc()
            self._total_hits += 1
        else:
            self.misses_total.inc()

        if self._enabled and self._total_lookups > 0:
            self.hit_rate.set(self._total_hits / self._total_lookups)

    def record_eviction(self, offloaded: bool) -> None:
        """Record an eviction event."""
        action = "offload" if offloaded else "delete"
        self.evictions_total.labels(action=action).inc()

    def record_prefetch(self, hit: bool) -> None:
        """Record a prefetch attempt."""
        result = "hit" if hit else "miss"
        self.prefetches_total.labels(result=result).inc()

    def update_memory(
        self,
        gpu_bytes: int,
        cpu_bytes: int,
        gpu_util: float,
    ) -> None:
        """Update memory usage gauges."""
        self.gpu_memory_bytes.set(gpu_bytes)
        self.cpu_memory_bytes.set(cpu_bytes)
        self.gpu_utilization.set(gpu_util)

    def update_cache_size(self, sequences: int, tokens: int) -> None:
        """Update cache size gauges."""
        self.cached_sequences.set(sequences)
        self.cached_tokens.set(tokens)

    @contextmanager
    def lookup_timer(self) -> Generator[None, None, None]:
        """Context manager to time lookup operations."""
        start = time.perf_counter()
        yield
        if self._enabled:
            self.lookup_duration.observe(time.perf_counter() - start)

    @contextmanager
    def eviction_timer(self) -> Generator[None, None, None]:
        """Context manager to time eviction operations."""
        start = time.perf_counter()
        yield
        if self._enabled:
            self.eviction_duration.observe(time.perf_counter() - start)

    @contextmanager
    def transfer_timer(self) -> Generator[None, None, None]:
        """Context manager to time GPU-CPU transfers."""
        start = time.perf_counter()
        yield
        if self._enabled:
            self.transfer_duration.observe(time.perf_counter() - start)
