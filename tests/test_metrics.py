"""Tests for metrics module."""

import pytest

from deltacache.metrics import DeltaCacheMetrics


class TestDeltaCacheMetrics:
    """Tests for Prometheus metrics (no-op mode)."""

    def test_disabled_metrics_no_error(self):
        """Disabled metrics should not raise errors."""
        metrics = DeltaCacheMetrics(enabled=False)
        metrics.record_lookup(hit=True)
        metrics.record_lookup(hit=False)
        metrics.record_eviction(offloaded=True)
        metrics.record_eviction(offloaded=False)
        metrics.record_prefetch(hit=True)
        metrics.update_memory(gpu_bytes=100, cpu_bytes=200, gpu_util=0.5)
        metrics.update_cache_size(sequences=10, tokens=1000)

    def test_disabled_metrics_timer(self):
        """Timer context managers should work when disabled."""
        metrics = DeltaCacheMetrics(enabled=False)
        with metrics.lookup_timer():
            pass
        with metrics.eviction_timer():
            pass
        with metrics.transfer_timer():
            pass

    def test_enabled_false_attribute(self):
        metrics = DeltaCacheMetrics(enabled=False)
        assert not metrics.enabled

    def test_hit_rate_tracking(self):
        """Internal hit rate tracking should work even when prometheus is disabled."""
        metrics = DeltaCacheMetrics(enabled=False)
        metrics.record_lookup(hit=True)
        metrics.record_lookup(hit=True)
        metrics.record_lookup(hit=False)
        assert metrics._total_lookups == 3
        assert metrics._total_hits == 2
