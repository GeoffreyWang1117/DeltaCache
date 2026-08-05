"""Tests for GPU memory monitor."""

import time
import threading
import pytest
import torch

from deltacache.core.memory_monitor import (
    GPUMemoryMonitor,
    MemoryPressure,
    MemoryState,
)


class TestMemoryState:
    """Tests for MemoryState dataclass."""

    def test_utilization(self):
        state = MemoryState(
            total_bytes=1000,
            free_bytes=300,
            allocated_bytes=500,
            reserved_bytes=700,
            timestamp=time.monotonic(),
        )
        assert state.utilization == 0.7
        assert state.used_bytes == 700
        assert state.available_for_cache == 300

    def test_zero_total(self):
        state = MemoryState(
            total_bytes=0,
            free_bytes=0,
            allocated_bytes=0,
            reserved_bytes=0,
            timestamp=time.monotonic(),
        )
        assert state.utilization == 0.0


class TestGPUMemoryMonitor:
    """Tests for GPUMemoryMonitor."""

    def test_init_valid(self):
        monitor = GPUMemoryMonitor(
            device=torch.device("cpu"),
            high_watermark=0.85,
            low_watermark=0.70,
            critical_threshold=0.95,
        )
        assert monitor.high_watermark == 0.85
        assert monitor.low_watermark == 0.70
        assert monitor.critical_threshold == 0.95

    def test_init_invalid_watermarks(self):
        with pytest.raises(ValueError, match="high_watermark"):
            GPUMemoryMonitor(
                device=torch.device("cpu"),
                high_watermark=0.70,
                low_watermark=0.85,
            )

    def test_init_invalid_critical(self):
        with pytest.raises(ValueError, match="critical_threshold"):
            GPUMemoryMonitor(
                device=torch.device("cpu"),
                high_watermark=0.85,
                low_watermark=0.70,
                critical_threshold=0.80,
            )

    def test_get_memory_state_cpu_device(self):
        """On CPU device, should return zero state."""
        monitor = GPUMemoryMonitor(device=torch.device("cpu"))
        state = monitor.get_memory_state()
        assert state.total_bytes == 0
        assert state.free_bytes == 0
        assert state.utilization == 0.0

    def test_pressure_levels(self):
        monitor = GPUMemoryMonitor(
            device=torch.device("cpu"),
            high_watermark=0.85,
            low_watermark=0.70,
            critical_threshold=0.95,
        )

        # Below low watermark
        state = MemoryState(1000, 500, 0, 0, time.monotonic())
        assert monitor.get_pressure_level(state) == MemoryPressure.NONE

        # Between low and high
        state = MemoryState(1000, 200, 0, 0, time.monotonic())
        assert monitor.get_pressure_level(state) == MemoryPressure.LOW

        # Above high watermark
        state = MemoryState(1000, 100, 0, 0, time.monotonic())
        assert monitor.get_pressure_level(state) == MemoryPressure.HIGH

        # Above critical
        state = MemoryState(1000, 30, 0, 0, time.monotonic())
        assert monitor.get_pressure_level(state) == MemoryPressure.CRITICAL

    def test_bytes_to_free(self):
        monitor = GPUMemoryMonitor(
            device=torch.device("cpu"),
            high_watermark=0.85,
            low_watermark=0.70,
        )

        # 90% used, need to get to 70%
        state = MemoryState(1000, 100, 0, 0, time.monotonic())
        to_free = monitor.bytes_to_free(state)
        assert to_free == 200  # 900 - 700 = 200

        # 50% used, no freeing needed
        state = MemoryState(1000, 500, 0, 0, time.monotonic())
        assert monitor.bytes_to_free(state) == 0

    def test_can_allocate(self):
        """On CPU device (no GPU), can_allocate should always return True."""
        monitor = GPUMemoryMonitor(device=torch.device("cpu"))
        # With 0 total bytes, utilization is 0, so should be below watermark
        assert monitor.can_allocate(100)

    def test_callback_on_pressure(self):
        """Test eviction callback is called on check_and_evict."""
        monitor = GPUMemoryMonitor(device=torch.device("cpu"))
        callback_called = []

        def on_evict(bytes_to_free, pressure):
            callback_called.append((bytes_to_free, pressure))

        monitor.set_eviction_callback(on_evict)
        # On CPU device, no pressure → no callback
        monitor.check_and_evict()
        assert len(callback_called) == 0

    def test_background_monitoring_lifecycle(self):
        monitor = GPUMemoryMonitor(
            device=torch.device("cpu"),
            poll_interval=0.05,
        )
        assert not monitor.is_monitoring

        monitor.start_background_monitoring()
        assert monitor.is_monitoring

        # Let it run a couple cycles
        time.sleep(0.15)

        monitor.stop_background_monitoring()
        assert not monitor.is_monitoring

    def test_repr(self):
        monitor = GPUMemoryMonitor(device=torch.device("cpu"))
        r = repr(monitor)
        assert "GPUMemoryMonitor" in r

        # After getting state
        monitor.get_memory_state()
        r = repr(monitor)
        assert "GPUMemoryMonitor" in r


class TestMemoryPressure:
    """Tests for MemoryPressure enum."""

    def test_values(self):
        assert MemoryPressure.NONE.value == "none"
        assert MemoryPressure.LOW.value == "low"
        assert MemoryPressure.HIGH.value == "high"
        assert MemoryPressure.CRITICAL.value == "critical"
