"""GPU memory monitoring with proactive eviction for DeltaCache.

Monitors real GPU memory usage and triggers eviction before OOM occurs.
Uses high/low watermark pattern similar to Linux kernel's memory management.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class MemoryPressure(Enum):
    """Memory pressure levels."""

    NONE = "none"  # Below low watermark
    LOW = "low"  # Between low and high watermark
    HIGH = "high"  # Between high watermark and critical
    CRITICAL = "critical"  # Above critical threshold


@dataclass
class MemoryState:
    """Snapshot of GPU memory state."""

    total_bytes: int
    free_bytes: int
    allocated_bytes: int  # PyTorch allocated
    reserved_bytes: int  # PyTorch reserved (includes cached free blocks)
    timestamp: float

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.free_bytes

    @property
    def utilization(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.used_bytes / self.total_bytes

    @property
    def available_for_cache(self) -> int:
        """Bytes available for new cache allocations.

        Uses free bytes from the GPU driver level (mem_get_info),
        not PyTorch's view, since other processes may share the GPU.
        """
        return self.free_bytes


class GPUMemoryMonitor:
    """Monitors GPU memory and triggers proactive eviction.

    Uses a watermark-based approach:
    - Below LOW_WATERMARK: no action needed
    - Above HIGH_WATERMARK: start evicting to reach LOW_WATERMARK
    - Above CRITICAL: emergency eviction (skip CPU offload, delete directly)

    The monitor queries actual GPU state via torch.cuda.mem_get_info(),
    which reflects memory used by ALL processes on the GPU, not just PyTorch.

    Args:
        device: CUDA device to monitor.
        high_watermark: Utilization ratio that triggers eviction (default: 0.85).
        low_watermark: Target utilization after eviction (default: 0.70).
        critical_threshold: Utilization ratio for emergency eviction (default: 0.95).
        poll_interval: Seconds between polls in background mode (default: 0.1).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        high_watermark: float = 0.85,
        low_watermark: float = 0.70,
        critical_threshold: float = 0.95,
        poll_interval: float = 0.1,
    ) -> None:
        if high_watermark <= low_watermark:
            raise ValueError(
                f"high_watermark ({high_watermark}) must be > low_watermark ({low_watermark})"
            )
        if critical_threshold <= high_watermark:
            raise ValueError(
                f"critical_threshold ({critical_threshold}) must be > "
                f"high_watermark ({high_watermark})"
            )

        self._device = device or torch.device("cuda")
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.critical_threshold = critical_threshold
        self.poll_interval = poll_interval

        # Callbacks
        self._eviction_callback: Optional[Callable[[int, MemoryPressure], None]] = None

        # Background monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # State tracking
        self._last_state: Optional[MemoryState] = None
        self._peak_usage: int = 0

    @property
    def device(self) -> torch.device:
        return self._device

    def set_eviction_callback(
        self, callback: Callable[[int, MemoryPressure], None]
    ) -> None:
        """Set callback invoked when eviction is needed.

        Args:
            callback: Function(bytes_to_free, pressure_level) -> None
        """
        self._eviction_callback = callback

    def get_memory_state(self) -> MemoryState:
        """Get current GPU memory state.

        Returns actual GPU-level memory info, reflecting all processes.
        """
        if not torch.cuda.is_available() or self._device.type != "cuda":
            return MemoryState(
                total_bytes=0,
                free_bytes=0,
                allocated_bytes=0,
                reserved_bytes=0,
                timestamp=time.monotonic(),
            )

        free, total = torch.cuda.mem_get_info(self._device)
        allocated = torch.cuda.memory_allocated(self._device)
        reserved = torch.cuda.memory_reserved(self._device)

        state = MemoryState(
            total_bytes=total,
            free_bytes=free,
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            timestamp=time.monotonic(),
        )

        with self._lock:
            self._last_state = state
            self._peak_usage = max(self._peak_usage, state.used_bytes)

        return state

    def get_pressure_level(self, state: Optional[MemoryState] = None) -> MemoryPressure:
        """Determine current memory pressure level."""
        if state is None:
            state = self.get_memory_state()

        util = state.utilization
        if util >= self.critical_threshold:
            return MemoryPressure.CRITICAL
        elif util >= self.high_watermark:
            return MemoryPressure.HIGH
        elif util >= self.low_watermark:
            return MemoryPressure.LOW
        return MemoryPressure.NONE

    def bytes_to_free(self, state: Optional[MemoryState] = None) -> int:
        """Calculate bytes that should be freed to reach low watermark.

        Returns 0 if no eviction is needed.
        """
        if state is None:
            state = self.get_memory_state()

        target_used = int(state.total_bytes * self.low_watermark)
        to_free = state.used_bytes - target_used
        return max(0, to_free)

    def check_and_evict(self) -> MemoryPressure:
        """Check memory and trigger eviction if needed.

        This is the main entry point for synchronous (non-background) usage.
        Call this before allocating new cache blocks.

        Returns:
            Current memory pressure level.
        """
        state = self.get_memory_state()
        pressure = self.get_pressure_level(state)

        if pressure in (MemoryPressure.HIGH, MemoryPressure.CRITICAL):
            to_free = self.bytes_to_free(state)
            if to_free > 0 and self._eviction_callback:
                logger.debug(
                    "Memory pressure %s: utilization=%.1f%%, freeing %d MB",
                    pressure.value,
                    state.utilization * 100,
                    to_free / (1024 * 1024),
                )
                self._eviction_callback(to_free, pressure)

        return pressure

    def can_allocate(self, required_bytes: int) -> bool:
        """Check if allocation of given size is safe.

        Considers the high watermark - returns False if allocation
        would push usage above the high watermark.
        """
        state = self.get_memory_state()
        if state.total_bytes == 0:
            return True  # No GPU or not trackable
        projected_usage = (state.used_bytes + required_bytes) / state.total_bytes
        return projected_usage < self.high_watermark

    def start_background_monitoring(self) -> None:
        """Start background monitoring thread.

        The thread polls GPU memory at `poll_interval` and triggers
        eviction callbacks when watermarks are exceeded.
        """
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="deltacache-memory-monitor",
        )
        self._monitor_thread.start()
        logger.info(
            "Started background memory monitoring (poll=%.2fs, high=%.0f%%, low=%.0f%%)",
            self.poll_interval,
            self.high_watermark * 100,
            self.low_watermark * 100,
        )

    def stop_background_monitoring(self) -> None:
        """Stop background monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
            logger.info("Stopped background memory monitoring")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self.check_and_evict()
            except Exception:
                logger.exception("Error in memory monitor loop")
            self._stop_event.wait(self.poll_interval)

    @property
    def peak_usage_bytes(self) -> int:
        """Peak GPU memory usage observed."""
        return self._peak_usage

    @property
    def last_state(self) -> Optional[MemoryState]:
        """Last observed memory state."""
        with self._lock:
            return self._last_state

    @property
    def is_monitoring(self) -> bool:
        """Whether background monitoring is active."""
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    def __del__(self) -> None:
        if hasattr(self, "_stop_event"):
            self.stop_background_monitoring()

    def __repr__(self) -> str:
        state = self._last_state
        if state:
            return (
                f"GPUMemoryMonitor(device={self._device}, "
                f"util={state.utilization:.1%}, "
                f"pressure={self.get_pressure_level(state).value})"
            )
        return f"GPUMemoryMonitor(device={self._device}, no_state)"
