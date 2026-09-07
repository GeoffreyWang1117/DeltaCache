"""Hierarchical memory management for DeltaCache.

This module provides enhanced memory management with:
1. Smart GPU -> CPU offloading with async data transfer
2. Predictive prefetching from CPU to GPU
3. Memory pressure monitoring and proactive management
4. Batch operations for efficiency
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Dict, List, Optional

import torch

from deltacache.core.cache_block import CacheBlock
from deltacache.core.memory_pool import MemoryPool, MemoryStats

logger = logging.getLogger(__name__)


@dataclass
class OffloadRequest:
    """Request to offload a cache block."""

    block_id: int
    priority: float  # Lower = more urgent
    timestamp: float = field(default_factory=time.time)


@dataclass
class PrefetchRequest:
    """Request to prefetch a cache block to GPU."""

    block_id: int
    priority: float  # Lower = more urgent
    timestamp: float = field(default_factory=time.time)


@dataclass
class HierarchicalMemoryStats(MemoryStats):
    """Extended memory statistics for hierarchical memory."""

    num_offloads: int = 0
    num_prefetches: int = 0
    offload_bytes: int = 0
    prefetch_bytes: int = 0
    avg_offload_latency_ms: float = 0.0
    avg_prefetch_latency_ms: float = 0.0


class HierarchicalMemoryManager:
    """
    Hierarchical memory manager with GPU/CPU tiering.

    Features:
    1. Async offloading: Non-blocking GPU->CPU transfers
    2. Predictive prefetching: Move likely-needed data back to GPU
    3. Memory pressure monitoring: Proactive management before OOM
    4. Access pattern tracking: Learn which entries to keep hot

    Usage:
        manager = HierarchicalMemoryManager(gpu_limit=8*1024**3, cpu_limit=32*1024**3)

        # Register blocks
        manager.register(cache_block)

        # Track access patterns
        manager.record_access(block_id)

        # Proactive management
        manager.maintain()  # Call periodically
    """

    def __init__(
        self,
        gpu_limit: int = 0,
        cpu_limit: int = 0,
        device: Optional[torch.device] = None,
        gpu_high_watermark: float = 0.85,
        gpu_low_watermark: float = 0.70,
        cpu_high_watermark: float = 0.90,
        enable_async: bool = True,
        prefetch_threshold: float = 0.5,
    ) -> None:
        """
        Initialize hierarchical memory manager.

        Args:
            gpu_limit: Maximum GPU memory in bytes. 0 = auto-detect.
            cpu_limit: Maximum CPU memory in bytes. 0 = unlimited.
            device: CUDA device to use.
            gpu_high_watermark: GPU utilization to trigger offloading.
            gpu_low_watermark: Target GPU utilization after offloading.
            cpu_high_watermark: CPU utilization to trigger deletion.
            enable_async: Enable async data transfers.
            prefetch_threshold: Access frequency threshold for prefetching.
        """
        self._lock = threading.RLock()
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize base memory pool
        self._pool = MemoryPool(gpu_limit, cpu_limit, device)

        # Watermarks
        self.gpu_high_watermark = gpu_high_watermark
        self.gpu_low_watermark = gpu_low_watermark
        self.cpu_high_watermark = cpu_high_watermark
        self.prefetch_threshold = prefetch_threshold

        # Access tracking
        self._access_history: OrderedDict[int, List[float]] = OrderedDict()
        self._access_window = 100  # Keep last N accesses per block

        # Async operations
        self._enable_async = enable_async
        self._offload_queue: Queue[OffloadRequest] = Queue()
        self._prefetch_queue: Queue[PrefetchRequest] = Queue()

        # Statistics
        self._stats = {
            "num_offloads": 0,
            "num_prefetches": 0,
            "offload_bytes": 0,
            "prefetch_bytes": 0,
            "offload_latencies": [],
            "prefetch_latencies": [],
        }

        # Start background worker if async enabled
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_worker = threading.Event()
        if enable_async:
            self._start_worker()

    def _start_worker(self) -> None:
        """Start background worker for async operations."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="HierarchicalMemory-Worker"
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background worker for processing async operations."""
        while not self._stop_worker.is_set():
            try:
                # Process offload requests
                try:
                    request = self._offload_queue.get(timeout=0.1)
                    self._do_offload(request.block_id)
                except Empty:
                    pass

                # Process prefetch requests
                try:
                    request = self._prefetch_queue.get(timeout=0.1)
                    self._do_prefetch(request.block_id)
                except Empty:
                    pass

            except Exception as e:
                logger.error(f"Worker error: {e}")

    def stop(self) -> None:
        """Stop background worker."""
        self._stop_worker.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    @property
    def pool(self) -> MemoryPool:
        """Access underlying memory pool."""
        return self._pool

    @property
    def gpu_used(self) -> int:
        return self._pool.gpu_used

    @property
    def cpu_used(self) -> int:
        return self._pool.cpu_used

    @property
    def gpu_utilization(self) -> float:
        return self._pool.gpu_utilization

    def register(self, block: CacheBlock) -> None:
        """Register a cache block."""
        with self._lock:
            self._pool.register(block)
            self._access_history[block.block_id] = [time.time()]

    def record_access(self, block_id: int) -> None:
        """Record an access to a block for pattern tracking."""
        with self._lock:
            if block_id not in self._access_history:
                self._access_history[block_id] = []

            history = self._access_history[block_id]
            history.append(time.time())

            # Trim to window size
            if len(history) > self._access_window:
                self._access_history[block_id] = history[-self._access_window :]

    def get_access_frequency(self, block_id: int) -> float:
        """Get recent access frequency for a block (accesses per second)."""
        with self._lock:
            if block_id not in self._access_history:
                return 0.0

            history = self._access_history[block_id]
            if len(history) < 2:
                return 0.0

            # Calculate frequency over recent window
            time_span = history[-1] - history[0]
            if time_span <= 0:
                return 0.0

            return len(history) / time_span

    def should_prefetch(self, block_id: int) -> bool:
        """Determine if a block should be prefetched to GPU."""
        block = self._pool.get_block(block_id)
        if block is None or block.is_on_gpu:
            return False

        frequency = self.get_access_frequency(block_id)
        return frequency >= self.prefetch_threshold

    def offload(self, block_id: int, async_op: bool = True) -> bool:
        """
        Offload a block from GPU to CPU.

        Args:
            block_id: Block to offload.
            async_op: If True, queue for async processing.

        Returns:
            True if offload initiated.
        """
        if async_op and self._enable_async:
            self._offload_queue.put(OffloadRequest(block_id=block_id, priority=time.time()))
            return True
        else:
            return self._do_offload(block_id)

    def _do_offload(self, block_id: int) -> bool:
        """Perform actual offload operation."""
        start_time = time.time()

        with self._lock:
            success = self._pool.move_to_cpu(block_id)

            if success:
                block = self._pool.get_block(block_id)
                if block:
                    self._stats["num_offloads"] += 1
                    self._stats["offload_bytes"] += block.memory_size
                    latency = (time.time() - start_time) * 1000
                    self._stats["offload_latencies"].append(latency)

        return success

    def prefetch(self, block_id: int, async_op: bool = True) -> bool:
        """
        Prefetch a block from CPU to GPU.

        Args:
            block_id: Block to prefetch.
            async_op: If True, queue for async processing.

        Returns:
            True if prefetch initiated.
        """
        if async_op and self._enable_async:
            self._prefetch_queue.put(PrefetchRequest(block_id=block_id, priority=time.time()))
            return True
        else:
            return self._do_prefetch(block_id)

    def _do_prefetch(self, block_id: int) -> bool:
        """Perform actual prefetch operation."""
        start_time = time.time()

        with self._lock:
            success = self._pool.move_to_gpu(block_id, self._device)

            if success:
                block = self._pool.get_block(block_id)
                if block:
                    self._stats["num_prefetches"] += 1
                    self._stats["prefetch_bytes"] += block.memory_size
                    latency = (time.time() - start_time) * 1000
                    self._stats["prefetch_latencies"].append(latency)

        return success

    def maintain(self) -> Dict[str, int]:
        """
        Perform maintenance operations.

        Should be called periodically (e.g., between requests).
        Handles proactive offloading and prefetching.

        Returns:
            Statistics about operations performed.
        """
        result = {
            "offloaded": 0,
            "prefetched": 0,
            "deleted": 0,
        }

        with self._lock:
            # Check GPU pressure
            if self.gpu_utilization > self.gpu_high_watermark:
                result["offloaded"] = self._proactive_offload()

            # Check CPU pressure
            cpu_limit = self._pool.cpu_limit
            if cpu_limit > 0:
                cpu_util = self._pool.cpu_used / cpu_limit
                if cpu_util > self.cpu_high_watermark:
                    result["deleted"] = self._proactive_delete()

            # Prefetch hot blocks
            result["prefetched"] = self._proactive_prefetch()

        return result

    def _proactive_offload(self) -> int:
        """Proactively offload blocks to reduce GPU pressure."""
        target_used = self._pool.gpu_limit * self.gpu_low_watermark
        to_free = self._pool.gpu_used - target_used

        if to_free <= 0:
            return 0

        offloaded = 0
        freed = 0

        # Get GPU blocks sorted by access frequency (low frequency first)
        gpu_blocks = []
        for block_id in self._pool.get_gpu_block_ids():
            block = self._pool.get_block(block_id)
            if block and block.metadata.ref_count == 0:
                freq = self.get_access_frequency(block_id)
                gpu_blocks.append((freq, block_id, block.memory_size))

        # Sort by frequency (ascending)
        gpu_blocks.sort()

        for _freq, block_id, size in gpu_blocks:
            if freed >= to_free:
                break

            if self._do_offload(block_id):
                offloaded += 1
                freed += size

        return offloaded

    def _proactive_delete(self) -> int:
        """Proactively delete CPU blocks to reduce CPU pressure."""
        cpu_limit = self._pool.cpu_limit
        target_used = cpu_limit * 0.7  # Target 70% utilization
        to_free = self._pool.cpu_used - target_used

        if to_free <= 0:
            return 0

        deleted = 0
        freed = 0

        # Get CPU blocks sorted by access frequency
        cpu_blocks = []
        for block_id in self._pool.get_cpu_block_ids():
            block = self._pool.get_block(block_id)
            if block and block.metadata.ref_count == 0:
                freq = self.get_access_frequency(block_id)
                cpu_blocks.append((freq, block_id, block.memory_size))

        cpu_blocks.sort()

        for _freq, block_id, size in cpu_blocks:
            if freed >= to_free:
                break

            if self._pool.free(block_id):
                self._access_history.pop(block_id, None)
                deleted += 1
                freed += size

        return deleted

    def _proactive_prefetch(self) -> int:
        """Prefetch hot blocks from CPU to GPU."""
        # Only prefetch if we have GPU headroom
        if self.gpu_utilization > self.gpu_low_watermark:
            return 0

        prefetched = 0
        available_gpu = self._pool.gpu_free

        # Get CPU blocks sorted by access frequency (high first)
        cpu_blocks = []
        for block_id in self._pool.get_cpu_block_ids():
            block = self._pool.get_block(block_id)
            if block:
                freq = self.get_access_frequency(block_id)
                if freq >= self.prefetch_threshold:
                    cpu_blocks.append((freq, block_id, block.memory_size))

        # Sort by frequency (descending - hot first)
        cpu_blocks.sort(reverse=True)

        used = 0
        for _freq, block_id, size in cpu_blocks:
            if used + size > available_gpu:
                break

            if self._do_prefetch(block_id):
                prefetched += 1
                used += size

        return prefetched

    def get_stats(self) -> HierarchicalMemoryStats:
        """Get extended memory statistics."""
        base_stats = self._pool.get_stats()

        avg_offload = 0.0
        if self._stats["offload_latencies"]:
            avg_offload = sum(self._stats["offload_latencies"]) / len(
                self._stats["offload_latencies"]
            )

        avg_prefetch = 0.0
        if self._stats["prefetch_latencies"]:
            avg_prefetch = sum(self._stats["prefetch_latencies"]) / len(
                self._stats["prefetch_latencies"]
            )

        return HierarchicalMemoryStats(
            total_bytes=base_stats.total_bytes,
            used_bytes=base_stats.used_bytes,
            free_bytes=base_stats.free_bytes,
            num_blocks=base_stats.num_blocks,
            num_gpu_blocks=base_stats.num_gpu_blocks,
            num_cpu_blocks=base_stats.num_cpu_blocks,
            num_offloads=self._stats["num_offloads"],
            num_prefetches=self._stats["num_prefetches"],
            offload_bytes=self._stats["offload_bytes"],
            prefetch_bytes=self._stats["prefetch_bytes"],
            avg_offload_latency_ms=avg_offload,
            avg_prefetch_latency_ms=avg_prefetch,
        )

    def clear(self) -> None:
        """Clear all blocks and reset state."""
        with self._lock:
            self._pool.clear()
            self._access_history.clear()
            self._stats = {
                "num_offloads": 0,
                "num_prefetches": 0,
                "offload_bytes": 0,
                "prefetch_bytes": 0,
                "offload_latencies": [],
                "prefetch_latencies": [],
            }

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
