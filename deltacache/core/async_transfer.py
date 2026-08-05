"""Async GPU-CPU transfer manager for tiered KV cache.

Handles non-blocking data transfer between GPU and CPU using CUDA streams
and pinned memory. Designed for PCIe 3.0 x16 (~12 GB/s) on RTX 3090.

Key design decisions (from PyTorch docs):
- Pre-allocate pinned CPU buffers (NOT .pin_memory() on existing tensors)
- Use dedicated CUDA streams for transfers (separate from compute)
- Use record_stream() to prevent allocator reuse before transfer completes
- Synchronize only when transferred data is actually needed
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class TransferRequest:
    """A pending async transfer."""

    block_id: int
    event: Optional[torch.cuda.Event] = None
    stream: Optional[torch.cuda.Stream] = None
    direction: str = "d2h"  # "d2h" (GPU->CPU) or "h2d" (CPU->GPU)
    completed: bool = False


class AsyncTransferManager:
    """Manages async GPU-CPU transfers with CUDA streams.

    Provides non-blocking offload (GPU->CPU) and prefetch (CPU->GPU)
    operations that can overlap with model computation.

    Args:
        device: CUDA device for GPU operations.
        max_pending: Maximum concurrent pending transfers.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        max_pending: int = 16,
    ) -> None:
        self._device = device or torch.device("cuda")
        self._max_pending = max_pending
        self._lock = threading.Lock()

        # Dedicated CUDA streams for transfers
        self._offload_stream: Optional[torch.cuda.Stream] = None
        self._prefetch_stream: Optional[torch.cuda.Stream] = None

        # Pending transfer tracking
        self._pending: Dict[int, TransferRequest] = {}

        # Initialize streams if CUDA is available
        if torch.cuda.is_available() and self._device.type == "cuda":
            self._offload_stream = torch.cuda.Stream(device=self._device)
            self._prefetch_stream = torch.cuda.Stream(device=self._device)

    @property
    def has_cuda(self) -> bool:
        return self._offload_stream is not None

    @property
    def num_pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def offload_async(
        self,
        block_id: int,
        key_gpu: Tensor,
        value_gpu: Tensor,
        key_cpu: Optional[Tensor] = None,
        value_cpu: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Start async GPU->CPU transfer.

        Args:
            block_id: Cache block identifier.
            key_gpu: Key tensor on GPU.
            value_gpu: Value tensor on GPU.
            key_cpu: Pre-allocated pinned CPU buffer for keys. Created if None.
            value_cpu: Pre-allocated pinned CPU buffer for values. Created if None.

        Returns:
            Tuple of (key_cpu, value_cpu) tensors. Data may not be ready yet -
            call wait_transfer() or is_complete() before reading.
        """
        if not self.has_cuda:
            # Fallback: synchronous transfer
            return key_gpu.to("cpu"), value_gpu.to("cpu")

        # Allocate pinned CPU buffers if not provided
        if key_cpu is None:
            key_cpu = torch.empty_like(key_gpu, device="cpu").pin_memory()
        if value_cpu is None:
            value_cpu = torch.empty_like(value_gpu, device="cpu").pin_memory()

        with torch.cuda.stream(self._offload_stream):
            key_cpu.copy_(key_gpu, non_blocking=True)
            value_cpu.copy_(value_gpu, non_blocking=True)
            event = self._offload_stream.record_event()

        with self._lock:
            self._pending[block_id] = TransferRequest(
                block_id=block_id,
                event=event,
                stream=self._offload_stream,
                direction="d2h",
            )

        logger.debug("Started async offload for block %d", block_id)
        return key_cpu, value_cpu

    def prefetch_async(
        self,
        block_id: int,
        key_cpu: Tensor,
        value_cpu: Tensor,
        key_gpu: Optional[Tensor] = None,
        value_gpu: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Start async CPU->GPU prefetch.

        Args:
            block_id: Cache block identifier.
            key_cpu: Key tensor on CPU (should be pinned for best performance).
            value_cpu: Value tensor on CPU (should be pinned).
            key_gpu: Pre-allocated GPU buffer for keys. Created if None.
            value_gpu: Pre-allocated GPU buffer for values. Created if None.

        Returns:
            Tuple of (key_gpu, value_gpu) tensors. Data may not be ready yet.
        """
        if not self.has_cuda:
            return key_cpu.to(self._device), value_cpu.to(self._device)

        if key_gpu is None:
            key_gpu = torch.empty_like(key_cpu, device=self._device)
        if value_gpu is None:
            value_gpu = torch.empty_like(value_cpu, device=self._device)

        with torch.cuda.stream(self._prefetch_stream):
            key_gpu.copy_(key_cpu, non_blocking=True)
            value_gpu.copy_(value_cpu, non_blocking=True)
            event = self._prefetch_stream.record_event()

        # Prevent allocator from reusing GPU buffers before transfer completes
        key_gpu.record_stream(self._prefetch_stream)
        value_gpu.record_stream(self._prefetch_stream)

        with self._lock:
            self._pending[block_id] = TransferRequest(
                block_id=block_id,
                event=event,
                stream=self._prefetch_stream,
                direction="h2d",
            )

        logger.debug("Started async prefetch for block %d", block_id)
        return key_gpu, value_gpu

    def wait_transfer(self, block_id: int) -> bool:
        """Wait for a specific transfer to complete.

        Args:
            block_id: Cache block identifier.

        Returns:
            True if transfer was found and completed, False if not found.
        """
        with self._lock:
            request = self._pending.get(block_id)
            if request is None:
                return False

        if request.event is not None:
            request.event.synchronize()

        with self._lock:
            request.completed = True
            del self._pending[block_id]

        logger.debug("Transfer complete for block %d", block_id)
        return True

    def is_complete(self, block_id: int) -> bool:
        """Check if a transfer has completed without blocking.

        Args:
            block_id: Cache block identifier.

        Returns:
            True if complete or not found, False if still in progress.
        """
        with self._lock:
            request = self._pending.get(block_id)
            if request is None:
                return True  # Not found = already completed or never started

        if request.event is not None and request.event.query():
            with self._lock:
                request.completed = True
                del self._pending[block_id]
            return True

        return False

    def wait_all(self) -> int:
        """Wait for all pending transfers to complete.

        Returns:
            Number of transfers completed.
        """
        with self._lock:
            pending_ids = list(self._pending.keys())

        count = 0
        for block_id in pending_ids:
            if self.wait_transfer(block_id):
                count += 1

        return count

    def cleanup_completed(self) -> int:
        """Remove completed transfers from tracking.

        Returns:
            Number of transfers cleaned up.
        """
        with self._lock:
            completed = [
                bid
                for bid, req in self._pending.items()
                if req.event is not None and req.event.query()
            ]
            for bid in completed:
                del self._pending[bid]
            return len(completed)
