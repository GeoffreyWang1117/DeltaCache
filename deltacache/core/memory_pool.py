"""Memory pool management for DeltaCache."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Callable

import torch
from torch import Tensor

from deltacache.core.cache_block import CacheBlock, DeviceType


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_bytes: int
    used_bytes: int
    free_bytes: int
    num_blocks: int
    num_gpu_blocks: int
    num_cpu_blocks: int

    @property
    def utilization(self) -> float:
        """Memory utilization ratio (0.0 to 1.0)."""
        if self.total_bytes == 0:
            return 0.0
        return self.used_bytes / self.total_bytes


class MemoryPool:
    """
    Memory pool for managing KV cache blocks across GPU and CPU.

    Provides unified allocation/deallocation interface and tracks memory usage.
    Cooperates with PyTorch's CUDA caching allocator.

    Attributes:
        gpu_limit: Maximum GPU memory in bytes.
        cpu_limit: Maximum CPU memory in bytes.
    """

    def __init__(
        self,
        gpu_limit: int = 0,
        cpu_limit: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize memory pool.

        Args:
            gpu_limit: Maximum GPU memory in bytes. 0 means auto-detect.
            cpu_limit: Maximum CPU memory in bytes. 0 means unlimited.
            device: CUDA device to use.
        """
        self._lock = threading.RLock()
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Auto-detect GPU memory if not specified and device is CUDA
        if gpu_limit == 0 and self._device.type == "cuda" and torch.cuda.is_available():
            gpu_limit = int(torch.cuda.get_device_properties(self._device).total_memory * 0.9)

        self.gpu_limit = gpu_limit
        self.cpu_limit = cpu_limit

        # Block tracking
        self._blocks: Dict[int, CacheBlock] = {}
        self._gpu_blocks: Set[int] = set()
        self._cpu_blocks: Set[int] = set()

        # Memory usage tracking
        self._gpu_used: int = 0
        self._cpu_used: int = 0

        # Eviction callback
        self._eviction_callback: Optional[Callable[[List[int]], None]] = None

    @property
    def gpu_used(self) -> int:
        """GPU memory currently used in bytes."""
        return self._gpu_used

    @property
    def cpu_used(self) -> int:
        """CPU memory currently used in bytes."""
        return self._cpu_used

    @property
    def gpu_free(self) -> int:
        """Free GPU memory in bytes."""
        return max(0, self.gpu_limit - self._gpu_used)

    @property
    def cpu_free(self) -> int:
        """Free CPU memory in bytes."""
        if self.cpu_limit == 0:
            return float('inf')
        return max(0, self.cpu_limit - self._cpu_used)

    @property
    def gpu_utilization(self) -> float:
        """GPU memory utilization (0.0 to 1.0)."""
        if self.gpu_limit == 0:
            return 0.0
        return self._gpu_used / self.gpu_limit

    def set_eviction_callback(self, callback: Callable[[List[int]], None]) -> None:
        """Set callback to be invoked when eviction is needed."""
        self._eviction_callback = callback

    def allocate(
        self,
        num_layers: int,
        num_tokens: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        on_gpu: bool = True,
    ) -> Optional[CacheBlock]:
        """
        Allocate a new cache block.

        Args:
            num_layers: Number of transformer layers.
            num_tokens: Number of tokens.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            dtype: Data type for tensors.
            on_gpu: Whether to allocate on GPU.

        Returns:
            Allocated cache block, or None if allocation failed.
        """
        # Calculate required memory
        element_size = torch.tensor([], dtype=dtype).element_size()
        required = 2 * num_layers * num_tokens * num_heads * head_dim * element_size

        with self._lock:
            if on_gpu:
                if required > self.gpu_free:
                    # Try to trigger eviction
                    if self._eviction_callback:
                        self._eviction_callback([])
                    if required > self.gpu_free:
                        return None
                device = self._device
            else:
                if self.cpu_limit > 0 and required > self.cpu_free:
                    return None
                device = torch.device("cpu")

            # Allocate block
            block = CacheBlock.create_empty(
                num_layers=num_layers,
                num_tokens=num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=str(device),
            )

            # Track allocation
            self._register_block(block)

            return block

    def register(self, block: CacheBlock) -> None:
        """
        Register an externally created block with the pool.

        Args:
            block: Cache block to register.
        """
        with self._lock:
            self._register_block(block)

    def _register_block(self, block: CacheBlock) -> None:
        """Internal block registration."""
        self._blocks[block.block_id] = block

        if block.is_on_gpu:
            self._gpu_blocks.add(block.block_id)
            self._gpu_used += block.memory_size
        else:
            self._cpu_blocks.add(block.block_id)
            self._cpu_used += block.memory_size

    def free(self, block_id: int) -> bool:
        """
        Free a cache block.

        Args:
            block_id: ID of block to free.

        Returns:
            True if block was freed, False if not found.
        """
        with self._lock:
            if block_id not in self._blocks:
                return False

            block = self._blocks[block_id]

            # Update tracking
            if block.is_on_gpu:
                self._gpu_blocks.discard(block_id)
                self._gpu_used -= block.memory_size
            else:
                self._cpu_blocks.discard(block_id)
                self._cpu_used -= block.memory_size

            del self._blocks[block_id]

            # Help garbage collection
            del block.key_cache
            del block.value_cache

            return True

    def move_to_cpu(self, block_id: int) -> bool:
        """
        Move a block from GPU to CPU.

        Args:
            block_id: ID of block to move.

        Returns:
            True if moved, False if not found or already on CPU.
        """
        with self._lock:
            if block_id not in self._blocks:
                return False

            block = self._blocks[block_id]
            if not block.is_on_gpu:
                return False

            # Check CPU capacity
            if self.cpu_limit > 0 and block.memory_size > self.cpu_free:
                return False

            memory_size = block.memory_size

            # Move block
            block.to_cpu()

            # Update tracking
            self._gpu_blocks.discard(block_id)
            self._cpu_blocks.add(block_id)
            self._gpu_used -= memory_size
            self._cpu_used += memory_size

            return True

    def move_to_gpu(self, block_id: int, device: Optional[torch.device] = None) -> bool:
        """
        Move a block from CPU to GPU.

        Args:
            block_id: ID of block to move.
            device: Target GPU device.

        Returns:
            True if moved, False if not found or no GPU space.
        """
        with self._lock:
            if block_id not in self._blocks:
                return False

            block = self._blocks[block_id]
            if block.is_on_gpu:
                return False

            # Check GPU capacity
            if block.memory_size > self.gpu_free:
                return False

            memory_size = block.memory_size

            # Move block
            block.to_gpu(device or self._device)

            # Update tracking
            self._cpu_blocks.discard(block_id)
            self._gpu_blocks.add(block_id)
            self._cpu_used -= memory_size
            self._gpu_used += memory_size

            return True

    def get_block(self, block_id: int) -> Optional[CacheBlock]:
        """Get a block by ID."""
        return self._blocks.get(block_id)

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        with self._lock:
            return MemoryStats(
                total_bytes=self.gpu_limit + self.cpu_limit,
                used_bytes=self._gpu_used + self._cpu_used,
                free_bytes=self.gpu_free + (self.cpu_free if self.cpu_limit > 0 else 0),
                num_blocks=len(self._blocks),
                num_gpu_blocks=len(self._gpu_blocks),
                num_cpu_blocks=len(self._cpu_blocks),
            )

    def get_gpu_block_ids(self) -> List[int]:
        """Get list of block IDs on GPU."""
        with self._lock:
            return list(self._gpu_blocks)

    def get_cpu_block_ids(self) -> List[int]:
        """Get list of block IDs on CPU."""
        with self._lock:
            return list(self._cpu_blocks)

    def clear(self) -> None:
        """Clear all blocks from the pool."""
        with self._lock:
            block_ids = list(self._blocks.keys())
            for block_id in block_ids:
                self.free(block_id)

    def __len__(self) -> int:
        """Number of blocks in pool."""
        return len(self._blocks)

    def __contains__(self, block_id: int) -> bool:
        """Check if block is in pool."""
        return block_id in self._blocks
