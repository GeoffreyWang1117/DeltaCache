"""KV cache block definition for DeltaCache."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum

import torch
from torch import Tensor


class DeviceType(Enum):
    """Device type for cache block storage."""
    GPU = "cuda"
    CPU = "cpu"


@dataclass
class CacheBlockMetadata:
    """Metadata for a cache block."""
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    ref_count: int = 0

    def touch(self) -> None:
        """Update access time and count."""
        self.last_access = time.time()
        self.access_count += 1

    def add_ref(self) -> None:
        """Increment reference count."""
        self.ref_count += 1

    def release_ref(self) -> None:
        """Decrement reference count."""
        self.ref_count = max(0, self.ref_count - 1)

    @property
    def is_referenced(self) -> bool:
        """Check if block is still referenced."""
        return self.ref_count > 0


class CacheBlock:
    """
    A block of KV cache for a sequence of tokens.

    Stores key and value tensors for a contiguous range of token positions,
    along with metadata for cache management decisions.

    Attributes:
        block_id: Unique identifier for this block.
        num_tokens: Number of tokens this block covers.
        key_cache: Key tensor of shape [num_layers, num_tokens, num_heads, head_dim].
        value_cache: Value tensor of shape [num_layers, num_tokens, num_heads, head_dim].
        metadata: Access and reference metadata.
        device: Current device where tensors are stored.
    """

    _next_id: int = 0

    def __init__(
        self,
        key_cache: Tensor,
        value_cache: Tensor,
        block_id: Optional[int] = None,
        normalized: bool = False,
    ) -> None:
        """
        Initialize a cache block.

        Args:
            key_cache: Key tensor [num_layers, num_tokens, num_heads, head_dim].
            value_cache: Value tensor [num_layers, num_tokens, num_heads, head_dim].
            block_id: Optional block ID. If None, auto-generated.
            normalized: Whether KV are stored without position encoding (for RoPE).
        """
        if block_id is None:
            block_id = CacheBlock._next_id
            CacheBlock._next_id += 1

        self.block_id = block_id
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.normalized = normalized
        self.metadata = CacheBlockMetadata()
        self._device_type = self._get_device_type()

    def _get_device_type(self) -> DeviceType:
        """Get device type from tensor."""
        if self.key_cache.is_cuda:
            return DeviceType.GPU
        return DeviceType.CPU

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        return self.key_cache.shape[0]

    @property
    def num_tokens(self) -> int:
        """Number of tokens in this block."""
        return self.key_cache.shape[1]

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return self.key_cache.shape[2]

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.key_cache.shape[3]

    @property
    def device(self) -> torch.device:
        """Current device of tensors."""
        return self.key_cache.device

    @property
    def device_type(self) -> DeviceType:
        """Device type (GPU or CPU)."""
        return self._device_type

    @property
    def is_on_gpu(self) -> bool:
        """Check if block is on GPU."""
        return self._device_type == DeviceType.GPU

    @property
    def memory_size(self) -> int:
        """Total memory size in bytes."""
        return self.key_cache.numel() * self.key_cache.element_size() + \
               self.value_cache.numel() * self.value_cache.element_size()

    def touch(self) -> None:
        """Update access metadata."""
        self.metadata.touch()

    def add_ref(self) -> None:
        """Add a reference to this block."""
        self.metadata.add_ref()

    def release_ref(self) -> None:
        """Release a reference to this block."""
        self.metadata.release_ref()

    def to_gpu(self, device: Optional[torch.device] = None) -> CacheBlock:
        """
        Move block to GPU.

        Args:
            device: Target CUDA device. If None, uses default.

        Returns:
            Self for chaining.
        """
        if self.is_on_gpu:
            return self

        target = device or torch.device("cuda")
        self.key_cache = self.key_cache.to(target, non_blocking=True)
        self.value_cache = self.value_cache.to(target, non_blocking=True)
        self._device_type = DeviceType.GPU
        return self

    def to_cpu(self) -> CacheBlock:
        """
        Move block to CPU.

        Returns:
            Self for chaining.
        """
        if not self.is_on_gpu:
            return self

        self.key_cache = self.key_cache.to("cpu", non_blocking=True)
        self.value_cache = self.value_cache.to("cpu", non_blocking=True)
        self._device_type = DeviceType.CPU
        return self

    def get_kv(self) -> Tuple[Tensor, Tensor]:
        """
        Get key and value tensors.

        Returns:
            Tuple of (key_cache, value_cache).
        """
        self.touch()
        return self.key_cache, self.value_cache

    def get_kv_for_positions(
        self,
        start: int,
        end: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get KV for a range of positions.

        Args:
            start: Start position (inclusive).
            end: End position (exclusive). If None, goes to end.

        Returns:
            Tuple of (key_cache, value_cache) sliced to the range.
        """
        self.touch()
        if end is None:
            end = self.num_tokens
        return self.key_cache[:, start:end], self.value_cache[:, start:end]

    def clone(self) -> CacheBlock:
        """Create a copy of this block."""
        return CacheBlock(
            key_cache=self.key_cache.clone(),
            value_cache=self.value_cache.clone(),
            normalized=self.normalized,
        )

    @classmethod
    def create_empty(
        cls,
        num_layers: int,
        num_tokens: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> CacheBlock:
        """
        Create an empty cache block.

        Args:
            num_layers: Number of transformer layers.
            num_tokens: Number of tokens.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            dtype: Data type for tensors.
            device: Device to create tensors on.

        Returns:
            Empty cache block.
        """
        shape = (num_layers, num_tokens, num_heads, head_dim)
        key_cache = torch.empty(shape, dtype=dtype, device=device)
        value_cache = torch.empty(shape, dtype=dtype, device=device)
        return cls(key_cache, value_cache)

    @classmethod
    def from_kv(
        cls,
        key: Tensor,
        value: Tensor,
        normalized: bool = False,
    ) -> CacheBlock:
        """
        Create a cache block from key and value tensors.

        Args:
            key: Key tensor.
            value: Value tensor.
            normalized: Whether KV are position-normalized.

        Returns:
            New cache block.
        """
        return cls(key_cache=key, value_cache=value, normalized=normalized)

    def __repr__(self) -> str:
        return (
            f"CacheBlock(id={self.block_id}, tokens={self.num_tokens}, "
            f"device={self.device_type.value}, refs={self.metadata.ref_count}, "
            f"accesses={self.metadata.access_count})"
        )
