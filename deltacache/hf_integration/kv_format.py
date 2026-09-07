"""KV cache format conversion between HuggingFace and DeltaCache formats.

HuggingFace format:
    tuple[tuple[Tensor, Tensor], ...]
    Each layer: (key, value) with shape [batch, num_heads, seq_len, head_dim]

    OR (for transformers >= 4.36):
    DynamicCache object with key_cache and value_cache lists

DeltaCache format:
    tuple[Tensor, Tensor]
    key/value shape: [num_layers, seq_len, num_heads, head_dim]
"""

from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor

# Try to import DynamicCache for newer transformers versions
try:
    from transformers.cache_utils import DynamicCache

    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False
    DynamicCache = None


def hf_to_deltacache(
    past_key_values: Union[Tuple[Tuple[Tensor, Tensor], ...], Any],
    remove_batch_dim: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Convert HuggingFace KV cache format to DeltaCache format.

    Args:
        past_key_values: HuggingFace format KV cache.
            tuple of (key, value) per layer, each with shape [batch, heads, seq, dim]
            OR DynamicCache object (transformers >= 4.36)
        remove_batch_dim: If True, squeeze batch dimension (assumes batch_size=1)

    Returns:
        Tuple of (key_cache, value_cache) with shape [num_layers, seq_len, num_heads, head_dim]
    """
    # Handle DynamicCache object (transformers >= 4.36)
    if HAS_DYNAMIC_CACHE and isinstance(past_key_values, DynamicCache):
        # Access keys/values from each layer
        key_list = [layer.keys for layer in past_key_values.layers]
        value_list = [layer.values for layer in past_key_values.layers]
    else:
        # Legacy tuple format
        if not past_key_values:
            raise ValueError("past_key_values cannot be empty")
        key_list = [layer_kv[0] for layer_kv in past_key_values]
        value_list = [layer_kv[1] for layer_kv in past_key_values]

    # Stack keys and values from all layers
    # Each layer has shape [batch, heads, seq, dim]
    # Handle multi-GPU case: move all tensors to same device
    if key_list:
        target_device = key_list[0].device
        key_list = [k.to(target_device) for k in key_list]
        value_list = [v.to(target_device) for v in value_list]

    keys = torch.stack(key_list, dim=0)
    values = torch.stack(value_list, dim=0)
    # Shape: [num_layers, batch, heads, seq, dim]

    if remove_batch_dim:
        # Squeeze batch dimension (assumes batch_size=1)
        keys = keys.squeeze(1)  # [num_layers, heads, seq, dim]
        values = values.squeeze(1)

    # Transpose from [layers, heads, seq, dim] to [layers, seq, heads, dim]
    keys = keys.transpose(1, 2)  # [num_layers, seq, heads, dim]
    values = values.transpose(1, 2)

    return keys, values


def deltacache_to_hf(
    key_cache: Tensor,
    value_cache: Tensor,
    add_batch_dim: bool = True,
    use_dynamic_cache: bool = True,
) -> Union[Tuple[Tuple[Tensor, Tensor], ...], Any]:
    """Convert DeltaCache KV cache format to HuggingFace format.

    Args:
        key_cache: Key cache with shape [num_layers, seq_len, num_heads, head_dim]
        value_cache: Value cache with shape [num_layers, seq_len, num_heads, head_dim]
        add_batch_dim: If True, add batch dimension of size 1
        use_dynamic_cache: If True and available, return DynamicCache object

    Returns:
        HuggingFace format: tuple of (key, value) per layer,
        each with shape [batch, heads, seq, dim]
        OR DynamicCache object if use_dynamic_cache=True and transformers >= 4.36
    """
    # Transpose from [layers, seq, heads, dim] to [layers, heads, seq, dim]
    keys = key_cache.transpose(1, 2)
    values = value_cache.transpose(1, 2)

    if add_batch_dim:
        # Add batch dimension
        keys = keys.unsqueeze(1)  # [layers, 1, heads, seq, dim]
        values = values.unsqueeze(1)

    num_layers = keys.shape[0]

    # Return DynamicCache if available and requested
    if use_dynamic_cache and HAS_DYNAMIC_CACHE:
        cache = DynamicCache()
        for i in range(num_layers):
            cache.update(keys[i], values[i], i)
        return cache

    # Legacy tuple format
    past_key_values = tuple((keys[i], values[i]) for i in range(num_layers))

    return past_key_values


class KVFormatConverter:
    """Utility class for KV cache format conversion with validation."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize converter with expected dimensions.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads (key-value heads for GQA)
            head_dim: Dimension per attention head
            dtype: Expected tensor dtype
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

    def from_hf(
        self,
        past_key_values: Tuple[Tuple[Tensor, Tensor], ...],
        validate: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Convert from HuggingFace format with optional validation.

        Args:
            past_key_values: HuggingFace format KV cache
            validate: Whether to validate dimensions

        Returns:
            DeltaCache format (key_cache, value_cache)
        """
        if validate:
            self._validate_hf_format(past_key_values)

        return hf_to_deltacache(past_key_values)

    def to_hf(
        self,
        key_cache: Tensor,
        value_cache: Tensor,
        validate: bool = True,
    ) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """Convert to HuggingFace format with optional validation.

        Args:
            key_cache: DeltaCache key cache
            value_cache: DeltaCache value cache
            validate: Whether to validate dimensions

        Returns:
            HuggingFace format past_key_values
        """
        if validate:
            self._validate_deltacache_format(key_cache, value_cache)

        return deltacache_to_hf(key_cache, value_cache)

    def _validate_hf_format(
        self,
        past_key_values: Tuple[Tuple[Tensor, Tensor], ...],
    ) -> None:
        """Validate HuggingFace format KV cache."""
        if len(past_key_values) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {len(past_key_values)}")

        for i, (k, v) in enumerate(past_key_values):
            # Expected shape: [batch, heads, seq, dim]
            if k.dim() != 4:
                raise ValueError(f"Layer {i} key has {k.dim()} dims, expected 4")
            if v.dim() != 4:
                raise ValueError(f"Layer {i} value has {v.dim()} dims, expected 4")

            _, heads, _, dim = k.shape
            if heads != self.num_heads:
                raise ValueError(f"Layer {i}: expected {self.num_heads} heads, got {heads}")
            if dim != self.head_dim:
                raise ValueError(f"Layer {i}: expected head_dim {self.head_dim}, got {dim}")

    def _validate_deltacache_format(
        self,
        key_cache: Tensor,
        value_cache: Tensor,
    ) -> None:
        """Validate DeltaCache format KV cache."""
        # Expected shape: [num_layers, seq_len, num_heads, head_dim]
        if key_cache.dim() != 4:
            raise ValueError(f"key_cache has {key_cache.dim()} dims, expected 4")
        if value_cache.dim() != 4:
            raise ValueError(f"value_cache has {value_cache.dim()} dims, expected 4")

        layers, _, heads, dim = key_cache.shape
        if layers != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {layers}")
        if heads != self.num_heads:
            raise ValueError(f"Expected {self.num_heads} heads, got {heads}")
        if dim != self.head_dim:
            raise ValueError(f"Expected head_dim {self.head_dim}, got {dim}")

    def create_empty_hf_cache(
        self,
        seq_len: int,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """Create empty HuggingFace format KV cache.

        Args:
            seq_len: Sequence length
            batch_size: Batch size
            device: Device for tensors

        Returns:
            Empty HuggingFace format KV cache
        """
        shape = (batch_size, self.num_heads, seq_len, self.head_dim)
        return tuple(
            (
                torch.zeros(shape, dtype=self.dtype, device=device),
                torch.zeros(shape, dtype=self.dtype, device=device),
            )
            for _ in range(self.num_layers)
        )

    def create_empty_deltacache(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Create empty DeltaCache format KV cache.

        Args:
            seq_len: Sequence length
            device: Device for tensors

        Returns:
            Empty DeltaCache format (key_cache, value_cache)
        """
        shape = (self.num_layers, seq_len, self.num_heads, self.head_dim)
        return (
            torch.zeros(shape, dtype=self.dtype, device=device),
            torch.zeros(shape, dtype=self.dtype, device=device),
        )


def slice_hf_cache(
    past_key_values: Tuple[Tuple[Tensor, Tensor], ...],
    start: int,
    end: Optional[int] = None,
) -> Tuple[Tuple[Tensor, Tensor], ...]:
    """Slice HuggingFace KV cache along sequence dimension.

    Args:
        past_key_values: HuggingFace format KV cache
        start: Start index
        end: End index (None for rest of sequence)

    Returns:
        Sliced KV cache
    """
    return tuple((k[:, :, start:end, :], v[:, :, start:end, :]) for k, v in past_key_values)


def concat_hf_cache(
    cache1: Tuple[Tuple[Tensor, Tensor], ...],
    cache2: Tuple[Tuple[Tensor, Tensor], ...],
) -> Tuple[Tuple[Tensor, Tensor], ...]:
    """Concatenate two HuggingFace KV caches along sequence dimension.

    Args:
        cache1: First KV cache
        cache2: Second KV cache

    Returns:
        Concatenated KV cache
    """
    if len(cache1) != len(cache2):
        raise ValueError("Caches must have same number of layers")

    return tuple(
        (
            torch.cat([c1[0], c2[0]], dim=2),
            torch.cat([c1[1], c2[1]], dim=2),
        )
        for c1, c2 in zip(cache1, cache2)
    )


def get_hf_cache_seq_len(
    past_key_values: Tuple[Tuple[Tensor, Tensor], ...],
) -> int:
    """Get sequence length from HuggingFace KV cache.

    Args:
        past_key_values: HuggingFace format KV cache

    Returns:
        Sequence length
    """
    if not past_key_values:
        return 0
    # Shape: [batch, heads, seq, dim]
    return past_key_values[0][0].shape[2]
