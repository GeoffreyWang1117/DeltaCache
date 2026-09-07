"""Rotary Position Embedding (RoPE) handler for incremental KV cache."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


class RoPEHandler:
    """
    Handler for Rotary Position Embeddings in incremental KV cache.

    RoPE applies position-dependent rotations to keys and queries.
    For cache reuse, we need to handle the case where cached KV
    may be used at different absolute positions.

    Strategy:
    - Store KV with position embeddings already applied (standard approach)
    - For prefix reuse, positions are consistent, so no adjustment needed
    - Provide utilities for applying RoPE to new tokens at correct positions

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        head_dim: int,
        max_position: int = 8192,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize RoPE handler.

        Args:
            head_dim: Dimension per attention head.
            max_position: Maximum sequence length to support.
            base: Base for computing rotation frequencies.
            device: Device for precomputed tensors.
            dtype: Data type for computations.
        """
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Precompute inverse frequencies
        self._inv_freq = self._compute_inv_freq()

        # Cache for sin/cos values
        self._cos_cached: Optional[Tensor] = None
        self._sin_cached: Optional[Tensor] = None
        self._seq_len_cached: int = 0

    def _compute_inv_freq(self) -> Tensor:
        """Compute inverse frequencies for RoPE."""
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        return inv_freq.to(self.device)

    def _update_cos_sin_cache(self, seq_len: int) -> None:
        """Update cached cos/sin values if needed."""
        if seq_len <= self._seq_len_cached and self._cos_cached is not None:
            return

        self._seq_len_cached = max(seq_len, self.max_position)

        # Compute position indices
        t = torch.arange(self._seq_len_cached, device=self.device, dtype=torch.float32)

        # Compute frequencies: [seq_len, head_dim/2]
        freqs = torch.outer(t, self._inv_freq)

        # Compute cos and sin: [seq_len, head_dim]
        # Repeat to match head_dim
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos().to(self.dtype)
        self._sin_cached = emb.sin().to(self.dtype)

    def get_cos_sin(
        self,
        positions: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get cos and sin values for given positions.

        Args:
            positions: Position indices [batch_size, seq_len] or [seq_len].

        Returns:
            Tuple of (cos, sin) tensors.
        """
        max_pos = positions.max().item() + 1
        self._update_cos_sin_cache(int(max_pos))

        cos = self._cos_cached[positions]
        sin = self._sin_cached[positions]

        return cos, sin

    def apply_rotary_pos_emb(
        self,
        q: Tensor,
        k: Tensor,
        positions: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary position embeddings to queries and keys.

        Args:
            q: Query tensor [..., seq_len, num_heads, head_dim].
            k: Key tensor [..., seq_len, num_heads, head_dim].
            positions: Position indices [seq_len] or [batch, seq_len].

        Returns:
            Tuple of (rotated_q, rotated_k).
        """
        cos, sin = self.get_cos_sin(positions)

        # Expand dims for broadcasting with [..., seq_len, num_heads, head_dim]
        # cos/sin are [seq_len, head_dim], need [..., seq_len, 1, head_dim]
        if cos.dim() == 2:  # [seq_len, head_dim]
            cos = cos.unsqueeze(-2)  # [seq_len, 1, head_dim]
            sin = sin.unsqueeze(-2)

        # Add batch dimensions if needed
        while cos.dim() < q.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        q_rotated = self._apply_rope(q, cos, sin)
        k_rotated = self._apply_rope(k, cos, sin)

        return q_rotated, k_rotated

    def apply_rotary_pos_emb_to_k(
        self,
        k: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """
        Apply rotary position embeddings to keys only.

        Args:
            k: Key tensor [..., seq_len, num_heads, head_dim].
            positions: Position indices.

        Returns:
            Rotated key tensor.
        """
        cos, sin = self.get_cos_sin(positions)

        # Expand dims for broadcasting with [..., seq_len, num_heads, head_dim]
        if cos.dim() == 2:  # [seq_len, head_dim]
            cos = cos.unsqueeze(-2)  # [seq_len, 1, head_dim]
            sin = sin.unsqueeze(-2)

        while cos.dim() < k.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        return self._apply_rope(k, cos, sin)

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Apply rotation to input tensor.

        Uses the formula:
        x_rotated = x * cos + rotate_half(x) * sin
        """
        x_rotated = self._rotate_half(x)
        return x * cos + x_rotated * sin

    def _rotate_half(self, x: Tensor) -> Tensor:
        """
        Rotate half of the hidden dims.

        Splits tensor into two halves and rotates:
        [x1, x2] -> [-x2, x1]
        """
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def remove_rotary_pos_emb(
        self,
        k: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """
        Remove rotary position embeddings from keys.

        This is useful for storing position-normalized KV cache
        that can be reused at different positions.

        Note: This is the inverse operation of apply_rotary_pos_emb.

        Args:
            k: Key tensor with position embeddings.
            positions: Original position indices.

        Returns:
            Position-normalized key tensor.
        """
        cos, sin = self.get_cos_sin(positions)

        # Expand dims for broadcasting with [..., seq_len, num_heads, head_dim]
        if cos.dim() == 2:  # [seq_len, head_dim]
            cos = cos.unsqueeze(-2)  # [seq_len, 1, head_dim]
            sin = sin.unsqueeze(-2)

        while cos.dim() < k.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        # Inverse rotation: use -sin instead of sin
        return self._apply_rope(k, cos, -sin)

    def reposition_keys(
        self,
        k: Tensor,
        old_positions: Tensor,
        new_positions: Tensor,
    ) -> Tensor:
        """
        Adjust keys from old positions to new positions.

        This allows reusing cached keys at different absolute positions.

        Args:
            k: Key tensor with old position embeddings.
            old_positions: Original position indices.
            new_positions: Target position indices.

        Returns:
            Key tensor with new position embeddings.
        """
        # Remove old positions
        k_normalized = self.remove_rotary_pos_emb(k, old_positions)

        # Apply new positions
        return self.apply_rotary_pos_emb_to_k(k_normalized, new_positions)


def create_position_ids(
    seq_len: int,
    start_pos: int = 0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Create position IDs for a sequence.

    Args:
        seq_len: Sequence length.
        start_pos: Starting position.
        device: Target device.

    Returns:
        Position IDs tensor [seq_len].
    """
    return torch.arange(
        start_pos,
        start_pos + seq_len,
        dtype=torch.long,
        device=device,
    )


def create_batch_position_ids(
    batch_size: int,
    seq_len: int,
    start_positions: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Create position IDs for a batch.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        start_positions: Starting position for each batch item [batch_size].
        device: Target device.

    Returns:
        Position IDs tensor [batch_size, seq_len].
    """
    if start_positions is None:
        start_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

    positions = torch.arange(seq_len, dtype=torch.long, device=device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    positions = positions + start_positions.unsqueeze(1)

    return positions
