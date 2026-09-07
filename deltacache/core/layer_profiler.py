"""Online per-layer attention sparsity profiler.

Extracts attention statistics during prefill via forward hooks, without
requiring full O(S²) attention matrices. Uses the last-token attention
row (the row that matters for next-token generation) to compute:

  - Gini coefficient: measures attention sparsity (0 = uniform, 1 = concentrated)
  - Cumulative attention mass: fraction captured by top-k tokens
  - Attention entropy: Shannon entropy of the distribution

These signals drive LayerBudget's per-layer token retention decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import Tensor


@dataclass
class LayerProfile:
    """Attention profile for a single layer."""

    layer_idx: int
    gini: float  # Gini coefficient of last-token attention (0=uniform, 1=sparse)
    entropy: float  # Normalized Shannon entropy (0=concentrated, 1=uniform)
    top10_mass: float  # Fraction of attention in top 10% positions
    top20_mass: float  # Fraction of attention in top 20% positions
    max_attention: float  # Max attention weight (averaged over heads)


@dataclass
class ProfileResult:
    """Complete profiling result across all layers."""

    layer_profiles: List[LayerProfile]
    num_layers: int
    seq_len: int
    profiling_time_ms: float

    def gini_scores(self) -> Dict[int, float]:
        """Return {layer_idx: gini} mapping."""
        return {p.layer_idx: p.gini for p in self.layer_profiles}

    def entropy_scores(self) -> Dict[int, float]:
        """Return {layer_idx: entropy} mapping."""
        return {p.layer_idx: p.entropy for p in self.layer_profiles}

    def sparsity_ranking(self) -> List[int]:
        """Return layer indices sorted by sparsity (most sparse first)."""
        return sorted(
            range(self.num_layers),
            key=lambda layer_i: self.layer_profiles[layer_i].gini,
            reverse=True,
        )


def compute_gini(weights: Tensor) -> float:
    """Compute Gini coefficient of a 1-D probability distribution.

    Args:
        weights: 1-D tensor of attention weights (should sum to ~1).

    Returns:
        Gini coefficient in [0, 1]. Higher = more concentrated/sparse.
    """
    n = weights.numel()
    if n <= 1:
        return 0.0

    sorted_w, _ = weights.sort()
    cumsum = sorted_w.cumsum(0)
    # Gini = 1 - 2 * area under Lorenz curve
    # Area = sum of cumsum / (n * total)
    total = sorted_w.sum()
    if total < 1e-10:
        return 0.0

    area = cumsum.sum() / (n * total)
    gini = 1.0 - 2.0 * area + 1.0 / n
    return max(0.0, min(1.0, gini.item()))


def compute_entropy(weights: Tensor) -> float:
    """Compute normalized Shannon entropy of a distribution.

    Returns value in [0, 1] where 0 = delta, 1 = uniform.
    """
    n = weights.numel()
    if n <= 1:
        return 0.0

    w = weights.clamp(min=1e-10)
    entropy = -(w * w.log()).sum().item()
    max_entropy = math.log(n)
    if max_entropy < 1e-10:
        return 0.0
    return min(1.0, entropy / max_entropy)


class LayerAttentionProfiler:
    """Online attention profiler using forward hooks.

    Attaches hooks to attention layers to capture the last-token
    attention row during prefill. Computes per-layer sparsity metrics
    without materializing full S×S attention matrices.

    Usage::

        profiler = LayerAttentionProfiler()
        result = profiler.profile(input_ids, model, device="cuda:0")
        gini_scores = result.gini_scores()
    """

    def __init__(self) -> None:
        self._hooks: List[Any] = []
        self._attention_rows: Dict[int, Tensor] = {}

    def _make_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook that captures last-token attention."""

        def hook_fn(module: Any, args: Any, output: Any) -> None:
            # Transformer attention modules return (attn_output, attn_weights, ...)
            # attn_weights shape: (batch, num_heads, seq_len, seq_len)
            attn_weights = None

            if isinstance(output, tuple) and len(output) >= 2:
                candidate = output[1]
                if candidate is not None and isinstance(candidate, Tensor) and candidate.dim() == 4:
                    attn_weights = candidate

            if attn_weights is not None:
                # Extract last-token row, average over heads → (seq_len,)
                # This is O(S) memory, not O(S²)
                last_row = attn_weights[0, :, -1, :].mean(dim=0).detach().cpu()
                self._attention_rows[layer_idx] = last_row

        return hook_fn

    def _find_attention_modules(self, model: Any) -> List[Tuple[int, Any]]:
        """Find attention modules in common transformer architectures."""
        attention_modules = []

        # Try common attribute paths
        layers = None
        for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                layers = obj
                break
            except AttributeError:
                continue

        if layers is None:
            return []

        for idx, layer in enumerate(layers):
            # Find the self-attention module within each layer
            attn = None
            for attr in ("self_attn", "attn", "attention"):
                if hasattr(layer, attr):
                    attn = getattr(layer, attr)
                    break
            if attn is not None:
                attention_modules.append((idx, attn))

        return attention_modules

    def register_hooks(self, model: Any) -> int:
        """Attach forward hooks to all attention layers.

        Args:
            model: HuggingFace model (LlamaForCausalLM, etc.)

        Returns:
            Number of hooks registered.
        """
        self.remove_hooks()
        self._attention_rows.clear()

        attn_modules = self._find_attention_modules(model)

        for layer_idx, module in attn_modules:
            hook = module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

        return len(self._hooks)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def profile(
        self,
        input_ids: Tensor,
        model: Any,
        device: str = "cuda:0",
    ) -> ProfileResult:
        """Run prefill and compute per-layer attention profiles.

        Args:
            input_ids: Token IDs, shape (seq_len,) or (1, seq_len).
            model: HuggingFace model with attention layers.
            device: Device for inference.

        Returns:
            ProfileResult with per-layer sparsity metrics.
        """
        import time

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)

        seq_len = input_ids.shape[1]

        # Register hooks
        num_hooks = self.register_hooks(model)
        if num_hooks == 0:
            raise RuntimeError(
                "No attention modules found. Ensure model has self_attn/attn/attention sub-modules."
            )

        # Run prefill with attention output enabled
        t0 = time.perf_counter()
        with torch.no_grad():
            model(
                input_ids=input_ids,
                output_attentions=True,
                return_dict=True,
            )
        profiling_time_ms = (time.perf_counter() - t0) * 1000

        # Compute per-layer profiles from captured attention rows
        layer_profiles = []
        num_layers = len(self._attention_rows)

        for layer_idx in sorted(self._attention_rows.keys()):
            attn_row = self._attention_rows[layer_idx]  # (seq_len,)

            # Normalize (should already sum to ~1, but ensure)
            attn_row = attn_row / (attn_row.sum() + 1e-10)

            gini = compute_gini(attn_row)
            entropy = compute_entropy(attn_row)

            # Top-k mass
            k10 = max(1, seq_len // 10)
            k20 = max(1, seq_len // 5)
            top10_vals, _ = attn_row.topk(min(k10, seq_len))
            top20_vals, _ = attn_row.topk(min(k20, seq_len))
            top10_mass = top10_vals.sum().item()
            top20_mass = top20_vals.sum().item()

            # Max attention (already head-averaged in hook)
            max_attn = attn_row.max().item()

            layer_profiles.append(
                LayerProfile(
                    layer_idx=layer_idx,
                    gini=gini,
                    entropy=entropy,
                    top10_mass=top10_mass,
                    top20_mass=top20_mass,
                    max_attention=max_attn,
                )
            )

        # Cleanup
        self.remove_hooks()
        self._attention_rows.clear()

        return ProfileResult(
            layer_profiles=layer_profiles,
            num_layers=num_layers,
            seq_len=seq_len,
            profiling_time_ms=profiling_time_ms,
        )

    def profile_from_attention_weights(
        self,
        attention_weights: List[Tensor],
    ) -> ProfileResult:
        """Compute profiles from pre-extracted attention weights.

        Useful when attention weights are already available (e.g.,
        from output_attentions=True).

        Args:
            attention_weights: List of per-layer attention tensors,
                each shape (batch, heads, seq_len, seq_len).

        Returns:
            ProfileResult.
        """
        import time

        t0 = time.perf_counter()
        layer_profiles = []

        for layer_idx, attn in enumerate(attention_weights):
            # Extract last-token row, average over heads
            last_row = attn[0, :, -1, :].mean(dim=0).detach().cpu()
            last_row = last_row / (last_row.sum() + 1e-10)

            seq_len = last_row.numel()
            gini = compute_gini(last_row)
            entropy = compute_entropy(last_row)

            k10 = max(1, seq_len // 10)
            k20 = max(1, seq_len // 5)
            top10_vals, _ = last_row.topk(min(k10, seq_len))
            top20_vals, _ = last_row.topk(min(k20, seq_len))

            layer_profiles.append(
                LayerProfile(
                    layer_idx=layer_idx,
                    gini=gini,
                    entropy=entropy,
                    top10_mass=top10_vals.sum().item(),
                    top20_mass=top20_vals.sum().item(),
                    max_attention=last_row.max().item(),
                )
            )

        profiling_time_ms = (time.perf_counter() - t0) * 1000
        seq_len = attention_weights[0].shape[-1] if attention_weights else 0

        return ProfileResult(
            layer_profiles=layer_profiles,
            num_layers=len(attention_weights),
            seq_len=seq_len,
            profiling_time_ms=profiling_time_ms,
        )
