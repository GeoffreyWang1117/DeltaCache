"""Tier-B adapter: the quantized KV cache that ships inside transformers.

This is upstream code, not a reimplementation, which is what makes it evidence.
It is also the cleanest possible statement of the axis claim: the cache exposes
``nbits``, a small set of integers, and nothing else. A requested compression
ratio of 3x has no representation in it. The adapter records what was asked for,
what the implementation could actually provide, and whether the two differ.

Requires a backend:

    pip install optimum-quanto     # backend="quanto", nbits in {2, 4}
    pip install hqq                # backend="hqq",   nbits in {1, 2, 3, 4, 8}

If the backend is missing the adapter raises. It must never fall back to
something that still returns a number, because a silent fallback and a genuinely
flat footprint are indistinguishable in the output.

Both cache APIs are supported. transformers 5.x exposes per-layer classes taking
their settings directly; 4.x exposes whole-cache classes built from a config
object and an update call that takes a layer index. The measurement is the same
either way, and which one ran is recorded in the cell.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch

from experiments.byte_audit.accountant import (
    allocator_probe,
    float_byte_fraction,
    is_simulated,
    measure_structural,
)
from experiments.byte_audit.gate import Cell, Shape, make_inputs

# What each backend can actually represent. Everything else must snap.
_NATIVE_NBITS: Dict[str, Tuple[int, ...]] = {
    "quanto": (2, 4),
    "hqq": (1, 2, 3, 4, 8),
}


def nbits_for_cr(cr: float, backend: str) -> Tuple[int, bool]:
    """Nearest representable bit width, and whether the request had to snap.

    The ideal width is 16 / cr against an FP16 baseline. A request is honoured
    exactly only when that lands on a width the backend implements.
    """
    available = _NATIVE_NBITS[backend]
    ideal = 16.0 / max(cr, 1e-9)
    nearest = min(available, key=lambda b: abs(b - ideal))
    return nearest, abs(nearest - ideal) > 1e-6


@dataclass
class HFQuantizedCacheAdapter:
    """Measures one upstream quantized-cache backend across a ratio grid."""

    backend: str = "quanto"
    q_group_size: int = 64
    residual_length: int = 128

    name: str = "hf_quantized_cache"
    category: str = "quantization"

    @staticmethod
    def _quanto_available() -> bool:
        """Probe both homes of the check; it moved between transformers releases."""
        from transformers import cache_utils as cu

        probe = getattr(cu, "is_optimum_quanto_available", None)
        if probe is None:
            from transformers.utils import import_utils as iu

            probe = getattr(iu, "is_optimum_quanto_available", None)
        if probe is None:  # neither home: decide by import, never by assumption
            try:
                import optimum.quanto  # noqa: F401
            except ImportError:
                return False
            return True
        return bool(probe())

    def _check_backend(self) -> None:
        if self.backend == "quanto":
            if not self._quanto_available():
                raise RuntimeError(
                    "backend 'quanto' selected but optimum-quanto is not installed; "
                    "run `pip install optimum-quanto` or pass --backend hqq. "
                    "Refusing to report bytes from a cache that never quantized."
                )
        elif self.backend == "hqq":
            try:
                import hqq  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "backend 'hqq' selected but hqq is not installed; run `pip install hqq`."
                ) from exc
        else:
            raise ValueError(f"unknown backend: {self.backend}")

    def _build(self, nbits: int) -> Tuple[object, Callable, str]:
        """Return (cache object, update callable, which API was used).

        The object is what gets measured, so it must be the thing that holds the
        quantized tensors, not a view returned by the update call.
        """
        from transformers import cache_utils as cu

        layer_cls = getattr(cu, f"{'Quanto' if self.backend == 'quanto' else 'HQQ'}QuantizedLayer",
                            None)
        if layer_cls is not None:  # transformers >= 5
            layer = layer_cls(
                nbits=nbits,
                q_group_size=self.q_group_size,
                residual_length=self.residual_length,
            )
            return layer, (lambda k, v: layer.update(k, v)), "layer-api"

        cache_cls = getattr(cu, f"{'Quanto' if self.backend == 'quanto' else 'HQQ'}QuantizedCache",
                            None)
        if cache_cls is None:
            raise RuntimeError(
                f"transformers exposes neither a quantized layer nor cache class for "
                f"backend '{self.backend}'"
            )
        cfg = cu.QuantizedCacheConfig(
            backend=self.backend,
            nbits=nbits,
            q_group_size=self.q_group_size,
            residual_length=self.residual_length,
        )
        cache = cache_cls(cfg)
        return cache, (lambda k, v: cache.update(k, v, 0)), "cache-api"

    def run(
        self,
        shape: Shape,
        grid: Sequence[float],
        device: torch.device,
        seed: int = 0,
    ) -> List[Cell]:
        self._check_backend()
        data = make_inputs(shape, device, seed=seed)
        # This cache takes (batch, heads, seq, head_dim); the gate holds
        # (layers, seq, heads, head_dim).
        keys = data["keys"][0].permute(1, 0, 2).unsqueeze(0).contiguous()
        values = data["values"][0].permute(1, 0, 2).unsqueeze(0).contiguous()
        fullkv = measure_structural([keys, values]).total_bytes

        cells: List[Cell] = []
        for cr in grid:
            nbits, snapped = nbits_for_cr(float(cr), self.backend)
            method_name = f"{self.name}[{self.backend}]"
            cell = Cell(
                method=method_name,
                category=self.category,
                compression_ratio=float(cr),
                ok=False,
                source=f"adapter:{self.name}",
                fullkv_bytes=fullkv,
                seq_len=shape.seq_len,
            )
            try:
                cache_obj, update, which_api = self._build(nbits)
                t0 = time.perf_counter()
                with allocator_probe(device) as probe:
                    # The first call quantizes but returns its input unchanged,
                    # so a single prefill leaves nothing to compare against.
                    # Decoding one further token forces the dequantize path and
                    # makes the returned cache observable.
                    update(keys[..., :-1, :].contiguous(), values[..., :-1, :].contiguous())
                    out_k, _ = update(keys[..., -1:, :].contiguous(),
                                      values[..., -1:, :].contiguous())
                cell.elapsed_ms = (time.perf_counter() - t0) * 1000.0

                fp = measure_structural(cache_obj)
                cell.ok = True
                # Upstream reports no footprint of its own; that absence is the
                # finding, so the field stays None rather than being invented.
                cell.reported_bytes = None
                cell.structural_bytes = fp.total_bytes
                cell.logical_bytes = fp.logical_bytes
                cell.view_overhead = round(fp.view_overhead, 4)
                cell.allocator_peak_bytes = probe["peak_bytes"]
                cell.retained_tokens = int(out_k.shape[-2])
                cell.values_changed = not torch.equal(out_k, keys)
                cell.did_something = bool(cell.values_changed)
                cell.simulated_storage = is_simulated(fp)
                cell.float_byte_fraction = round(float_byte_fraction(fp), 6)
                cell.by_dtype = fp.as_dict()["by_dtype"]
                cell.nominal_bits = nbits
                # Not a failure: this field carries the setting the request had
                # to be mapped onto, which is the axis claim in the raw data.
                cell.error = (
                    f"{which_api} nbits={nbits} residual={self.residual_length}"
                    + (" SNAPPED" if snapped else "")
                )
            except Exception as exc:
                cell.error = f"{type(exc).__name__}: {exc}"
            cells.append(cell)
        return cells


ADAPTERS = [HFQuantizedCacheAdapter(backend="quanto")]
