"""Tier-B adapter: the sliding-window KV cache inside transformers.

This matters because it is an eviction method living in the same library as the
quantized cache measured by the sibling adapter. Comparing the two removes the
objection that any difference between the families is really a difference
between projects: same codebase, same release, same measurement.

What is measured is the state the layer *keeps*, not what it returns. The update
call returns the full sequence because the current forward pass needs it; the
cache retains only the window, and the window is what a server pays for between
decode steps.

Measured in steady state, after one decode step. Immediately after prefill the
retained window is a *view* into the full prefill buffer, so the whole sequence
stays resident no matter how small the window is. That resolves on the next
update, when the concatenation copies. Measuring at prefill would therefore
report one transient peak for every window size and make the knob look flat.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence

import torch

from experiments.byte_audit.accountant import (
    allocator_probe,
    float_byte_fraction,
    is_simulated,
    measure_structural,
)
from experiments.byte_audit.gate import Cell, Shape, make_inputs


def window_for_cr(seq_len: int, cr: float) -> int:
    """Requested ratio as a window length in tokens."""
    return max(2, min(seq_len, round(seq_len / max(cr, 1e-9))))


@dataclass
class SlidingWindowAdapter:
    name: str = "transformers_sliding_window"
    category: str = "eviction"

    def run(self, shape: Shape, grid: Sequence[float], device: torch.device) -> List[Cell]:
        from transformers import cache_utils as cu

        layer_cls = getattr(cu, "DynamicSlidingWindowLayer", None)
        if layer_cls is None:
            raise RuntimeError(
                "this transformers release exposes no DynamicSlidingWindowLayer; "
                "refusing to substitute a different class"
            )

        data = make_inputs(shape, device)
        keys = data["keys"][0].permute(1, 0, 2).unsqueeze(0).contiguous()
        values = data["values"][0].permute(1, 0, 2).unsqueeze(0).contiguous()
        fullkv = measure_structural([keys, values]).total_bytes

        cells: List[Cell] = []
        for cr in grid:
            window = window_for_cr(shape.seq_len, float(cr))
            cell = Cell(
                method=self.name,
                category=self.category,
                compression_ratio=float(cr),
                ok=False,
                source=f"adapter:{self.name}",
                fullkv_bytes=fullkv,
                seq_len=shape.seq_len,
            )
            try:
                layer = layer_cls(sliding_window=window)
                t0 = time.perf_counter()
                with allocator_probe(device) as probe:
                    layer.update(keys, values)
                    prefill_bytes = measure_structural(layer).total_bytes
                    step = torch.randn(
                        (1, shape.num_heads, 1, shape.head_dim),
                        dtype=keys.dtype,
                        device=device,
                    )
                    layer.update(step, torch.randn_like(step))
                cell.elapsed_ms = (time.perf_counter() - t0) * 1000.0

                fp = measure_structural(layer)
                held = int(layer.keys.shape[-2]) if layer.keys is not None else 0
                cell.ok = True
                cell.reported_bytes = None
                cell.structural_bytes = fp.total_bytes
                cell.logical_bytes = fp.logical_bytes
                cell.view_overhead = round(fp.view_overhead, 4)
                cell.allocator_peak_bytes = probe["peak_bytes"]
                cell.retained_tokens = held
                cell.values_changed = held < shape.seq_len
                cell.did_something = held < shape.seq_len
                cell.simulated_storage = is_simulated(fp)
                cell.float_byte_fraction = round(float_byte_fraction(fp), 6)
                cell.by_dtype = fp.as_dict()["by_dtype"]
                # A window of n keeps n-1: one slot is reserved for the token
                # arriving next. Recorded rather than rounded away.
                cell.error = (
                    f"window={window} held={held} "
                    f"prefill_transient={prefill_bytes}B steady={fp.total_bytes}B"
                )
            except Exception as exc:
                cell.error = f"{type(exc).__name__}: {exc}"
            cells.append(cell)
        return cells


ADAPTERS = [SlidingWindowAdapter()]
