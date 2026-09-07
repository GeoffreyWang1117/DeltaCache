"""Template for pointing the audit at a cloned upstream repository.

Copy this file, fill in the two marked places, then:

    python -m experiments.byte_audit.gate --adapters experiments/byte_audit/adapters/mine.py
    python -m experiments.byte_audit.analyze --tier b

Rules that keep the result trustworthy:

1. Never wrap a failure in a default. If the upstream call cannot be made, let
   the exception reach ``Cell.error``. A cell that reports bytes it did not
   measure is worse than a missing cell.
2. Leave ``reported_bytes`` as None unless the repository itself states a
   footprint. Do not recompute their formula for them; the gap between what a
   project claims and what it allocates is the thing being measured.
3. Record the native setting you had to choose in ``error`` when the requested
   ratio is not representable. That snapping is evidence, not noise.
4. Measure in steady state, after at least one decode step. A cache can retain a
   view into the prefill buffer, so a measurement taken at prefill reports the
   same transient peak for every setting and hides the knob entirely.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

# float_byte_fraction and is_simulated are unused until step (2) below is
# filled in; they are imported here so the copy you edit already has them.
from experiments.byte_audit.accountant import (  # noqa: F401
    float_byte_fraction,
    is_simulated,
    measure_structural,
)
from experiments.byte_audit.gate import Cell, Shape, make_inputs


class UpstreamAdapter:
    name = "CHANGE_ME"
    category = "eviction"  # or "quantization" or "joint"

    def run(self, shape: Shape, grid: Sequence[float], device: torch.device) -> List[Cell]:
        data = make_inputs(shape, device)
        fullkv = measure_structural([data["keys"], data["values"]]).total_bytes
        cells: List[Cell] = []
        for cr in grid:
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
                # (1) call the upstream entry point here, producing a cache object
                cache_obj = ...  # noqa: F841
                raise NotImplementedError("fill in the upstream call")
                # (2) then, with cache_obj in hand:
                # fp = measure_structural(cache_obj)
                # cell.ok = True
                # cell.structural_bytes = fp.total_bytes
                # cell.simulated_storage = is_simulated(fp)
                # cell.float_byte_fraction = round(float_byte_fraction(fp), 6)
                # cell.by_dtype = fp.as_dict()["by_dtype"]
                # cell.did_something = ...
            except Exception as exc:
                cell.error = f"{type(exc).__name__}: {exc}"
            cells.append(cell)
        return cells


ADAPTERS = [UpstreamAdapter()]
