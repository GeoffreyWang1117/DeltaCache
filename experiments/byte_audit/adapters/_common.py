"""Shared recording for tier-B adapters.

Every adapter fills the same measurement fields, and an adapter that forgets one
produces a cell that looks measured but is not. Filling them in one place keeps
the record uniform and makes the positive control impossible to skip.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from experiments.byte_audit.accountant import (
    Footprint,
    float_byte_fraction,
    is_simulated,
    measure_structural,
)
from experiments.byte_audit.gate import Cell


def finish_cell(
    cell: Cell,
    measured: Any,
    probe: Dict[str, int],
    retained_tokens: int,
    seq_len: int,
    elapsed_ms: float,
    note: str,
    changed: Optional[bool] = None,
    reported_bytes: Optional[int] = None,
    nominal_bits: Optional[int] = None,
) -> Footprint:
    """Record one measured cell and return the footprint for further inspection.

    Args:
        measured: the object holding the cache. Measuring what a call *returns*
            is usually wrong; what the implementation keeps is what a server pays.
        retained_tokens: how many token positions survived.
        changed: whether the values themselves differ from the input. Defaults to
            "fewer tokens survived", which is the right test for eviction but not
            for quantization, where every position survives with altered values.
        reported_bytes: only when upstream states a footprint of its own. Leaving
            it None is a finding, not a gap in the record.
    """
    fp = measure_structural(measured)
    acted = retained_tokens < seq_len if changed is None else bool(changed)

    cell.ok = True
    cell.elapsed_ms = elapsed_ms
    cell.reported_bytes = reported_bytes
    cell.structural_bytes = fp.total_bytes
    cell.logical_bytes = fp.logical_bytes
    cell.view_overhead = round(fp.view_overhead, 4)
    cell.allocator_peak_bytes = probe["peak_bytes"]
    cell.retained_tokens = retained_tokens
    cell.values_changed = acted
    cell.did_something = acted
    cell.simulated_storage = is_simulated(fp)
    cell.float_byte_fraction = round(float_byte_fraction(fp), 6)
    cell.by_dtype = fp.as_dict()["by_dtype"]
    cell.nominal_bits = nominal_bits
    cell.error = note
    return fp
