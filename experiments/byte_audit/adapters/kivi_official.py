"""Tier-B adapter: official KIVI quantization.

Source: github.com/jy-yuan/KIVI, `quant/new_pack.py`.

KIVI is the reason this audit exists. Our own checkpoints showed its footprint
byte-identical at every requested compression ratio, which is what first
suggested that a ratio is not a knob a quantization method has. This measures
the real thing rather than our reimplementation of it.

Its knob is `bits`, restricted to 2, 4 or 8 by the packing routine, plus a group
size. Keys are quantized along the token axis and values along the feature axis,
which is KIVI's asymmetry and the reason both are measured here rather than one.

The module imports triton at the top, but the quantize-and-pack functions
themselves are plain torch, so extracting them runs the real packing on CPU.
Packing is genuine: values go into int32 words, 32 // bits of them per word.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from experiments.byte_audit.accountant import allocator_probe, measure_structural
from experiments.byte_audit.adapters._common import finish_cell
from experiments.byte_audit.gate import Cell, Shape, make_inputs
from experiments.byte_audit.upstream import MANIFEST, load_from_source

_NATIVE_BITS: Tuple[int, ...] = (2, 4, 8)
_SOURCE = "quant/new_pack.py"
_NAMES = ("pack_tensor", "quant_and_pack_kcache", "quant_and_pack_vcache")


def bits_for_cr(cr: float) -> Tuple[int, bool]:
    """Nearest bit width KIVI can pack, and whether the request had to snap."""
    ideal = 16.0 / max(cr, 1e-9)
    nearest = min(_NATIVE_BITS, key=lambda b: abs(b - ideal))
    return nearest, abs(nearest - ideal) > 1e-6


@dataclass
class KIVIOfficialAdapter:
    group_size: int = 32
    name: str = "kivi_official"
    category: str = "quantization"

    def run(self, shape: Shape, grid: Sequence[float], device: torch.device) -> List[Cell]:
        upstream = MANIFEST["kivi"]
        upstream.require()
        fns = load_from_source(upstream.path / _SOURCE, _NAMES)
        quant_k = fns["quant_and_pack_kcache"]
        quant_v = fns["quant_and_pack_vcache"]

        data = make_inputs(shape, device)
        keys = data["keys"][0].permute(1, 0, 2).unsqueeze(0).contiguous()
        values = data["values"][0].permute(1, 0, 2).unsqueeze(0).contiguous()
        fullkv = measure_structural([keys, values]).total_bytes

        cells: List[Cell] = []
        for cr in grid:
            bits, snapped = bits_for_cr(float(cr))
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
                t0 = time.perf_counter()
                with allocator_probe(device) as probe:
                    # Keys group along tokens, values along features: KIVI's
                    # asymmetry, kept rather than normalised away.
                    k_packed = quant_k(keys, self.group_size, bits)
                    v_packed = quant_v(values, self.group_size, bits)
                elapsed = (time.perf_counter() - t0) * 1000.0

                finish_cell(
                    cell,
                    [k_packed, v_packed],
                    probe,
                    retained_tokens=shape.seq_len,
                    seq_len=shape.seq_len,
                    elapsed_ms=elapsed,
                    # Every position survives; what changed is precision, so the
                    # token count cannot serve as the positive control here.
                    changed=True,
                    nominal_bits=bits,
                    note=(
                        f"bits={bits} group={self.group_size} commit={upstream.commit()}"
                        + (" SNAPPED" if snapped else "")
                    ),
                )
            except Exception as exc:
                cell.error = f"{type(exc).__name__}: {exc}"
            cells.append(cell)
        return cells


ADAPTERS = [KIVIOfficialAdapter()]
