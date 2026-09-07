"""Tier-B adapter: the group-wise quantizer shipped alongside H2O.

Source: github.com/FMInference/H2O, `h2o_flexgen/flexgen/compression.py`.

This is the control the audit most needs. The H2O repository ships an eviction
path and a quantization path, so a difference measured between the two cannot be
explained away as a difference between projects: same authors, same repository,
same commit.

It also has a property worth stating carefully rather than sensationally. The
repository contains two quantization paths. ``TorchCompressedDevice.allocate``
computes a packed layout, several values to a word. ``compress`` is labelled by
its own docstring as simulating group-wise quantization and returns one tensor
element per value. The two therefore deliver different bytes for the same
nominal bit width, and a paper reporting "4-bit" does not say which ran. This
adapter measures ``compress``, the path reachable without FlexGen's device
machinery, and labels it as such in every cell.
"""

from __future__ import annotations

import dataclasses
import time
from typing import List, Sequence, Tuple

import torch

from experiments.byte_audit.accountant import allocator_probe, measure_structural
from experiments.byte_audit.adapters._common import finish_cell
from experiments.byte_audit.gate import Cell, Shape, make_inputs
from experiments.byte_audit.upstream import MANIFEST, load_from_source

_SOURCE = "h2o_flexgen/flexgen/compression.py"
# compress() asserts this bound; within it any integer width is accepted, which
# is a wider native set than most quantized caches expose.
_NATIVE_BITS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)


def bits_for_cr(cr: float) -> Tuple[int, bool]:
    ideal = 16.0 / max(cr, 1e-9)
    nearest = min(_NATIVE_BITS, key=lambda b: abs(b - ideal))
    return nearest, abs(nearest - ideal) > 1e-6


@dataclasses.dataclass
class H2OFlexGenAdapter:
    group_size: int = 64
    group_dim: int = 2
    symmetric: bool = False
    name: str = "h2o_flexgen_quant"
    category: str = "quantization"

    def run(self, shape: Shape, grid: Sequence[float], device: torch.device) -> List[Cell]:
        upstream = MANIFEST["h2o"]
        upstream.require()
        loaded = load_from_source(
            upstream.path / _SOURCE,
            ("CompressionConfig", "compress"),
            namespace={"dataclasses": dataclasses},
        )
        config_cls, compress = loaded["CompressionConfig"], loaded["compress"]

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
                cfg = config_cls(
                    num_bits=bits,
                    group_size=self.group_size,
                    group_dim=self.group_dim,
                    symmetric=self.symmetric,
                )
                t0 = time.perf_counter()
                with allocator_probe(device) as probe:
                    packed_k = compress(keys, cfg)
                    packed_v = compress(values, cfg)
                elapsed = (time.perf_counter() - t0) * 1000.0

                finish_cell(
                    cell,
                    [packed_k, packed_v],
                    probe,
                    retained_tokens=shape.seq_len,
                    seq_len=shape.seq_len,
                    elapsed_ms=elapsed,
                    changed=True,
                    nominal_bits=bits,
                    note=(
                        f"path=compress(simulated) bits={bits} group={self.group_size} "
                        f"commit={upstream.commit()}" + (" SNAPPED" if snapped else "")
                    ),
                )
            except Exception as exc:
                cell.error = f"{type(exc).__name__}: {exc}"
            cells.append(cell)
        return cells


ADAPTERS = [H2OFlexGenAdapter()]
