"""Tier-B adapter: official SnapKV eviction.

Source: github.com/FasterDecoding/SnapKV, `snapkv/monkeypatch/snapkv_utils.py`.

SnapKV's knob is `max_capacity_prompt`, an absolute token count, so unlike a bit
width it can land on any requested ratio down to single-token resolution. It
scores positions by the attention the final `window_size` queries paid them,
pools the scores, and keeps the top ones plus that recent window.

Below the budget it returns its input untouched, which is the exact-zero region
every eviction policy has and which the positive control correctly rejects.

Upstream ships this as a monkeypatch over a specific transformers release, so
the module cannot be imported today. The cluster class itself is plain torch and
is extracted and run unmodified.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.byte_audit.accountant import allocator_probe, measure_structural
from experiments.byte_audit.adapters._common import finish_cell
from experiments.byte_audit.gate import Cell, Shape, make_inputs
from experiments.byte_audit.upstream import MANIFEST, load_from_source

_SOURCE = "snapkv/monkeypatch/snapkv_utils.py"


def capacity_for_cr(seq_len: int, cr: float, window: int) -> int:
    """Requested ratio as a token budget, kept above the recent window.

    The class asserts that the budget exceeds the window, so a ratio aggressive
    enough to fall below it is not representable and is clamped here rather than
    left to raise inside upstream code.
    """
    want = round(seq_len / max(cr, 1e-9))
    return max(window + 1, min(seq_len, want))


@dataclass
class SnapKVOfficialAdapter:
    window_size: int = 32
    kernel_size: int = 5
    pooling: str = "avgpool"
    name: str = "snapkv_official"
    category: str = "eviction"

    def run(self, shape: Shape, grid: Sequence[float], device: torch.device) -> List[Cell]:
        if shape.is_gqa:
            raise NotImplementedError(
                f"{self.name} scores keys against queries, and this adapter synthesises "
                "queries at the key-head count. Under grouped-query attention that would "
                "measure a smaller multi-head cache rather than a grouped one. Refusing "
                "instead of reporting a number for the wrong geometry."
            )
        upstream = MANIFEST["snapkv"]
        upstream.require()
        loaded = load_from_source(
            upstream.path / _SOURCE,
            ("repeat_kv", "SnapKVCluster"),
            namespace={"math": math, "F": F, "nn": nn},
        )
        cluster_cls = loaded["SnapKVCluster"]

        data = make_inputs(shape, device)
        keys = data["keys"][0].permute(1, 0, 2).unsqueeze(0).contiguous()
        values = data["values"][0].permute(1, 0, 2).unsqueeze(0).contiguous()
        # SnapKV scores keys against the queries, so queries are part of the
        # input rather than something derivable from the cache.
        queries = torch.randn_like(keys)
        fullkv = measure_structural([keys, values]).total_bytes

        cells: List[Cell] = []
        for cr in grid:
            capacity = capacity_for_cr(shape.seq_len, float(cr), self.window_size)
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
                cluster = cluster_cls(
                    window_size=self.window_size,
                    max_capacity_prompt=capacity,
                    kernel_size=self.kernel_size,
                    pooling=self.pooling,
                )
                t0 = time.perf_counter()
                with allocator_probe(device) as probe:
                    out_k, out_v = cluster.update_kv(keys, queries, values, None, 1)
                elapsed = (time.perf_counter() - t0) * 1000.0

                finish_cell(
                    cell,
                    [out_k, out_v],
                    probe,
                    retained_tokens=int(out_k.shape[2]),
                    seq_len=shape.seq_len,
                    elapsed_ms=elapsed,
                    note=(
                        f"capacity={capacity} window={self.window_size} "
                        f"commit={upstream.commit()}"
                    ),
                )
            except Exception as exc:
                cell.error = f"{type(exc).__name__}: {exc}"
            cells.append(cell)
        return cells


ADAPTERS = [SnapKVOfficialAdapter()]
