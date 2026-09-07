"""Tier-B adapter: the official H2O implementation, vendored under baselines/.

Source: github.com/FMInference/H2O at ac75c2a, `h2o_hf/utils_real_drop/modify_llama.py`.
The "real drop" variant is the one worth measuring: it removes entries rather
than masking them, so the cache it returns is the cache a server would hold.

The surrounding module imports names from `transformers.models.llama` that moved
after the release H2O was written against, so importing it fails on a current
transformers. Rather than patch upstream code, which would make the measurement
about our patch, this loads the class definition verbatim out of the file and
executes that node alone. What runs is their source, unedited.

H2O's knob is a pair of absolute token counts, `hh_size` and `recent_size`, whose
sum is the budget. A requested ratio has to be turned into that budget here, and
that conversion is exactly what the audit is about: the knob is tokens, not a
ratio, and it is continuous in a way a bit width is not.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from experiments.byte_audit.accountant import (
    allocator_probe,
    float_byte_fraction,
    is_simulated,
    measure_structural,
)
from experiments.byte_audit.gate import Cell, Shape, make_inputs

_UPSTREAM = (
    Path(__file__).resolve().parents[3]
    / "baselines/h2o_official/h2o_hf/utils_real_drop/modify_llama.py"
)
_CLASS_NAME = "H2OKVCache_LayerWise"


def load_upstream_class(path: Path = _UPSTREAM, name: str = _CLASS_NAME) -> type:
    """Execute one class definition out of the upstream file, unmodified."""
    if not path.exists():
        raise FileNotFoundError(
            f"upstream H2O source not found at {path}; the vendored checkout under "
            "baselines/h2o_official is required for this adapter"
        )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name),
        None,
    )
    if node is None:
        raise AttributeError(f"{name} not found in {path}")
    module = ast.Module(body=[node], type_ignores=[])
    namespace: Dict[str, Any] = {"torch": torch}
    # Executing upstream source is the point: a rewrite would measure the rewrite.
    exec(compile(module, filename=str(path), mode="exec"), namespace)
    return namespace[name]


def budget_for_cr(seq_len: int, cr: float) -> Tuple[int, int]:
    """Split a requested ratio into H2O's (heavy hitter, recent) token counts.

    The budget is a token count, so unlike a bit width it can land on any
    requested ratio up to the resolution of one token. Half goes to the recent
    window and half to heavy hitters, which is the split the paper uses.
    """
    keep = max(2, min(seq_len, round(seq_len / max(cr, 1e-9))))
    recent = max(1, keep // 2)
    heavy = max(1, keep - recent)
    return heavy, recent


@dataclass
class H2OOfficialAdapter:
    name: str = "h2o_official"
    category: str = "eviction"

    def run(self, shape: Shape, grid: Sequence[float], device: torch.device) -> List[Cell]:
        if shape.is_gqa:
            raise NotImplementedError(
                f"{self.name} scores keys against queries, and this adapter synthesises "
                "queries at the key-head count. Under grouped-query attention that would "
                "measure a smaller multi-head cache rather than a grouped one. Refusing "
                "instead of reporting a number for the wrong geometry."
            )
        cache_cls = load_upstream_class()
        data = make_inputs(shape, device)
        # H2O works one layer at a time on (batch, heads, seq, head_dim).
        keys = data["keys"][0].permute(1, 0, 2).unsqueeze(0).contiguous()
        values = data["values"][0].permute(1, 0, 2).unsqueeze(0).contiguous()
        attn = data["attention_weights"][0]
        fullkv = measure_structural([keys, values]).total_bytes

        cells: List[Cell] = []
        for cr in grid:
            heavy, recent = budget_for_cr(shape.seq_len, float(cr))
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
                cache = cache_cls(hh_size=heavy, recent_size=recent)
                t0 = time.perf_counter()
                with allocator_probe(device) as probe:
                    out = cache((keys, values), attn)
                cell.elapsed_ms = (time.perf_counter() - t0) * 1000.0

                fp = measure_structural(list(out))
                retained = int(out[0].shape[2])
                cell.ok = True
                # Upstream states no footprint of its own; that absence is data.
                cell.reported_bytes = None
                cell.structural_bytes = fp.total_bytes
                cell.logical_bytes = fp.logical_bytes
                cell.view_overhead = round(fp.view_overhead, 4)
                cell.allocator_peak_bytes = probe["peak_bytes"]
                cell.retained_tokens = retained
                cell.values_changed = retained < shape.seq_len
                cell.did_something = retained < shape.seq_len
                cell.simulated_storage = is_simulated(fp)
                cell.float_byte_fraction = round(float_byte_fraction(fp), 6)
                cell.by_dtype = fp.as_dict()["by_dtype"]
                cell.error = f"budget={heavy}+{recent}={heavy + recent} tokens"
            except Exception as exc:
                cell.error = f"{type(exc).__name__}: {exc}"
            cells.append(cell)
        return cells


ADAPTERS = [H2OOfficialAdapter()]
