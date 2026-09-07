"""The gate: does the compression-ratio knob move the bytes a method delivers?

Runs every registered baseline across a grid of requested compression ratios on
synthetic tensors, and records three independent numbers per cell: what the
method reports, what its returned tensors actually hold, and what the allocator
saw. No model, no tokenizer, no GPU required.

    python -m experiments.byte_audit.gate --out results.jsonl

A cell is only admitted to the verdict if the method demonstrably did something
(dropped tokens or changed values). Without that check a broken call site and a
method with a genuinely flat footprint produce identical readings.

Scope: the baselines in this repository return dequantized FP16 for quality
comparison and report an analytic footprint. Measuring them validates the
instrument and quantifies our own reported-vs-delivered gap. It says nothing
about upstream implementations -- for that, write an adapter (see README).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from experiments.byte_audit.accountant import (  # noqa: E402
    allocator_probe,
    float_byte_fraction,
    is_simulated,
    measure_structural,
)

DEFAULT_GRID = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)


# Attention weights are quadratic in sequence length. Past this, refuse rather
# than quietly shrinking the request or exhausting the machine.
_ATTENTION_BYTE_BUDGET = 4 << 30  # 4 GiB for one layer
_ATTENTION_CACHE_BUDGET = 2 << 30  # 2 GiB held across layers


@dataclass
class Shape:
    """Synthetic cache geometry. Small by default; the audit is about bytes."""

    num_layers: int = 8
    num_heads: int = 8
    head_dim: int = 64
    seq_len: int = 512
    hidden_dim: int = 512
    # Key/value heads, fewer than query heads under grouped-query attention,
    # which every current model uses and which changes cache bytes directly.
    num_kv_heads: Optional[int] = None

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_heads

    @property
    def is_gqa(self) -> bool:
        return self.kv_heads != self.num_heads


# Geometries of models cached locally, so a sweep can be run at a real shape
# rather than a convenient one.
GEOMETRIES: Dict[str, Dict[str, int]] = {
    "mha-toy": {"num_heads": 8, "num_kv_heads": 8, "head_dim": 64, "num_layers": 8},
    "qwen2.5-7b": {"num_heads": 28, "num_kv_heads": 4, "head_dim": 128, "num_layers": 28},
    "qwen3-8b": {"num_heads": 32, "num_kv_heads": 8, "head_dim": 128, "num_layers": 36},
    "llama-2-7b": {"num_heads": 32, "num_kv_heads": 32, "head_dim": 128, "num_layers": 32},
}


@dataclass
class Cell:
    """One (method, compression_ratio) measurement."""

    method: str
    category: str
    compression_ratio: float
    ok: bool
    # "local" for this repository's baselines, "adapter:<name>" for upstream
    # code. The verdict is computed per source so tiers cannot be mixed.
    source: str = "local"
    error: Optional[str] = None
    reported_bytes: Optional[int] = None
    structural_bytes: Optional[int] = None
    logical_bytes: Optional[int] = None
    view_overhead: Optional[float] = None
    # Bit width a quantization method was asked for. The delivered
    # equivalent is derived from bytes, so the two can be compared.
    nominal_bits: Optional[int] = None
    fullkv_bytes: Optional[int] = None
    allocator_peak_bytes: Optional[int] = None
    retained_tokens: Optional[int] = None
    seq_len: Optional[int] = None
    values_changed: Optional[bool] = None
    did_something: Optional[bool] = None
    simulated_storage: Optional[bool] = None
    float_byte_fraction: Optional[float] = None
    by_dtype: Optional[Dict[str, int]] = None
    elapsed_ms: Optional[float] = None


class LazyAttention:
    """Per-layer attention weights, built on access and not before.

    Only eviction methods need these, they are quadratic in sequence length, and
    the sweep runs to lengths where materialising every layer at once would not
    fit. Indexing builds one layer; the memory guard refuses rather than swaps.
    """

    def __init__(self, shape: Shape, device: torch.device, seed: int) -> None:
        self._shape = shape
        self._device = device
        self._seed = seed
        self._cache: Dict[int, Tensor] = {}

    def __len__(self) -> int:
        return self._shape.num_layers

    def __getitem__(self, layer: int) -> Tensor:
        # Bounds first, and IndexError specifically: the legacy iteration
        # protocol walks __getitem__ from zero and stops only on IndexError.
        # Without this a caller writing `for a in attention_weights` never
        # terminates, building a fresh tensor for every index forever.
        n = self._shape.num_layers
        if layer < 0:
            layer += n
        if not 0 <= layer < n:
            raise IndexError(f"layer {layer} out of range for {n} layers")
        if layer in self._cache:
            return self._cache[layer]
        s = self._shape
        need = s.num_heads * s.seq_len * s.seq_len * 2
        if need > _ATTENTION_BYTE_BUDGET:
            raise MemoryError(
                f"attention for one layer at seq_len={s.seq_len} over {s.num_heads} heads "
                f"needs {need / 2**30:.1f} GiB, above the {_ATTENTION_BYTE_BUDGET / 2**30:.0f} "
                "GiB budget; this cell is skipped rather than silently shrunk"
            )
        g = torch.Generator(device="cpu").manual_seed(self._seed + 1000 + layer)
        raw = torch.rand(
            (1, s.num_heads, s.seq_len, s.seq_len), generator=g, dtype=torch.float32
        )
        out = torch.softmax(raw, dim=-1).half().to(self._device)
        # Keep layers while they fit. Callers revisit the same layer many times,
        # and rebuilding on every access costs more than the memory saved at the
        # lengths where everything fits anyway. Past the budget, hold one.
        if (len(self._cache) + 1) * need > _ATTENTION_CACHE_BUDGET:
            self._cache.clear()
        self._cache[layer] = out
        return out


def make_inputs(shape: Shape, device: torch.device, seed: int = 0) -> Dict[str, Any]:
    """Deterministic synthetic KV cache, attention weights and hidden states."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    size = (shape.num_layers, shape.seq_len, shape.kv_heads, shape.head_dim)
    keys = torch.randn(size, generator=g, dtype=torch.float32).half().to(device)
    values = torch.randn(size, generator=g, dtype=torch.float32).half().to(device)

    hidden = [
        torch.randn((1, shape.seq_len, shape.hidden_dim), generator=g, dtype=torch.float32)
        .half()
        .to(device)
        for _ in range(shape.num_layers)
    ]
    return {
        "keys": keys,
        "values": values,
        "attention_weights": LazyAttention(shape, device, seed),
        "hidden_states": hidden,
    }


def fullkv_reference(keys: Tensor, values: Tensor) -> int:
    """Bytes the uncompressed cache occupies, measured the same way as a cell."""
    return measure_structural([keys, values]).total_bytes


def _did_something(
    layers: Sequence[Tuple[Tensor, Tensor, Tensor]],
    keys: Tensor,
    seq_len: int,
) -> Tuple[int, bool, bool]:
    """Positive control.

    Returns (retained_tokens, values_changed, did_something). A method that
    retained every token and returned bit-identical values did nothing, and its
    byte reading is meaningless regardless of what it reports.
    """
    k0, _, idx0 = layers[0]
    retained = int(idx0.numel())
    ref = keys[0].index_select(0, idx0.to(keys.device))
    got = k0.reshape(ref.shape) if k0.numel() == ref.numel() else None
    changed = bool(got is not None and not torch.equal(got, ref))
    return retained, changed, bool(retained < seq_len or changed)


def run_local_baselines(
    shape: Shape,
    grid: Sequence[float],
    device: torch.device,
    only: Optional[Sequence[str]] = None,
) -> List[Cell]:
    """Sweep every baseline in the repository registry across the ratio grid."""
    # Imported here rather than at module scope: the import is what populates
    # the baseline registry, and the audit must not require it to merely load.
    import experiments.baselines as bl

    data = make_inputs(shape, device)
    full_bytes = fullkv_reference(data["keys"], data["values"])
    cells: List[Cell] = []

    for name, cls in sorted(bl.REGISTRY.items()):
        if only and name not in only:
            continue
        for cr in grid:
            cell = Cell(
                method=name,
                category=getattr(cls, "category", ""),
                compression_ratio=float(cr),
                ok=False,
                source="local",
                fullkv_bytes=full_bytes,
                seq_len=shape.seq_len,
            )
            try:
                method = cls(
                    num_layers=shape.num_layers,
                    num_heads=shape.num_heads,
                    head_dim=shape.head_dim,
                )
                t0 = time.perf_counter()
                with allocator_probe(device) as probe:
                    layers = method.compress(
                        data["keys"],
                        data["values"],
                        float(cr),
                        attention_weights=data["attention_weights"],
                        hidden_states=data["hidden_states"],
                    )
                cell.elapsed_ms = (time.perf_counter() - t0) * 1000.0

                fp = measure_structural(layers)
                retained, changed, acted = _did_something(layers, data["keys"], shape.seq_len)

                cell.ok = True
                cell.reported_bytes = int(method.memory_bytes(layers))
                cell.structural_bytes = fp.total_bytes
                cell.logical_bytes = fp.logical_bytes
                cell.view_overhead = round(fp.view_overhead, 4)
                cell.allocator_peak_bytes = probe["peak_bytes"]
                cell.retained_tokens = retained
                cell.values_changed = changed
                cell.did_something = acted
                cell.simulated_storage = is_simulated(fp)
                cell.float_byte_fraction = round(float_byte_fraction(fp), 6)
                cell.by_dtype = fp.as_dict()["by_dtype"]
            except Exception as exc:  # a broken baseline must not hide the others
                cell.error = f"{type(exc).__name__}: {exc}"
            cells.append(cell)
    return cells


def load_external_adapters(paths: Sequence[str]) -> List[Any]:
    """Import adapter modules for upstream repositories (see README, tier B)."""
    adapters: List[Any] = []
    for p in paths:
        path = Path(p).resolve()
        name = f"_byte_audit_adapter_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load adapter module: {path}")
        mod = importlib.util.module_from_spec(spec)
        # Registered before execution: @dataclass resolves annotations through
        # sys.modules[cls.__module__], which is None for an unregistered module.
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(name, None)
            raise
        if not hasattr(mod, "ADAPTERS"):
            raise AttributeError(f"{path} defines no ADAPTERS list")
        adapters.extend(mod.ADAPTERS)
    return adapters


def write_cells(path: Path, cells: Sequence[Cell], mode: str = "w") -> None:
    """Append cells to the results file, so partial work survives a later crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as fh:
        for cell in cells:
            fh.write(json.dumps(asdict(cell)) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="experiments/byte_audit/results.jsonl")
    ap.add_argument("--grid", type=float, nargs="+", default=list(DEFAULT_GRID))
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512],
                    help="sweep these context lengths; one full grid per length")
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--geometry", choices=sorted(GEOMETRIES), default="mha-toy",
                    help="cache shape to synthesise; non-toy shapes use grouped-query heads")
    ap.add_argument("--device", default="cpu", help="cpu or cuda; cpu is enough")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these method names")
    ap.add_argument("--skip-local", action="store_true",
                    help="run adapters only. Tier A validates the instrument and does not "
                         "need long contexts; sweeping it there costs memory for no evidence")
    ap.add_argument("--adapters", nargs="*", default=[], help="external adapter modules")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    out = Path(args.out)
    geometry = dict(GEOMETRIES[args.geometry])
    if args.num_layers is not None:
        geometry["num_layers"] = args.num_layers
    adapters = load_external_adapters(args.adapters)

    cells: List[Cell] = []
    mode = "w"
    for seq_len in args.seq_lens:
        shape = Shape(seq_len=seq_len, **geometry)
        if not args.skip_local:
            run = run_local_baselines(shape, args.grid, device, only=args.only)
            # Flushed as each stage completes. An adapter that cannot reach its
            # backend is supposed to raise loudly, and that must not cost the
            # rows already measured.
            write_cells(out, run, mode=mode)
            mode = "a"
            cells.extend(run)

        for adapter in adapters:
            new = adapter.run(shape, args.grid, device)
            write_cells(out, new, mode=mode)
            mode = "a"
            cells.extend(new)

    n_ok = sum(1 for c in cells if c.ok)
    n_acted = sum(1 for c in cells if c.did_something)
    lens = ",".join(str(s) for s in args.seq_lens)
    print(
        f"wrote {len(cells)} cells to {out}  ({n_ok} ran, {n_acted} passed positive control) "
        f"geometry={args.geometry} seq_lens={lens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
