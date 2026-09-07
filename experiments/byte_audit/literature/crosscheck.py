"""Cross-check published comparison axes against measured operating points.

The audit measures how many footprints an implementation can actually deliver.
This asks what axis published tables compare along, and flags the case where a
table sweeps more columns than the baseline in it has operating points.

    python -m experiments.byte_audit.literature.crosscheck

What the rows can and cannot support is worth stating plainly. We do not
re-run anyone's experiments and cannot say a published number is wrong. What is
checkable is structural: whether a table places a fixed-bit-width method on an
axis with more columns than that method has representable settings.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
DEFAULT_CLAIMS = _HERE / "claims.csv"
DEFAULT_RESULTS = _HERE.parent / "results.jsonl"


def measured_operating_points(results: Path) -> Dict[str, int]:
    """Distinct delivered footprints per upstream quantization method."""
    if not results.exists():
        return {}
    seen: Dict[str, set] = defaultdict(set)
    for line in results.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        c = json.loads(line)
        if not c.get("ok") or not c.get("did_something"):
            continue
        if c.get("source", "local") == "local" or c.get("category") != "quantization":
            continue
        seen[c["method"]].add(c["structural_bytes"])
    return {m: len(v) for m, v in sorted(seen.items())}


def render(rows: Sequence[Dict[str, str]], points: Dict[str, int]) -> str:
    out: List[str] = []
    out.append("Published comparison axes")
    out.append("=" * 78)
    head = f"{'paper':<32}{'family':<13}{'axis':<14}{'levels':>7}{'quant baseline':>16}"
    out.append(head)
    out.append("-" * len(head))
    for r in rows:
        levels = len([x for x in r["axis_levels"].split(";") if x])
        out.append(
            f"{r['bib_key']:<32}{r['paper_family']:<13}{r['axis_kind']:<14}"
            f"{levels:>7}{r['quant_baseline_present']:>16}"
        )

    cross = [r for r in rows if r["cross_family_axis"] == "yes"]
    out.append("")
    out.append(f"Tables placing both families on one ratio axis: {len(cross)}/{len(rows)}")
    for r in cross:
        levels = len([x for x in r["axis_levels"].split(";") if x])
        out.append(f"  {r['bib_key']}: {levels} ratio columns over a fixed-bit-width baseline")
        out.append(f"    {r['note']}")

    out.append("")
    out.append("Measured operating points, upstream quantization implementations")
    if points:
        for m, n in points.items():
            out.append(f"  {m:<34}{n} distinct delivered footprint(s)")
    else:
        out.append("  none; run the gate with tier-B adapters first")

    out.append("")
    out.append("Reading")
    out.append("-" * 78)
    out.append("Within a family the axis is the native knob: eviction papers index by token")
    out.append("budget, quantization papers by bit width. The mismatch appears only where a")
    out.append("table sweeps a ratio across both families, and that is rare, because the two")
    out.append("families largely do not appear in the same table at all.")
    out.append("")
    out.append("So the finding is not that the field miscompares routinely. It is that the")
    out.append("cross-family comparison is the one nobody grounds, and the measurements above")
    out.append("say why: a fixed-bit-width method has two or three delivered footprints, so a")
    out.append("ratio axis with more columns than that cannot be matched at every column.")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--claims", default=str(DEFAULT_CLAIMS))
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    args = ap.parse_args(argv)

    with Path(args.claims).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(render(rows, measured_operating_points(Path(args.results))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
