"""Turn the gate's cells into a verdict.

    python -m experiments.byte_audit.analyze results.jsonl

The decision rule is fixed here, in the code, so that it is read before the
numbers are. Two claims are tested separately because they need different
evidence:

  A. The axis claim -- quantization-family methods expose fewer distinct
     delivered operating points than eviction-family methods over the same
     requested-ratio grid. If true, plotting both against one "compression
     ratio" axis compares a swept budget against a fixed one.

  B. The accounting claim -- what a method reports is not what it delivers.

Claim B is true by construction for the baselines in this repository, which
dequantize for quality comparison and report an analytic footprint. It becomes
evidence about the field only when measured through adapters onto upstream
implementations. The verdict block labels which tier produced each number.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

TIER_A = "local baselines (instrument validation only)"
TIER_B = "upstream implementations (evidence about the field)"


def classify_knob(n_distinct: int, n_points: int) -> str:
    if n_distinct <= 1:
        return "FLAT"
    if n_distinct >= n_points:
        return "CONTINUOUS"
    return "DISCRETE"


def summarize(cells: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in cells:
        by_method[c["method"]].append(c)

    rows: List[Dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        usable = [c for c in group if c.get("ok") and c.get("did_something")]
        failed = [c for c in group if not c.get("ok")]
        inert = [c for c in group if c.get("ok") and not c.get("did_something")]
        if not usable:
            rows.append(
                {
                    "method": method,
                    "category": group[0].get("category", ""),
                    "status": "UNVERIFIED",
                    "note": f"{len(failed)} errored, {len(inert)} no-op",
                }
            )
            continue

        reported = [c["reported_bytes"] for c in usable if c["reported_bytes"] is not None]
        # Distinct footprints must be counted within one context length. Pooling
        # lengths would count "the context grew" as "the knob moved", which is
        # the very conflation this analysis exists to separate.
        by_len: Dict[Any, List[int]] = defaultdict(list)
        for c in usable:
            if c["structural_bytes"] is not None:
                by_len[c.get("seq_len")].append(c["structural_bytes"])
        per_len_distinct = sorted(len(set(v)) for v in by_len.values())
        delivered_distinct = (
            statistics.median_low(per_len_distinct) if per_len_distinct else 0
        )
        points_per_len = statistics.median_low(sorted(len(v) for v in by_len.values())) or 1
        ratios = [
            c["reported_bytes"] / c["structural_bytes"]
            for c in usable
            if c.get("reported_bytes") and c.get("structural_bytes")
        ]
        # Claim B'. A quantization method states a bit width, and bytes say what
        # that width really cost. Sixteen is the FP16 baseline each cell is
        # measured against, so the quotient is bits actually spent per value.
        bit_pairs = [
            (c["nominal_bits"], 16.0 * c["structural_bytes"] / c["fullkv_bytes"])
            for c in usable
            if c.get("nominal_bits") and c.get("fullkv_bytes")
        ]
        rows.append(
            {
                "method": method,
                "category": group[0].get("category", ""),
                "status": "OK",
                "n_points": len(usable),
                "reported_distinct": len(set(reported)),
                "delivered_distinct": delivered_distinct,
                "seq_lens": sorted(x for x in by_len if x is not None),
                "distinct_varies_by_length": len(set(per_len_distinct)) > 1,
                # A project that publishes no footprint of its own is not a
                # project claiming a flat one. That absence is itself a finding.
                "reported_knob": (
                    classify_knob(len(set(reported)), points_per_len) if reported else "NONE"
                ),
                "delivered_knob": classify_knob(delivered_distinct, points_per_len),
                "reported_over_delivered": round(statistics.median(ratios), 4) if ratios else None,
                "simulated_storage": all(bool(c.get("simulated_storage")) for c in usable),
                "nominal_bits": sorted({n for n, _ in bit_pairs}) or None,
                "effective_bits": (
                    sorted({round(e, 2) for _, e in bit_pairs}) if bit_pairs else None
                ),
                "worst_bit_gap": (
                    round(max(e - n for n, e in bit_pairs), 2) if bit_pairs else None
                ),
                "note": f"{len(failed)} errored, {len(inert)} no-op" if failed or inert else "",
            }
        )
    return rows


def render(rows: Sequence[Dict[str, Any]], tier: str) -> str:
    head = (
        f"{'method':<28}{'family':<14}{'delivered':<12}"
        f"{'nominal bits':>13}{'effective':>22}{'gap':>7}  {'storage':<10}"
    )
    lines = [f"scope: {tier}", "", head, "-" * len(head)]
    for r in rows:
        if r["status"] != "OK":
            lines.append(
                f"{r['method']:<28}{r['category']:<14}{'UNVERIFIED':<12}"
                f"{'-':>13}{'-':>22}{'-':>7}  {r['note']}"
            )
            continue
        storage = "simulated" if r["simulated_storage"] else "packed"
        nominal = ",".join(str(b) for b in r["nominal_bits"]) if r["nominal_bits"] else "-"
        eff = r["effective_bits"]
        if not eff:
            effective = "-"
        elif len(eff) <= 3:
            effective = ",".join(f"{e:g}" for e in eff)
        else:
            # Long lists come from the context-length sweep nudging the metadata
            # share; a range says that without burying the column.
            effective = f"{min(eff):g}-{max(eff):g}"
        gap = r["worst_bit_gap"]
        lines.append(
            f"{r['method']:<28}{r['category']:<14}{r['delivered_knob']:<12}"
            f"{nominal:>13}{effective:>22}{gap if gap is not None else '-':>7}  {storage:<10}"
            f"{r['note']}"
        )
    return "\n".join(lines)


def verdict(rows: Sequence[Dict[str, Any]], tier: str) -> str:
    ok = [r for r in rows if r["status"] == "OK"]
    quant = [r for r in ok if r["category"] == "quantization"]
    evict = [r for r in ok if r["category"] == "eviction"]

    out = ["", "=" * 78, "VERDICT", "=" * 78]
    if not quant or not evict:
        out.append("INCONCLUSIVE: need at least one verified method in each family.")
        return "\n".join(out)

    q_points = statistics.median(r["delivered_distinct"] for r in quant)
    e_points = statistics.median(r["delivered_distinct"] for r in evict)
    axis_holds = q_points < e_points

    # Claim B'. Nobody publishes a footprint, but a bit width is a stated number
    # and bytes can be held against it.
    claimants = [r for r in ok if r["nominal_bits"]]
    gaps = [r for r in claimants if (r["worst_bit_gap"] or 0) >= 1.0]
    frac_gap = len(gaps) / len(claimants) if claimants else 0.0

    out.append(f"A. axis claim   quantization median distinct delivered points = {q_points}")
    out.append(f"                eviction     median distinct delivered points = {e_points}")
    out.append(f"                -> {'HOLDS' if axis_holds else 'DOES NOT HOLD'}")
    if claimants:
        out.append(
            f"B'. bit width   {len(gaps)}/{len(claimants)} quantization methods spend at least "
            f"one bit per value  ({frac_gap:.0%})"
        )
        for r in claimants:
            out.append(
                f"                {r['method']:<28} nominal {r['nominal_bits']} "
                f"-> effective {r['effective_bits']}"
            )
    else:
        out.append("B'. bit width   UNTESTABLE: no verified method states a nominal bit width.")
    out.append("")
    if tier == TIER_A:
        out.append("GATE: not decidable at this tier. These baselines dequantize by design, so")
        out.append("      claim B is true by construction and carries no information about the")
        out.append("      field. What this run establishes is that the instrument works and")
        out.append("      that claim A is measurable. Write adapters and rerun for the verdict.")
    elif axis_holds and claimants and frac_gap >= 0.5:
        out.append("GATE: GO. Both claims hold on upstream code. This is a paper.")
    elif axis_holds and not claimants:
        out.append("GATE: PARTIAL. The axis claim holds on upstream code, but no quantization")
        out.append("      method here states a bit width, so claim B' cannot be tested.")
    elif axis_holds:
        out.append("GATE: PARTIAL. The axis claim holds but the bit-width gap is not broad.")
        out.append("      Narrow the paper to the axis claim or stop.")
    else:
        out.append("GATE: NO-GO. The axis claim does not hold. File the finding as a bug report")
        out.append("      against the one implementation that showed it and archive the project.")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", nargs="?", default="experiments/byte_audit/results.jsonl")
    ap.add_argument("--tier", choices=["a", "b"], default="a")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    cells = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line]
    tier = TIER_A if args.tier == "a" else TIER_B
    # Tier A is this repository's own code and tier B is everyone else's. Mixing
    # them would let our own construction stand in as evidence about the field.
    want_local = args.tier == "a"
    cells = [c for c in cells if (c.get("source", "local") == "local") == want_local]
    if not cells:
        print(f"no cells for tier {args.tier} in {args.results}")
        return 1
    rows = summarize(cells)

    print(render(rows, tier))
    print(verdict(rows, tier))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"tier": tier, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
