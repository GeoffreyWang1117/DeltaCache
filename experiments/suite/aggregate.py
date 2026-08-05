#!/usr/bin/env python3
"""Aggregate experiment results into paper-ready tables and figures.

Usage:
    python -m suite.aggregate --run-id 20260404_120000
    python -m suite.aggregate --run-id latest --format latex
    python -m suite.aggregate --run-id latest --format csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .checkpoint import CheckpointManager, find_latest_run, SUITE_DIR


def load_results(run_id: str) -> Dict:
    """Load all checkpointed results for a run."""
    ckpt = CheckpointManager(run_id=run_id)
    return ckpt.aggregate_results()


def build_ppl_table(results: Dict) -> Dict:
    """Build the main PPL ratio table (Table 1 format).

    Returns nested dict: {model -> {seq_len -> {method -> {cr -> ratio}}}}
    """
    # First pass: collect all full_kv PPL values
    full_kv_ppl = {}  # (model, seq_len) -> ppl
    for key, data in results.items():
        if data.get("task") != "ppl" or data.get("error"):
            continue
        if data.get("method") == "full_kv":
            k = (data["model"], data["seq_len"])
            full_kv_ppl[k] = data["mean_ppl"]

    # Second pass: compute ratios
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for key, data in results.items():
        if data.get("task") != "ppl" or data.get("error"):
            continue

        model = data["model"]
        seq_len = data["seq_len"]
        method = data["method"]
        cr = data["compression_ratio"]

        if data.get("ppl_ratio") is not None:
            ratio = data["ppl_ratio"]
        else:
            fk = full_kv_ppl.get((model, seq_len))
            ratio = data["mean_ppl"] / fk if fk else None

        if ratio is not None:
            table[model][seq_len][method][cr] = round(ratio, 3)

    return dict(table)


def build_downstream_table(results: Dict, task: str) -> Dict:
    """Build downstream task table.

    Returns: {model -> {method -> {cr -> metric_value}}}
    """
    table = defaultdict(lambda: defaultdict(dict))
    for key, data in results.items():
        if data.get("task") != task or data.get("error"):
            continue
        model = data["model"]
        method = data["method"]
        cr = data["compression_ratio"]

        if task == "mmlu":
            val = data.get("accuracy", 0)
        elif task == "gsm8k":
            val = data.get("accuracy", 0)
        elif task == "longbench":
            val = data.get("overall_mean", 0)
        elif task in ("ruler", "niah"):
            val = data.get("overall_accuracy", 0)
        else:
            continue

        table[model][method][cr] = round(val, 4)

    return dict(table)


def format_latex_ppl(table: Dict, crs: List[float] = None) -> str:
    """Format PPL table as LaTeX."""
    if crs is None:
        crs = [2.0, 3.0, 4.0, 6.0]

    lines = []
    lines.append(r"\begin{tabular}{@{}ll" + "c" * len(crs) + r"@{}}")
    lines.append(r"\toprule")
    cr_header = " & ".join(f"{cr:.0f}$\\times$" for cr in crs)
    lines.append(f"Model & Method & {cr_header} \\\\")
    lines.append(r"\midrule")

    for model in sorted(table.keys()):
        for seq_len in sorted(table[model].keys()):
            lines.append(f"\\multicolumn{{{2 + len(crs)}}}{{c}}"
                         f"{{\\textit{{{model} @ {seq_len} tokens}}}} \\\\")
            lines.append(r"\midrule")

            methods = table[model][seq_len]
            for method in sorted(methods.keys()):
                vals = []
                for cr in crs:
                    r = methods[method].get(cr)
                    if r is not None:
                        s = f"{r:.3f}"
                        if r <= 1.01:
                            s = f"\\textbf{{{s}}}"
                    else:
                        s = "---"
                    vals.append(s)
                line = f"& {method} & {' & '.join(vals)} \\\\"
                lines.append(line)
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def format_csv_ppl(table: Dict, crs: List[float] = None) -> str:
    """Format PPL table as CSV."""
    if crs is None:
        crs = [2.0, 3.0, 4.0, 6.0]

    lines = ["model,seq_len,method," + ",".join(f"cr_{cr}" for cr in crs)]
    for model in sorted(table.keys()):
        for seq_len in sorted(table[model].keys()):
            methods = table[model][seq_len]
            for method in sorted(methods.keys()):
                vals = [str(methods[method].get(cr, "")) for cr in crs]
                lines.append(f"{model},{seq_len},{method},{','.join(vals)}")

    return "\n".join(lines)


def print_summary(results: Dict):
    """Print a concise summary of all results."""
    # Count by task
    by_task = defaultdict(lambda: {"ok": 0, "err": 0})
    for key, data in results.items():
        task = data.get("task", "unknown")
        if data.get("error"):
            by_task[task]["err"] += 1
        else:
            by_task[task]["ok"] += 1

    print(f"\n{'='*50}")
    print(f"  RESULTS SUMMARY ({len(results)} total)")
    print(f"{'='*50}")
    for task in sorted(by_task.keys()):
        ok = by_task[task]["ok"]
        err = by_task[task]["err"]
        print(f"  {task:15s}: {ok:4d} ok, {err:3d} errors")
    print()

    # PPL highlights
    ppl_table = build_ppl_table(results)
    if ppl_table:
        print("  PPL Ratio Highlights (LayerBudget):")
        for model in sorted(ppl_table.keys()):
            for seq_len in sorted(ppl_table[model].keys()):
                lb = ppl_table[model][seq_len].get("layer_budget", {})
                if lb:
                    vals = " / ".join(f"{lb.get(cr, '?')}" for cr in [2, 3, 4, 6])
                    print(f"    {model}@{seq_len}: {vals}")
        print()

    # Downstream highlights
    for task_name in ["mmlu", "gsm8k", "longbench"]:
        dt = build_downstream_table(results, task_name)
        if dt:
            print(f"  {task_name.upper()} Highlights:")
            for model in sorted(dt.keys()):
                for method in ["full_kv", "layer_budget", "h2o_uniform"]:
                    if method in dt[model]:
                        vals = dt[model][method]
                        v_str = ", ".join(f"{cr}x={v}" for cr, v in sorted(vals.items()))
                        print(f"    {model}/{method}: {v_str}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--format", choices=["summary", "latex", "csv"],
                        default="summary")
    parser.add_argument("--task", default="ppl",
                        help="Task to format (for latex/csv)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    args = parser.parse_args()

    run_id = args.run_id
    if run_id == "latest":
        run_id = find_latest_run()
        if not run_id:
            print("No runs found.")
            return

    print(f"Loading results from: {run_id}")
    results = load_results(run_id)
    print(f"Loaded {len(results)} results")

    if args.format == "summary":
        print_summary(results)
    elif args.format == "latex":
        table = build_ppl_table(results)
        output = format_latex_ppl(table)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Written to {args.output}")
        else:
            print(output)
    elif args.format == "csv":
        table = build_ppl_table(results)
        output = format_csv_ppl(table)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Written to {args.output}")
        else:
            print(output)


if __name__ == "__main__":
    main()
