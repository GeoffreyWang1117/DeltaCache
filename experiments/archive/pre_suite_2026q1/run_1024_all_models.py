#!/usr/bin/env python3
"""Run 1024-token experiments on Mistral-7B and Llama-3.1-8B.

Fills Table 1 gaps for the NeurIPS 2026 submission.
Uses the hook-based profiler from run_1024_and_mmlu.py.

Usage:
    python run_1024_all_models.py                  # run both models
    python run_1024_all_models.py --model mistral   # run Mistral only
    python run_1024_all_models.py --model llama3     # run Llama-3.1 only
"""

import subprocess
import sys
from pathlib import Path

MODELS = {
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "short": "Mistral-7B",
    },
    "llama3": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "short": "Llama-3.1-8B",
    },
}

SCRIPT = Path(__file__).parent / "run_1024_and_mmlu.py"


def run_model(key: str):
    m = MODELS[key]
    print(f"\n{'='*70}")
    print(f"  Running 1024-token experiment: {m['short']}")
    print(f"{'='*70}\n")

    cmd = [
        sys.executable, str(SCRIPT),
        "--model", m["name"],
        "--model-short", m["short"],
        "--exps", "27",  # 1024-token PPL only, skip MMLU
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: {m['short']} exited with code {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        help="Run specific model only")
    args = parser.parse_args()

    targets = [args.model] if args.model else list(MODELS.keys())

    for key in targets:
        rc = run_model(key)
        if rc != 0:
            print(f"Stopping after {key} failure")
            sys.exit(1)

    print("\nAll models completed successfully!")
    print("Results saved to experiments/results/main_v2/")
