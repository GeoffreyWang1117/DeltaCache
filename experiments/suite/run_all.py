#!/usr/bin/env python3
import sys
sys.setrecursionlimit(50000)
from suite.single_process_guard import acquire_single_instance_lock
acquire_single_instance_lock()

"""Entry point for the unified experiment suite.

────────────────────────────────────────────────────────────────
Phased execution workflow (GPU resource aware)
────────────────────────────────────────────────────────────────

Phase 1 — Local, free (validates all code, small models only)
  python -m suite.run_all --tier full --model-size small \\
      --hardware local_3090x2_9gb

Phase 2 — Local 7B (short context, 4-bit, fits on 2×3090 with 9GB free)
  python -m suite.run_all --tier paper --model-size 7b \\
      --hardware local_3090x2_9gb --tasks ppl,mmlu

Phase 3 — Rented A100: 7B long-context + 13B all-context
  python -m suite.run_all --tier paper --model-size 7b \\
      --hardware a100_80g
  python -m suite.run_all --tier paper --model-size 13b \\
      --hardware a100_80g

Phase 4 — Rented H100×2: 70B (only if resources allow)
  python -m suite.run_all --tier full --model-size 70b \\
      --hardware h100_80gx2

────────────────────────────────────────────────────────────────
Other common usage
────────────────────────────────────────────────────────────────

  # Cost estimate (no GPU needed):
  python -m suite.run_all --dry-run --tier paper --hardware a100_80g

  # Quick smoke test (smallest model, 1 CR, 1 seq_len):
  python -m suite.run_all --tier quick --model-size small \\
      --hardware local_3090x2_9gb

  # Resume interrupted run:
  python -m suite.run_all --resume latest

  # Only specific tasks:
  python -m suite.run_all --tier paper --model-size 7b --tasks ppl,longbench

  # Only specific models (override model-size):
  python -m suite.run_all --tier paper --models mistral-7b,llama2-13b
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="LayerBudget unified experiment suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--tier", choices=["quick", "paper", "full"],
                        default="paper",
                        help="Experiment scope (default: paper)")
    parser.add_argument("--hardware", default="a100_80g",
                        help="Hardware profile key (default: a100_80g). "
                             "Use local_3090x2_9gb for local phased runs.")
    parser.add_argument("--gpu-speedup", type=float, default=2.0,
                        help="Speed multiplier vs RTX 3090 (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show cost estimate, don't run")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume run ID (or 'latest')")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task filter (e.g., ppl,mmlu)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model filter (overrides --model-size)")
    parser.add_argument("--model-size",
                        choices=["small", "7b", "13b", "70b"], default=None,
                        help="Run only models in this size tier. "
                             "small=Qwen/TinyLlama (local), 7b=Llama-2/Mistral/Llama-3.1, "
                             "13b=Llama-2-13B/Qwen2.5-14B, 70b=Llama-3.1-70B")

    args = parser.parse_args()

    from .runner import ExperimentRunner
    from .config import build_experiment_matrix

    runner = ExperimentRunner(
        tier=args.tier,
        hardware=args.hardware,
        gpu_speedup=args.gpu_speedup,
        model_size=args.model_size,
    )

    # Apply filters (--models overrides --model-size at the unit level)
    if args.tasks:
        task_filter = set(args.tasks.split(","))
        runner.units = [u for u in runner.units if u.task_name in task_filter]

    if args.models:
        model_filter = set(args.models.split(","))
        runner.units = [u for u in runner.units if u.model_key in model_filter]

    if args.dry_run:
        runner.dry_run()
        return

    if not runner.units:
        print("No experiment units match the given filters.")
        return

    runner.run(resume_from=args.resume)


if __name__ == "__main__":
    main()
