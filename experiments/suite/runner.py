"""Main experiment runner with cost optimization.

Orchestrates the full experiment pipeline:
1. Build experiment matrix based on tier + hardware
2. Estimate cost (dry-run mode)
3. Group by model → load once, run all tasks
4. Checkpoint each unit → resume on interruption
5. OOM recovery → skip unit, continue with next
6. Aggregate results → summary JSON + LaTeX tables

Usage:
    from suite.runner import ExperimentRunner
    runner = ExperimentRunner(tier="paper", hardware="a100_80g")
    runner.dry_run()    # show cost estimate
    runner.run()        # execute all experiments
    runner.run(resume_from="20260404_120000")  # resume interrupted run
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .config import (
    ExperimentUnit, ModelSpec, MODEL_ZOO,
    build_experiment_matrix, estimate_cost,
)
from .checkpoint import CheckpointManager, find_latest_run
from .model_pool import ModelPool, clear_gpu
from .tasks import TASK_REGISTRY


class ExperimentRunner:
    """Orchestrates the complete experiment suite."""

    def __init__(
        self,
        tier: str = "paper",
        hardware: str = "a100_80g",
        gpu_speedup: float = 2.0,
        model_size: Optional[str] = None,
    ):
        self.tier = tier
        self.hardware = hardware
        self.gpu_speedup = gpu_speedup
        self.model_size = model_size
        self.units = build_experiment_matrix(tier, hardware, model_size=model_size)

    def dry_run(self) -> Dict:
        """Print cost estimate without running anything."""
        est = estimate_cost(self.units, self.hardware, self.gpu_speedup)

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT SUITE — DRY RUN")
        print(f"{'='*60}")
        print(f"  Tier:       {self.tier}")
        print(f"  Hardware:   {est['gpu']} ({est['hardware']})")
        print(f"  Cost/hour:  ${est['cost_per_hour']:.2f}")
        print(f"  Models:     {est['n_models']}")
        print(f"  Units:      {est['n_units']}")
        print(f"{'='*60}")
        print(f"\n  Per-task breakdown:")
        for task, secs in sorted(est["per_task_seconds"].items()):
            hrs = secs / 3600
            cost = hrs * est["cost_per_hour"]
            count = sum(1 for u in self.units if u.task_name == task)
            print(f"    {task:15s}: {count:4d} units, "
                  f"{hrs:5.1f}h, ${cost:6.2f}")
        print(f"\n  Model loading:  {est['model_load_seconds']/60:.1f} min")
        print(f"  {'─'*40}")
        print(f"  TOTAL:          {est['total_hours']:.1f}h, "
              f"${est['estimated_cost_usd']:.2f}")
        print(f"{'='*60}\n")

        # Show model × task matrix
        models = sorted(set(u.model_key for u in self.units))
        tasks = sorted(set(u.task_name for u in self.units))
        print("  Model × Task unit counts:")
        header = f"  {'model':20s}" + "".join(f"{t:>12s}" for t in tasks)
        print(header)
        for m in models:
            counts = []
            for t in tasks:
                c = sum(1 for u in self.units
                        if u.model_key == m and u.task_name == t)
                counts.append(f"{c:>12d}")
            print(f"  {m:20s}" + "".join(counts))
        print()

        return est

    def run(self, resume_from: Optional[str] = None) -> Dict:
        """Execute the full experiment suite.

        Args:
            resume_from: Run ID to resume from. If None, starts fresh.
                         Pass "latest" to auto-detect.
        """
        # Setup checkpoint manager
        if resume_from == "latest":
            resume_from = find_latest_run()
            if resume_from:
                print(f"Resuming from: {resume_from}")

        ckpt = CheckpointManager(run_id=resume_from)
        pool = ModelPool(ckpt)

        # Initialize task instances (reused across models).
        # For local validation runs (small models), use lightweight task configs
        # so we don't burn hours on full 14K MMLU or 500-question GSM8K.
        task_kwargs = {}
        if self.model_size == "small":
            task_kwargs["mmlu"] = {"n_per_subject": 4}      # 228 questions
            task_kwargs["gsm8k"] = {"max_samples": 200}
            task_kwargs["math"] = {"max_samples": 200}       # 5 per subject
            task_kwargs["longbench"] = {"max_samples": 10}
            task_kwargs["ruler"] = {"n_samples": 10}
            task_kwargs["niah"] = {"n_depths": 5, "n_repeats": 2}
            task_kwargs["throughput"] = {"n_warmup": 1, "n_measure": 3}
        elif self.model_size in ("7b", "13b"):
            # Standard paper configs: enough for stable estimates, not overkill.
            # MMLU: 20 per subject ≈ 1140 questions (vs full 14K = 12× faster)
            # MATH: 100 questions (vs 200 default = 2× faster)
            # GSM8K: 200 questions (vs 500 default = 2.5× faster)
            task_kwargs["mmlu"] = {"n_per_subject": 20}      # ~1140 questions
            task_kwargs["gsm8k"] = {"max_samples": 200}       # enough for trend
            task_kwargs["math"] = {"max_samples": 200}        # enough for trend
            task_kwargs["longbench"] = {"max_samples": 20}
            task_kwargs["niah"] = {"n_depths": 5, "n_repeats": 3}
            task_kwargs["ruler"] = {"n_samples": 40}

        task_instances = {}
        for name, cls in TASK_REGISTRY.items():
            task_instances[name] = cls(**task_kwargs.get(name, {}))

        # Group units by model for efficient loading
        by_model = defaultdict(list)
        for u in self.units:
            by_model[u.model_key].append(u)

        # Track progress
        total = len(self.units)
        skipped = sum(1 for u in self.units if ckpt.is_done(u.checkpoint_key))
        errors = []
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT SUITE — RUNNING")
        print(f"  Total units: {total}, Already done: {skipped}")
        print(f"  Run ID: {ckpt.run_id}")
        print(f"{'='*60}\n")

        for model_key in sorted(by_model.keys()):
            model_units = by_model[model_key]
            pending = [u for u in model_units
                       if not ckpt.is_done(u.checkpoint_key)]

            if not pending:
                print(f"\n[{model_key}] All {len(model_units)} units done, skipping")
                continue

            print(f"\n[{model_key}] {len(pending)}/{len(model_units)} units pending")

            # Load model
            try:
                model, tokenizer = pool.get(model_key)
            except Exception as e:
                print(f"  ERROR loading {model_key}: {e}")
                errors.extend([(u.checkpoint_key, str(e)) for u in pending])
                continue

            spec = pool.spec
            device = str(next(model.parameters()).device)

            # Setup tasks for this model (data loading)
            loaded_tasks = set()
            for u in pending:
                if u.task_name not in loaded_tasks:
                    print(f"  Setting up task: {u.task_name}")
                    try:
                        task_instances[u.task_name].setup(tokenizer, device)
                        loaded_tasks.add(u.task_name)
                    except Exception as e:
                        print(f"  ERROR setting up {u.task_name}: {e}")

            # Sort pending by priority then task (minimize task switching)
            pending.sort(key=lambda u: (u.priority, u.task_name, u.compression_ratio))

            # Run units
            for i, unit in enumerate(pending):
                if unit.task_name not in loaded_tasks:
                    continue

                ck = unit.checkpoint_key
                elapsed = time.time() - start_time
                done_so_far = skipped + i
                rate = done_so_far / elapsed if elapsed > 0 else 0
                eta_min = (total - done_so_far) / rate / 60 if rate > 0 else 0

                print(f"  [{ckpt.progress(total)}] "
                      f"{unit.task_name}/{unit.method_name}/"
                      f"cr{unit.compression_ratio}"
                      f"{'/' + str(unit.seq_len) + 'tok' if unit.seq_len else ''}"
                      f"  (ETA: {eta_min:.0f}min)", end="", flush=True)

                try:
                    task = task_instances[unit.task_name]
                    result = task.run_unit(
                        unit, model, tokenizer, spec,
                        pool.gini_scores, pool.importance_weights, device,
                    )
                    ckpt.save_result(ck, result)
                    print(f"  ✓", flush=True)

                except torch.cuda.OutOfMemoryError:
                    clear_gpu()
                    err_msg = "CUDA OOM"
                    print(f"  ✗ OOM", flush=True)
                    ckpt.save_result(ck, {"error": err_msg, **_unit_meta(unit, spec)})
                    errors.append((ck, err_msg))

                except Exception as e:
                    clear_gpu()
                    err_msg = f"{type(e).__name__}: {e}"
                    print(f"  ✗ {err_msg}", flush=True)
                    traceback.print_exc()
                    ckpt.save_result(ck, {"error": err_msg, **_unit_meta(unit, spec)})
                    errors.append((ck, err_msg))

            # Release model before loading next
            pool.release()

        # Aggregate
        total_time = time.time() - start_time
        summary = {
            "run_id": ckpt.run_id,
            "tier": self.tier,
            "hardware": self.hardware,
            "total_units": total,
            "completed": len(ckpt.completed_keys),
            "errors": len(errors),
            "total_time_seconds": round(total_time, 1),
            "error_details": errors[:20],  # cap at 20
        }
        ckpt.save_summary(summary)

        print(f"\n{'='*60}")
        print(f"  COMPLETE: {summary['completed']}/{total} units")
        print(f"  Errors: {summary['errors']}")
        print(f"  Time: {total_time/3600:.1f}h")
        print(f"  Results: {ckpt.run_dir}")
        print(f"{'='*60}\n")

        return summary


def _unit_meta(unit: ExperimentUnit, spec: ModelSpec) -> Dict:
    return {
        "task": unit.task_name,
        "model": spec.short_name,
        "method": unit.method_name,
        "compression_ratio": unit.compression_ratio,
        "seq_len": unit.seq_len,
    }
