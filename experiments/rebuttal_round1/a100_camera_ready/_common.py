"""Shared helpers for A100 camera-ready scripts.

Provides:
  - per-cell checkpointing (skips already-completed cells on rerun)
  - elapsed-time logging
  - safe path resolution for vast.ai layouts
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


# Resolve repo root regardless of where this script is launched from.
# Strategy: walk up looking for `pyproject.toml`.
def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(8):
        if (p / "pyproject.toml").exists() and (p / "deltacache").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Could not find DeltaCache repo root from {start}")


def setup_paths(this_file: str) -> Path:
    here = Path(this_file).resolve()
    root = find_repo_root(here)
    # Insert paths in priority order
    sys.path.insert(0, str(root / "experiments" / "rebuttal_round1"))  # layer_budget_kv
    sys.path.insert(0, str(root / "experiments"))  # baselines, suite
    sys.path.insert(0, str(root))  # deltacache
    return root


class CellCheckpoint:
    """One JSON file per (model, cr, method) cell.

    Allows rerunning a script after a pre-emption / OOM without losing
    completed cells. Mirrors experiments/suite/checkpoint.py.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def key_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self.run_dir / f"{safe}.json"

    def is_done(self, key: str) -> bool:
        return self.key_path(key).exists()

    def save(self, key: str, result: dict) -> None:
        result["_checkpoint_key"] = key
        result["_saved_at"] = datetime.utcnow().isoformat()
        with open(self.key_path(key), "w") as f:
            json.dump(result, f, indent=2)

    def load(self, key: str) -> Optional[dict]:
        p = self.key_path(key)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def aggregate(self, output_path: Path) -> None:
        """Aggregate all cell JSONs into one bundle."""
        bundle = {"date": datetime.utcnow().isoformat(), "cells": []}
        for p in sorted(self.run_dir.glob("*.json")):
            if p.name.startswith("_aggregate"):
                continue
            with open(p) as f:
                bundle["cells"].append(json.load(f))
        with open(output_path, "w") as f:
            json.dump(bundle, f, indent=2)


class Timer:
    """Simple phase timer with logging."""

    def __init__(self, label: str):
        self.label = label
        self.t0 = time.time()
        print(f"[T] {label} starting", flush=True)

    def end(self) -> float:
        elapsed = time.time() - self.t0
        m, s = divmod(elapsed, 60)
        print(f"[T] {self.label} done in {int(m)}m{int(s):02d}s", flush=True)
        return elapsed


def announce(msg: str) -> None:
    """Print a banner-style log line that's easy to grep."""
    bar = "=" * 70
    print(f"\n{bar}\n# {msg}\n{bar}", flush=True)
