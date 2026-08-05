"""Checkpoint/resume system for experiment runs.

Each completed (model, task, method, config) unit is saved as a JSON file.
On resume, completed units are skipped. This is critical for rented servers
where preemption can happen at any time.

Directory layout:
    results/suite/{run_id}/
        checkpoints/
            {checkpoint_key}.json    # one per completed unit
        profiles/
            {model_key}.json         # cached Gini profiles
        summary.json                 # aggregated results
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

SUITE_DIR = Path(__file__).parent.parent / "results" / "suite"


class CheckpointManager:
    """Manages experiment checkpoints for resume capability."""

    def __init__(self, run_id: Optional[str] = None):
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.run_dir = SUITE_DIR / run_id
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.profile_dir = self.run_dir / "profiles"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._completed: Optional[Set[str]] = None

    @property
    def completed_keys(self) -> Set[str]:
        """Lazily scan checkpoint directory for completed keys."""
        if self._completed is None:
            self._completed = set()
            for f in self.ckpt_dir.glob("*.json"):
                self._completed.add(f.stem)
        return self._completed

    def is_done(self, checkpoint_key: str) -> bool:
        return checkpoint_key in self.completed_keys

    def save_result(self, checkpoint_key: str, result: Dict[str, Any]) -> None:
        """Save a completed experiment unit."""
        result["_checkpoint_key"] = checkpoint_key
        result["_timestamp"] = datetime.now().isoformat()
        path = self.ckpt_dir / f"{checkpoint_key}.json"
        path.write_text(json.dumps(result, indent=2, default=str))
        self.completed_keys.add(checkpoint_key)

    def load_result(self, checkpoint_key: str) -> Optional[Dict]:
        path = self.ckpt_dir / f"{checkpoint_key}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def save_profile(self, model_key: str, profile_data: Dict) -> None:
        """Cache Gini profile for a model (reusable across all tasks)."""
        path = self.profile_dir / f"{model_key}.json"
        path.write_text(json.dumps(profile_data, indent=2))

    def load_profile(self, model_key: str) -> Optional[Dict]:
        path = self.profile_dir / f"{model_key}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def aggregate_results(self) -> Dict[str, Any]:
        """Aggregate all checkpointed results into a summary."""
        results = {}
        for f in sorted(self.ckpt_dir.glob("*.json")):
            data = json.loads(f.read_text())
            key = data.get("_checkpoint_key", f.stem)
            results[key] = data
        return results

    def save_summary(self, summary: Dict) -> None:
        path = self.run_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str))

    def progress(self, total: int) -> str:
        done = len(self.completed_keys)
        pct = done / total * 100 if total > 0 else 0
        return f"{done}/{total} ({pct:.0f}%)"


def find_latest_run() -> Optional[str]:
    """Find the most recent run_id for resuming."""
    if not SUITE_DIR.exists():
        return None
    runs = sorted(SUITE_DIR.iterdir(), reverse=True)
    for r in runs:
        if (r / "checkpoints").is_dir():
            return r.name
    return None
