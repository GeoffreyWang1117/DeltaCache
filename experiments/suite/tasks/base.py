"""Base class for evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..config import ExperimentUnit, ModelSpec


class BaseTask(ABC):
    """Interface for all evaluation tasks.

    Each task implements:
      - setup(): one-time data loading (cached across units)
      - run_unit(): evaluate one (method, cr, seq_len) combo
      - needs_attention(): whether methods need attention weights
    """

    name: str = ""

    def __init__(self):
        self._data_loaded = False

    @abstractmethod
    def setup(self, tokenizer, device: str) -> None:
        """Load task data. Called once per model switch."""
        ...

    @abstractmethod
    def run_unit(
        self,
        unit: ExperimentUnit,
        model,
        tokenizer,
        spec: ModelSpec,
        gini_scores: Dict[int, float],
        importance_weights: Dict[int, float],
        device: str,
    ) -> Dict[str, Any]:
        """Run a single experiment unit. Returns result dict."""
        ...

    @property
    def needs_attention(self) -> bool:
        """Does this task require attention weights for baselines?"""
        return True

    @property
    def needs_generation(self) -> bool:
        """Does this task call model.generate()?"""
        return False

    def estimate_vram_mb(self, spec: ModelSpec, seq_len: int) -> float:
        """Estimate additional VRAM needed beyond model weights."""
        # KV cache + attention weights + working memory
        kv_mb = spec.kv_bytes_per_token * seq_len / 1024 / 1024
        if self.needs_attention:
            # Attention: (batch, heads, seq, seq) * fp32 * layers
            attn_mb = (spec.num_layers * spec.num_kv_heads
                       * seq_len * seq_len * 4) / 1024 / 1024
        else:
            attn_mb = 0
        return kv_mb + attn_mb + 256  # 256 MB working memory buffer
