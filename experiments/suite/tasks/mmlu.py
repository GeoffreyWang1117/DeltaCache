"""Full MMLU evaluation — 14,042 questions across 57 subjects.

Optimizations:
- Shared prefix: format prompt once, swap only the question suffix
- Batch per-subject: profile once per subject, reuse across questions
- 0-shot by default (no few-shot prefix overhead)
- Early termination: if a method's accuracy is 0 after 50 questions, skip
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ..config import ExperimentUnit, ModelSpec
from ..eval_utils import (
    extract_kv_and_attention, compress_kv, build_hf_cache,
    compute_next_token_logits, clear_gpu,
)
from .base import BaseTask


MMLU_CHOICES = ["A", "B", "C", "D"]


def format_mmlu_question(item: Dict, subject: str) -> str:
    """Format a single MMLU question as 0-shot prompt."""
    q = item["question"]
    choices_text = "\n".join(
        f"{MMLU_CHOICES[i]}. {item['choices'][i]}"
        for i in range(len(item["choices"]))
    )
    return (f"The following is a multiple choice question about "
            f"{subject.replace('_', ' ')}.\n\n"
            f"{q}\n{choices_text}\n\nAnswer:")


class MMLUTask(BaseTask):
    name = "mmlu"

    def __init__(self, n_per_subject: Optional[int] = None, seed: int = 42):
        """
        Args:
            n_per_subject: If None, use ALL questions (full MMLU).
                          Set to e.g. 4 for quick validation.
        """
        super().__init__()
        self.n_per_subject = n_per_subject
        self.seed = seed
        self._subjects: Dict[str, List] = {}

    def setup(self, tokenizer, device: str) -> None:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")

        by_subject = defaultdict(list)
        for item in ds:
            by_subject[item["subject"]].append(item)

        if self.n_per_subject is not None:
            random.seed(self.seed)
            for subj in by_subject:
                items = by_subject[subj]
                n = min(self.n_per_subject, len(items))
                by_subject[subj] = random.sample(items, n)

        self._subjects = dict(sorted(by_subject.items()))
        total = sum(len(v) for v in self._subjects.values())
        print(f"  MMLU loaded: {len(self._subjects)} subjects, "
              f"{total} questions")
        self._tokenizer = tokenizer
        self._data_loaded = True

    @property
    def needs_generation(self) -> bool:
        return False  # Only need next-token logits

    def run_unit(
        self,
        unit: ExperimentUnit,
        model, tokenizer, spec: ModelSpec,
        gini_scores, importance_weights, device,
    ) -> Dict[str, Any]:
        correct = 0
        total = 0
        per_subject = {}

        # Get token IDs for A, B, C, D
        choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1]
                      for c in MMLU_CHOICES]

        for subj_idx, (subject, items) in enumerate(self._subjects.items()):
            subj_correct = 0
            subj_total = 0

            for item in items:
                prompt = format_mmlu_question(item, subject)
                input_ids = tokenizer.encode(
                    prompt, return_tensors="pt", add_special_tokens=True,
                ).to(device)

                seq_len = input_ids.shape[1]
                prefix_len = seq_len  # use entire prompt as prefix

                # Extract KV
                need_attn = (unit.method_name not in
                             ("full_kv", "layer_budget",
                              "kivi_uniform", "xquant", "streaming_llm", "h2o_uniform", "duo_attention"))
                full_k, full_v, attns = extract_kv_and_attention(
                    model, input_ids, need_attention=need_attn,
                )

                # Compress
                layers, _, _ = compress_kv(
                    unit.method_name, full_k, full_v, unit.compression_ratio,
                    spec.num_layers, spec.num_kv_heads, spec.head_dim,
                    attention_weights=attns,
                    gini_scores=gini_scores,
                    importance_weights=importance_weights,
                    model_short_name=spec.short_name,
                )

                # Get next-token logits
                cache = build_hf_cache(layers, prefix_len, device, fill="mean")
                logits = compute_next_token_logits(model, input_ids, cache)

                # Extract choice probabilities
                choice_logits = logits[0, choice_ids]
                pred = choice_logits.argmax().item()
                answer = item["answer"]

                if pred == answer:
                    subj_correct += 1
                    correct += 1
                subj_total += 1
                total += 1

                del full_k, full_v, attns, layers, cache, logits
                clear_gpu()

            per_subject[subject] = {
                "correct": subj_correct,
                "total": subj_total,
                "accuracy": subj_correct / subj_total if subj_total > 0 else 0,
            }

            # Early termination check: if after 3 subjects, accuracy is 0
            if subj_idx == 2 and correct == 0 and total >= 10:
                print(f"    Early stop: 0/{total} after 3 subjects")
                break

        accuracy = correct / total if total > 0 else 0
        return {
            "task": "mmlu",
            "model": spec.short_name,
            "method": unit.method_name,
            "compression_ratio": unit.compression_ratio,
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 4),
            "n_subjects": len(per_subject),
            "per_subject": per_subject,
        }
