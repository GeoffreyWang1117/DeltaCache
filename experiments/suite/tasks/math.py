"""MATH benchmark (Hendrycks et al., 2021).

Tests advanced mathematical reasoning across 7 subjects (Algebra, Counting,
Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus) and
5 difficulty levels.  Compared to GSM8K, MATH problems require longer chain-
of-thought (≈400-1000 tokens of reasoning) and more sophisticated symbolic
manipulation, so KV cache compression has more opportunity to break things.

Evaluation: extract the boxed final answer (\\boxed{...}) and compare to gold,
after string normalization (whitespace, fractions, leading zeros).

Reference:
    Hendrycks et al., "Measuring Mathematical Problem Solving with the
    MATH Dataset", NeurIPS 2021.  https://arxiv.org/abs/2103.03874
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..config import ExperimentUnit, ModelSpec
from ..eval_utils import (
    extract_kv_and_attention, compress_kv, build_hf_cache,
    clear_gpu, generate_from_cache,
)
from .base import BaseTask


# ── Answer extraction / normalization ────────────────────────────────

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed_answer(text: str) -> str:
    """Extract content inside the LAST \\boxed{...} in the text."""
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    # Fallback: look for "answer is" pattern
    m = re.search(r"answer\s*is\s*[:=]?\s*([^\n.]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Last resort: last number in text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def normalize_math_answer(s: str) -> str:
    """Light normalization for string equality on MATH answers."""
    if not s:
        return ""
    s = s.strip()
    # Strip outer parens / dollar signs / spaces
    s = s.strip("$").strip()
    # Collapse whitespace
    s = re.sub(r"\s+", "", s)
    # Strip trailing period
    s = s.rstrip(".")
    # Common LaTeX simplifications
    s = s.replace("\\!", "")
    s = s.replace("\\,", "")
    s = s.replace("\\;", "")
    s = s.replace("\\left", "").replace("\\right", "")
    # 0.5 ↔ .5
    if s.startswith("."):
        s = "0" + s
    # Strip "+"
    if s.startswith("+"):
        s = s[1:]
    return s


def is_correct(pred: str, gold: str) -> bool:
    p = normalize_math_answer(pred)
    g = normalize_math_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    # Try numeric comparison
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


# ── Task class ───────────────────────────────────────────────────────

class MATHTask(BaseTask):
    name = "math"

    def __init__(self, max_samples: int = 200, seed: int = 42,
                 subjects: List[str] | None = None):
        super().__init__()
        self.max_samples = max_samples
        self.seed = seed
        self.subjects = subjects  # None = all
        self._data: List[Dict] = []

    def setup(self, tokenizer, device: str) -> None:
        from datasets import load_dataset
        # MATH-500: the canonical 500-problem evaluation subset of the full
        # Hendrycks MATH dataset, used by every modern LLM benchmark suite.
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

        items = []
        for ex in ds:
            subj = ex.get("subject") or ex.get("type") or "unknown"
            if self.subjects and subj not in self.subjects:
                continue
            items.append({
                "problem": ex.get("problem", ""),
                "answer": ex.get("answer", ""),  # gold answer (string)
                "solution": ex.get("solution", ""),
                "subject": subj,
                "level": ex.get("level", "?"),
            })

        # Stratified sample: evenly across subjects, keeping deterministic
        import random
        rng = random.Random(self.seed)
        by_subject: Dict[str, List[Dict]] = {}
        for it in items:
            by_subject.setdefault(it["subject"], []).append(it)
        per_subject = max(1, self.max_samples // max(1, len(by_subject)))
        sampled: List[Dict] = []
        for subj, group in by_subject.items():
            rng.shuffle(group)
            sampled.extend(group[:per_subject])
        rng.shuffle(sampled)
        self._data = sampled[:self.max_samples]
        self._tokenizer = tokenizer
        self._data_loaded = True
        print(f"  MATH loaded: {len(self._data)} questions "
              f"across {len(by_subject)} subjects")

    @property
    def needs_generation(self) -> bool:
        return True

    def run_unit(
        self,
        unit: ExperimentUnit,
        model, tokenizer, spec: ModelSpec,
        gini_scores, importance_weights, device,
    ) -> Dict[str, Any]:
        correct = 0
        total = 0
        per_subject: Dict[str, Dict[str, int]] = {}

        for idx, item in enumerate(self._data):
            problem = item["problem"]
            # MATH-500 provides a clean `answer` field; fall back to
            # extracting from `solution` if missing.
            gold = item.get("answer") or extract_boxed_answer(item["solution"])
            subj = item["subject"]

            prompt = (
                "Solve the following math problem step by step. "
                "Put your final answer in \\boxed{}.\n\n"
                f"Problem: {problem}\n\nSolution:"
            )

            input_ids = tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=True,
            ).to(device)

            seq_len = input_ids.shape[1]
            prefix_len = max(1, int(seq_len * 0.85))
            suffix_ids = input_ids[:, prefix_len:]

            need_attn = (unit.method_name not in
                         ("full_kv", "layer_budget", "kivi_uniform",
                          "xquant", "streaming_llm", "h2o_uniform", "duo_attention"))
            full_k, full_v, attns = extract_kv_and_attention(
                model, input_ids[:, :prefix_len], need_attention=need_attn,
            )

            layers, _, _ = compress_kv(
                unit.method_name, full_k, full_v, unit.compression_ratio,
                spec.num_layers, spec.num_kv_heads, spec.head_dim,
                attention_weights=attns,
                gini_scores=gini_scores,
                importance_weights=importance_weights,
                model_short_name=spec.short_name,
            )

            cache = build_hf_cache(layers, prefix_len, device, fill="mean")
            output = generate_from_cache(
                model, tokenizer, cache, prefix_len,
                suffix_ids=suffix_ids,
                max_new_tokens=384,
                device=device,
                stop_patterns=["\\boxed{"],  # Early stop once answer is found
            )
            pred = extract_boxed_answer(output)

            ok = is_correct(pred, gold)
            if ok:
                correct += 1
            total += 1
            stats = per_subject.setdefault(subj, {"correct": 0, "total": 0})
            stats["correct"] += int(ok)
            stats["total"] += 1

            del full_k, full_v, attns, layers, cache
            clear_gpu()

            # Early termination if hopelessly broken
            if idx == 49 and correct == 0:
                print(f"    Early stop: 0/{total} after 50 problems")
                break

        accuracy = correct / total if total > 0 else 0.0
        return {
            "task": "math",
            "model": spec.short_name,
            "method": unit.method_name,
            "compression_ratio": unit.compression_ratio,
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 4),
            "per_subject": {
                k: {**v, "accuracy": round(v["correct"] / v["total"], 4)}
                for k, v in per_subject.items() if v["total"] > 0
            },
        }
