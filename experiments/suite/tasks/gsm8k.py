"""GSM8K math reasoning evaluation.

Tests whether KV cache compression preserves chain-of-thought reasoning.
Uses exact-match on the final numerical answer (after "####").

Optimizations:
- Only decode the final number, not the full chain-of-thought
- Limit max_new_tokens to 512 (most GSM8K solutions are < 300 tokens)
- Early termination if accuracy = 0 after 100 questions
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List

import torch

from ..config import ExperimentUnit, ModelSpec
from ..eval_utils import (
    extract_kv_and_attention, compress_kv, build_hf_cache,
    clear_gpu, generate_from_cache,
)
from .base import BaseTask


def extract_gsm8k_answer(text: str) -> str:
    """Extract numerical answer from GSM8K format (after ####)."""
    # Model output: look for "#### number" pattern
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    # Fallback: last number in the text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""


def extract_gold_answer(text: str) -> str:
    """Extract gold answer from GSM8K dataset."""
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    return text.strip()


class GSM8KTask(BaseTask):
    name = "gsm8k"

    def __init__(self, max_samples: int = 500, seed: int = 42):
        super().__init__()
        self.max_samples = max_samples
        self.seed = seed
        self._data: List[Dict] = []

    def setup(self, tokenizer, device: str) -> None:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        self._data = list(ds)[:self.max_samples]
        self._tokenizer = tokenizer
        self._data_loaded = True
        print(f"  GSM8K loaded: {len(self._data)} questions")

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
        errors = []

        for idx, item in enumerate(self._data):
            question = item["question"]
            gold = extract_gold_answer(item["answer"])

            prompt = (f"Solve this math problem step by step. "
                      f"End with 'The answer is #### [number]'.\n\n"
                      f"Question: {question}\n\nSolution:")

            input_ids = tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=True,
            ).to(device)

            seq_len = input_ids.shape[1]
            prefix_len = int(seq_len * 0.85)
            suffix_ids = input_ids[:, prefix_len:]

            # Extract KV
            need_attn = (unit.method_name not in
                         ("full_kv", "layer_budget",
                          "kivi_uniform", "xquant", "streaming_llm", "h2o_uniform", "duo_attention"))
            full_k, full_v, attns = extract_kv_and_attention(
                model, input_ids[:, :prefix_len],
                need_attention=need_attn,
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

            # Generate
            cache = build_hf_cache(layers, prefix_len, device, fill="mean")
            output = generate_from_cache(
                model, tokenizer, cache, prefix_len,
                suffix_ids=suffix_ids,
                max_new_tokens=512, device=device,
                stop_patterns=["####"],  # Early stop once answer marker found
            )
            pred = extract_gsm8k_answer(output)

            if pred == gold:
                correct += 1
            total += 1

            del full_k, full_v, attns, layers, cache
            clear_gpu()

            # Early termination
            if idx == 99 and correct == 0:
                print(f"    Early stop: 0/{total} after 100 questions")
                break

        accuracy = correct / total if total > 0 else 0
        return {
            "task": "gsm8k",
            "model": spec.short_name,
            "method": unit.method_name,
            "compression_ratio": unit.compression_ratio,
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 4),
        }
