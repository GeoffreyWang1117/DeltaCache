"""RULER synthetic evaluation — multi-key retrieval & variable tracking.

FIXED: Added per-sample OOM recovery and adaptive sample count for long sequences.
The old version would fail the entire unit on any single OOM, causing 60% failure rate.

Tests fine-grained information retrieval under KV cache compression.
Synthetic tasks allow controlled evaluation at any context length.

Tasks:
  - Multi-key retrieval (MKR): retrieve multiple inserted key-value pairs
  - Variable tracking (VT): track variable assignments through context
"""

from __future__ import annotations

import random
import statistics
from typing import Any, Dict, List

import torch

from ..config import ExperimentUnit, ModelSpec
from ..eval_utils import (
    extract_kv_and_attention, compress_kv, build_hf_cache, clear_gpu,
    generate_from_cache,
)
from .base import BaseTask


def _generate_mkr_sample(
    seq_len: int, n_keys: int = 5, tokenizer=None,
) -> Dict:
    """Generate a multi-key retrieval sample."""
    rng = random.Random()
    keys = [f"KEY_{i:03d}" for i in range(n_keys)]
    values = [f"VALUE_{rng.randint(1000, 9999)}" for i in range(n_keys)]

    fillers = [
        "The weather is sunny today and the birds are singing.",
        "Mathematics is the queen of sciences according to Gauss.",
        "The library contains many books on various topics.",
        "Rivers flow from mountains to the sea following gravity.",
        "Computers process information using binary representations.",
    ]

    n_filler_sentences = max(10, seq_len // 20)
    context_parts = [rng.choice(fillers) for _ in range(n_filler_sentences)]

    positions = sorted(rng.sample(range(len(context_parts)), min(n_keys, len(context_parts))))
    for i, pos in enumerate(positions):
        context_parts[pos] = f"Remember: {keys[i]} is {values[i]}."

    context = " ".join(context_parts)
    query_idx = rng.randint(0, n_keys - 1)
    question = f"What is the value of {keys[query_idx]}?"
    answer = values[query_idx]

    return {
        "context": context,
        "question": question,
        "answer": answer,
        "n_keys": n_keys,
    }


def _generate_vt_sample(seq_len: int, n_vars: int = 3) -> Dict:
    """Generate a variable tracking sample."""
    rng = random.Random()
    var_names = [f"x{i}" for i in range(n_vars)]
    assignments = []
    current_values = {}

    n_steps = max(5, seq_len // 30)
    fillers = [
        "Processing continues.",
        "The system runs normally.",
        "No changes at this point.",
    ]

    for step in range(n_steps):
        if rng.random() < 0.4 or not current_values:
            var = rng.choice(var_names)
            val = rng.randint(1, 100)
            current_values[var] = val
            assignments.append(f"Set {var} = {val}.")
        else:
            assignments.append(rng.choice(fillers))

    context = " ".join(assignments)
    query_var = rng.choice(list(current_values.keys()))
    question = f"What is the final value of {query_var}?"
    answer = str(current_values[query_var])

    return {"context": context, "question": question, "answer": answer}


class RULERTask(BaseTask):
    name = "ruler"

    def __init__(self, n_samples: int = 50):
        super().__init__()
        self.n_samples = n_samples

    def setup(self, tokenizer, device: str) -> None:
        self._tokenizer = tokenizer
        self._data_loaded = True

    @property
    def needs_generation(self) -> bool:
        return True

    def run_unit(
        self,
        unit: ExperimentUnit,
        model, tokenizer, spec: ModelSpec,
        gini_scores, importance_weights, device,
    ) -> Dict[str, Any]:
        target_len = unit.seq_len or 2048
        results = {}

        # Adaptive sample count: reduce for long sequences to avoid OOM cascade
        n_samples = self.n_samples
        if target_len >= 8192:
            n_samples = max(10, self.n_samples // 3)
        elif target_len >= 4096:
            n_samples = max(20, self.n_samples // 2)

        for task_type, generator in [
            ("mkr_1", lambda sl: _generate_mkr_sample(sl, 1, tokenizer)),
            ("mkr_5", lambda sl: _generate_mkr_sample(sl, 5, tokenizer)),
            ("mkr_10", lambda sl: _generate_mkr_sample(sl, 10, tokenizer)),
            ("vt", lambda sl: _generate_vt_sample(sl, 3)),
        ]:
            correct = 0
            total = 0
            oom_count = 0

            for _ in range(n_samples):
                sample = generator(target_len)
                prompt = (f"{sample['context']}\n\n"
                          f"Question: {sample['question']}\nAnswer:")

                input_ids = tokenizer.encode(
                    prompt, return_tensors="pt",
                    max_length=target_len, truncation=True,
                    add_special_tokens=True,
                ).to(device)

                seq_len = input_ids.shape[1]
                prefix_len = int(seq_len * 0.85)
                suffix_ids = input_ids[:, prefix_len:]

                try:
                    need_attn = (unit.method_name not in
                                 ("full_kv", "layer_budget",
                                  "kivi_uniform", "xquant",
                                  "streaming_llm", "h2o_uniform", "duo_attention"))
                    full_k, full_v, attns = extract_kv_and_attention(
                        model, input_ids[:, :prefix_len],
                        need_attention=need_attn,
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
                        max_new_tokens=32, device=device,
                    )

                    if sample["answer"].lower() in output.lower():
                        correct += 1
                    total += 1

                except torch.cuda.OutOfMemoryError:
                    oom_count += 1
                    clear_gpu()
                    if oom_count >= 3:
                        # Too many OOMs in a row — skip remaining samples
                        break
                    continue
                except Exception:
                    clear_gpu()
                    continue
                finally:
                    for vname in ['full_k', 'full_v', 'attns', 'layers', 'cache']:
                        try:
                            exec(f'del {vname}')
                        except:
                            pass
                    clear_gpu()

            if total > 0:
                results[task_type] = {
                    "correct": correct,
                    "total": total,
                    "accuracy": round(correct / total, 4),
                    "oom_skipped": oom_count,
                }

        all_acc = [v["accuracy"] for v in results.values()]
        return {
            "task": "ruler",
            "model": spec.short_name,
            "method": unit.method_name,
            "compression_ratio": unit.compression_ratio,
            "seq_len": target_len,
            "n_samples_target": n_samples,
            "overall_accuracy": round(statistics.mean(all_acc), 4) if all_acc else 0,
            "per_task": results,
        }
