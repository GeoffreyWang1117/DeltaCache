"""Needle-In-A-Haystack evaluation.

FIXED: Added per-sample OOM recovery (same as RULER/LongBench fixes).

Inserts a "needle" fact at various depth positions in a long context,
then asks the model to recall it. Produces the classic NIAH heatmap
(context length × needle depth → accuracy).
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


NEEDLES = [
    ("The secret project code name is 'Operation Midnight Sun'.",
     "Operation Midnight Sun", "What is the secret project code name?"),
    ("The combination to the vault is 73-21-84.",
     "73-21-84", "What is the combination to the vault?"),
    ("The best restaurant in San Francisco is 'Golden Phoenix' on Market Street.",
     "Golden Phoenix", "What is the best restaurant in San Francisco?"),
    ("The annual budget for department X is exactly $4,782,319.",
     "$4,782,319", "What is the annual budget for department X?"),
    ("The next team meeting is scheduled for March 15th at 3pm in room B42.",
     "March 15", "When is the next team meeting scheduled?"),
]

HAYSTACK_SENTENCES = [
    "The study of computational complexity provides deep insights into algorithm design.",
    "Natural language processing has seen remarkable advances in recent years.",
    "Database systems form the backbone of modern information management.",
    "Operating systems manage hardware resources for application programs.",
    "Computer networks enable communication between distributed systems.",
    "Software engineering practices improve code quality and maintainability.",
    "Machine learning models learn patterns from data without explicit programming.",
    "Cryptographic protocols ensure secure communication over public networks.",
    "Distributed systems coordinate multiple machines to solve complex problems.",
    "Human-computer interaction research improves usability of technology.",
]


def _build_haystack(target_tokens: int, tokenizer) -> str:
    rng = random.Random(42)
    sentences = []
    est_tokens = 0
    while est_tokens < target_tokens:
        s = rng.choice(HAYSTACK_SENTENCES)
        sentences.append(s)
        est_tokens += len(s.split()) * 1.3
    text = " ".join(sentences)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens)


class NIAHTask(BaseTask):
    name = "niah"

    def __init__(self, n_depths: int = 10, n_repeats: int = 3):
        super().__init__()
        self.n_depths = n_depths
        self.n_repeats = n_repeats
        self.depths = [i / (n_depths - 1) for i in range(n_depths)]

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

        haystack_budget = target_len - 100
        haystack_text = _build_haystack(haystack_budget, tokenizer)
        haystack_tokens = tokenizer.encode(
            haystack_text, add_special_tokens=False)

        depth_results = {}
        oom_total = 0

        for depth in self.depths:
            correct = 0
            total = 0

            for rep in range(self.n_repeats):
                needle_fact, needle_answer, question = NEEDLES[
                    rep % len(NEEDLES)]

                insert_pos = int(len(haystack_tokens) * depth)
                needle_tokens = tokenizer.encode(
                    f" {needle_fact} ", add_special_tokens=False)

                combined = (haystack_tokens[:insert_pos]
                            + needle_tokens
                            + haystack_tokens[insert_pos:])

                context = tokenizer.decode(combined[:haystack_budget])
                prompt = (f"Read the following text carefully.\n\n"
                          f"{context}\n\n"
                          f"Based on the text above, answer: {question}\n"
                          f"Answer:")

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
                                  "streaming_llm", "duo_attention",
                                  "h2o_uniform"))
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

                    if needle_answer.lower() in output.lower():
                        correct += 1
                    total += 1

                except torch.cuda.OutOfMemoryError:
                    oom_total += 1
                    clear_gpu()
                    if oom_total >= 5:
                        break
                    continue
                except Exception:
                    clear_gpu()
                    continue
                finally:
                    # Drop the references before clear_gpu(), which starts with gc.collect():
                    # the collector can only reclaim what nothing points at any more.
                    # This was exec("del <name>") in a loop, which cannot work. exec gets a
                    # copy of the function's locals, so the del applied to the copy and every
                    # tensor stayed alive until the frame exited. Rebinding does drop them,
                    # and unlike del it is safe when a name was never assigned.
                    full_k = full_v = attns = layers = cache = None
                    clear_gpu()

            if oom_total >= 5:
                break

            if total > 0:
                depth_key = "%.1f%%" % (depth * 100)
                depth_results[depth_key] = {
                    "depth": depth,
                    "correct": correct,
                    "total": total,
                    "accuracy": round(correct / total, 4),
                }

        all_acc = [v["accuracy"] for v in depth_results.values()]
        return {
            "task": "niah",
            "model": spec.short_name,
            "method": unit.method_name,
            "compression_ratio": unit.compression_ratio,
            "seq_len": target_len,
            "overall_accuracy": round(statistics.mean(all_acc), 4) if all_acc else 0,
            "per_depth": depth_results,
            "oom_skipped": oom_total,
        }
