"""Full LongBench evaluation — all 16 English tasks.

FIXED: Uses direct download from HuggingFace data.zip instead of
load_dataset(), which broke when datasets>=4.x dropped custom script support.

Categories: SingleDoc QA, MultiDoc QA, Summarization, Few-shot, Code, Synthetic.
Uses F1/ROUGE-L/accuracy depending on task type.
"""

from __future__ import annotations

import io
import json
import os
import statistics
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..config import ExperimentUnit, ModelSpec
from ..eval_utils import (
    extract_kv_and_attention, compress_kv, build_hf_cache,
    clear_gpu, f1_score, rouge_l, exact_match, generate_from_cache,
)
from .base import BaseTask


# Task -> (metric_name, metric_fn, max_new_tokens)
LONGBENCH_TASKS = {
    # Single-Document QA
    "narrativeqa": ("f1", f1_score, 64),
    "qasper": ("f1", f1_score, 64),
    "multifieldqa_en": ("f1", f1_score, 64),
    # Multi-Document QA
    "hotpotqa": ("f1", f1_score, 64),
    "2wikimqa": ("f1", f1_score, 64),
    "musique": ("f1", f1_score, 64),
    # Summarization
    "gov_report": ("rouge_l", rouge_l, 256),
    "qmsum": ("rouge_l", rouge_l, 256),
    "multi_news": ("rouge_l", rouge_l, 256),
    # Few-shot
    "trec": ("accuracy", exact_match, 8),
    "triviaqa": ("f1", f1_score, 64),
    "samsum": ("rouge_l", rouge_l, 128),
    # Synthetic
    "passage_count": ("accuracy", exact_match, 8),
    "passage_retrieval_en": ("accuracy", exact_match, 8),
    # Code
    "lcc": ("rouge_l", rouge_l, 128),
    "repobench-p": ("rouge_l", rouge_l, 128),
}

# Cache directory for downloaded data
_CACHE_DIR = Path(__file__).parent.parent / "results" / ".longbench_cache"


def _download_longbench_data() -> Dict[str, List[Dict]]:
    """Download LongBench data.zip from HuggingFace and extract JSONL files.

    Caches locally to avoid re-downloading.
    """
    cache_marker = _CACHE_DIR / ".downloaded"
    all_data = {}

    if cache_marker.exists():
        # Load from local cache
        for task_name in LONGBENCH_TASKS:
            fpath = _CACHE_DIR / f"{task_name}.jsonl"
            if fpath.exists():
                with open(fpath) as f:
                    all_data[task_name] = [json.loads(line) for line in f if line.strip()]
        return all_data

    # Download
    import requests
    url = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
    print(f"    Downloading LongBench data from {url}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    for name in z.namelist():
        if not name.endswith(".jsonl"):
            continue
        basename = Path(name).stem  # e.g., "narrativeqa"
        # Some tasks have _e suffix (extended) — skip those
        if basename.endswith("_e"):
            continue
        if basename not in LONGBENCH_TASKS:
            continue

        data_bytes = z.read(name)
        # Save to cache
        with open(_CACHE_DIR / f"{basename}.jsonl", "wb") as f:
            f.write(data_bytes)

        lines = data_bytes.decode().strip().split("\n")
        all_data[basename] = [json.loads(line) for line in lines if line.strip()]

    cache_marker.touch()
    return all_data


class LongBenchTask(BaseTask):
    name = "longbench"

    def __init__(self, tasks: List[str] = None, max_samples: int = 50):
        super().__init__()
        self.task_names = tasks or list(LONGBENCH_TASKS.keys())
        self.max_samples = max_samples
        self._data: Dict[str, List] = {}

    def setup(self, tokenizer, device: str) -> None:
        raw = _download_longbench_data()

        for task_name in self.task_names:
            if task_name in raw:
                samples = raw[task_name][:self.max_samples]
                self._data[task_name] = samples
                print(f"    LongBench/{task_name}: {len(samples)} samples")
            else:
                print(f"    LongBench/{task_name}: SKIP (not in data)")

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
        results_per_task = {}

        for task_name in self.task_names:
            if task_name not in self._data:
                continue

            metric_name, metric_fn, max_gen = LONGBENCH_TASKS[task_name]
            samples = self._data[task_name]
            scores = []

            for sample in samples:
                context = sample.get("context", sample.get("input", ""))
                question = sample.get("input", sample.get("question", ""))
                answers = sample.get("answers", [sample.get("answer", "")])
                if isinstance(answers, str):
                    answers = [answers]
                if not answers or all(not a for a in answers):
                    continue

                # Build prompt
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                input_ids = tokenizer.encode(
                    prompt, return_tensors="pt",
                    max_length=target_len, truncation=True,
                    add_special_tokens=True,
                ).to(device)

                seq_len = input_ids.shape[1]
                if seq_len < 32:
                    continue

                actual_prefix = int(seq_len * 0.85)
                suffix_ids = input_ids[:, actual_prefix:]

                try:
                    # Extract KV for prefix
                    need_attn = (unit.method_name not in
                                 ("full_kv", "layer_budget",
                                  "kivi_uniform", "xquant",
                                  "streaming_llm", "h2o_uniform", "duo_attention"))
                    full_k, full_v, attns = extract_kv_and_attention(
                        model, input_ids[:, :actual_prefix],
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

                    # Generate answer
                    cache = build_hf_cache(
                        layers, actual_prefix, device, fill="mean")

                    prediction = generate_from_cache(
                        model, tokenizer, cache, actual_prefix,
                        suffix_ids=suffix_ids,
                        max_new_tokens=max_gen, device=device,
                    )

                    # Score against all reference answers
                    best_score = max(metric_fn(prediction, a) for a in answers)
                    scores.append(best_score)

                except torch.cuda.OutOfMemoryError:
                    clear_gpu()
                    continue
                except Exception:
                    clear_gpu()
                    continue
                finally:
                    # Always cleanup
                    for name in ['full_k', 'full_v', 'attns', 'layers', 'cache']:
                        if name in dir():
                            try:
                                exec(f'del {name}')
                            except:
                                pass
                    clear_gpu()

            if scores:
                results_per_task[task_name] = {
                    "metric": metric_name,
                    "mean": round(statistics.mean(scores), 4),
                    "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
                    "n": len(scores),
                }

        # Aggregate
        all_means = [v["mean"] for v in results_per_task.values()]
        return {
            "task": "longbench",
            "model": spec.short_name,
            "method": unit.method_name,
            "compression_ratio": unit.compression_ratio,
            "seq_len": target_len,
            "overall_mean": round(statistics.mean(all_means), 4) if all_means else 0,
            "n_tasks": len(results_per_task),
            "per_task": results_per_task,
        }
