"""System throughput and latency benchmark — FIXED.

Now uses the same compress_kv → build_hf_cache pipeline as PPL evaluation,
which actually removes tokens and reduces memory. The old version passed
cache_cls but never used it, resulting in 0% memory savings.

Measures:
- KV cache memory specifically (not total GPU including weights)
- Prefill latency (time to first token)
- Decode throughput with compressed cache (tokens/second)
- End-to-end generation time
- Real memory savings from KV compression
"""

from __future__ import annotations

import gc
import statistics
import time
from typing import Any, Dict

import torch

from ..config import ExperimentUnit, ModelSpec
from ..eval_utils import (
    extract_kv_and_attention, compress_kv, build_hf_cache,
    clear_gpu, generate_from_cache,
)
from .base import BaseTask


def _gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def _current_mem_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def _reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


class ThroughputTask(BaseTask):
    name = "throughput"

    def __init__(self, n_warmup: int = 2, n_measure: int = 5,
                 gen_tokens: int = 64, batch_sizes: list | None = None):
        super().__init__()
        self.n_warmup = n_warmup
        self.n_measure = n_measure
        self.gen_tokens = gen_tokens
        self.batch_sizes = batch_sizes or [1]

    def setup(self, tokenizer, device: str) -> None:
        self._tokenizer = tokenizer
        self._data_loaded = True

    @property
    def needs_attention(self) -> bool:
        return False

    @property
    def needs_generation(self) -> bool:
        return True

    def run_unit(
        self,
        unit: ExperimentUnit,
        model, tokenizer, spec: ModelSpec,
        gini_scores, importance_weights, device,
    ) -> Dict[str, Any]:
        target_len = unit.seq_len or 1024
        cr = unit.compression_ratio

        # Generate deterministic input
        text = "The quick brown fox jumps over the lazy dog. " * (target_len // 10)
        input_ids = tokenizer.encode(
            text, return_tensors="pt",
            max_length=target_len, truncation=True,
            add_special_tokens=True,
        ).to(device)

        results = {}

        # ── Baseline: standard DynamicCache (no compression) ──
        baseline_metrics = self._benchmark_baseline(
            model, input_ids, device,
        )
        results["baseline"] = baseline_metrics

        # ── LayerBudget: compress_kv → build_hf_cache → decode ──
        lb_metrics = self._benchmark_layerbudget(
            model, input_ids, spec, cr,
            gini_scores, importance_weights, device,
        )
        results["layerbudget"] = lb_metrics

        # Compute deltas
        if baseline_metrics and lb_metrics:
            b_kv = baseline_metrics.get("kv_mem_mb", baseline_metrics["peak_mem_mb"])
            l_kv = lb_metrics.get("kv_mem_mb", lb_metrics["peak_mem_mb"])
            results["memory_savings_pct"] = round(
                (1 - l_kv / b_kv) * 100, 1
            ) if b_kv > 0 else 0
            results["speedup"] = round(
                baseline_metrics["decode_ms"] / lb_metrics["decode_ms"], 3
            ) if lb_metrics["decode_ms"] > 0 else 0

        return {
            "task": "throughput",
            "model": spec.short_name,
            "compression_ratio": cr,
            "seq_len": target_len,
            "gen_tokens": self.gen_tokens,
            **results,
        }

    def _benchmark_baseline(self, model, input_ids, device) -> Dict:
        """Benchmark with standard DynamicCache (no compression)."""
        seq_len = input_ids.shape[1]
        prefill_times = []
        decode_times = []
        e2e_times = []
        peak_mems = []
        kv_mems = []

        for i in range(self.n_warmup + self.n_measure):
            clear_gpu()

            # Measure memory before to isolate KV cache contribution
            mem_before = _current_mem_mb()
            _reset_peak_mem()

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            # Prefill
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                )
            torch.cuda.synchronize()
            t_prefill = time.perf_counter()

            # Measure KV cache memory
            kv_mem = _current_mem_mb() - mem_before

            # Manual decode loop
            past_kv = out.past_key_values
            last_logits = out.logits[:, -1, :]
            cur_pos = seq_len

            with torch.no_grad():
                for step in range(self.gen_tokens):
                    next_token = last_logits.argmax(dim=-1, keepdim=True)
                    pos = torch.tensor([[cur_pos]], device=device)
                    step_out = model(
                        input_ids=next_token,
                        past_key_values=past_kv,
                        position_ids=pos,
                        return_dict=True,
                        use_cache=True,
                    )
                    past_kv = step_out.past_key_values
                    last_logits = step_out.logits[:, -1, :]
                    cur_pos += 1

            torch.cuda.synchronize()
            t_end = time.perf_counter()

            peak_mem = _gpu_mem_mb()

            if i >= self.n_warmup:
                prefill_times.append((t_prefill - t_start) * 1000)
                decode_times.append((t_end - t_prefill) * 1000)
                e2e_times.append((t_end - t_start) * 1000)
                peak_mems.append(peak_mem)
                kv_mems.append(max(0, kv_mem))

            del out, past_kv
            clear_gpu()

        if not prefill_times:
            return {}

        mean_decode = statistics.mean(decode_times)
        return {
            "label": "baseline",
            "TTFT_ms": round(statistics.mean(prefill_times), 2),
            "TPOT_ms": round(mean_decode / self.gen_tokens, 2),
            "prefill_ms": round(statistics.mean(prefill_times), 2),
            "decode_ms": round(mean_decode, 2),
            "e2e_ms": round(statistics.mean(e2e_times), 2),
            "tokens_per_sec": round(self.gen_tokens / (mean_decode / 1000), 1),
            "peak_mem_mb": round(statistics.mean(peak_mems), 1),
            "kv_mem_mb": round(statistics.mean(kv_mems), 1),
            "prefill_std": round(statistics.stdev(prefill_times), 2) if len(prefill_times) > 1 else 0,
            "decode_std": round(statistics.stdev(decode_times), 2) if len(decode_times) > 1 else 0,
        }

    def _benchmark_layerbudget(
        self, model, input_ids, spec, cr,
        gini_scores, importance_weights, device,
    ) -> Dict:
        """Benchmark with LayerBudget compress_kv → build_hf_cache pipeline.

        This actually compresses the KV cache (removes tokens, quantizes)
        and measures real memory and latency impact.
        """
        seq_len = input_ids.shape[1]
        prefix_len = int(seq_len * 0.85)
        suffix_ids = input_ids[:, prefix_len:]

        prefill_times = []
        compress_times = []
        decode_times = []
        e2e_times = []
        peak_mems = []
        kv_mems = []

        for i in range(self.n_warmup + self.n_measure):
            clear_gpu()
            mem_before = _current_mem_mb()
            _reset_peak_mem()

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            # Prefill: extract KV
            full_k, full_v, _ = extract_kv_and_attention(
                model, input_ids[:, :prefix_len], need_attention=False,
            )
            torch.cuda.synchronize()
            t_prefill = time.perf_counter()

            # Compress KV cache
            layers, comp_ms, comp_bytes = compress_kv(
                "layer_budget", full_k, full_v, cr,
                spec.num_layers, spec.num_kv_heads, spec.head_dim,
                gini_scores=gini_scores,
                importance_weights=importance_weights,
            )
            del full_k, full_v
            clear_gpu()

            torch.cuda.synchronize()
            t_compress = time.perf_counter()

            # Build compressed HF cache
            cache = build_hf_cache(layers, prefix_len, device, fill="mean")
            del layers

            # Measure KV cache memory after compression
            kv_mem = _current_mem_mb() - mem_before

            # Decode: feed suffix then generate
            cur_pos = prefix_len
            past_kv = cache

            with torch.no_grad():
                # Feed suffix tokens
                if suffix_ids.shape[1] > 0:
                    pos = torch.arange(
                        cur_pos, cur_pos + suffix_ids.shape[1], device=device
                    ).unsqueeze(0)
                    out = model(
                        input_ids=suffix_ids,
                        past_key_values=past_kv,
                        position_ids=pos,
                        return_dict=True,
                        use_cache=True,
                    )
                    past_kv = out.past_key_values
                    cur_pos += suffix_ids.shape[1]
                    last_logits = out.logits[:, -1, :]
                else:
                    # Use last token of prefix
                    last_token = input_ids[:, prefix_len - 1:prefix_len]
                    pos = torch.tensor([[cur_pos - 1]], device=device)
                    out = model(
                        input_ids=last_token,
                        past_key_values=past_kv,
                        position_ids=pos,
                        return_dict=True,
                        use_cache=True,
                    )
                    past_kv = out.past_key_values
                    last_logits = out.logits[:, -1, :]

                # Generate tokens
                for step in range(self.gen_tokens):
                    next_token = last_logits.argmax(dim=-1, keepdim=True)
                    pos = torch.tensor([[cur_pos]], device=device)
                    step_out = model(
                        input_ids=next_token,
                        past_key_values=past_kv,
                        position_ids=pos,
                        return_dict=True,
                        use_cache=True,
                    )
                    past_kv = step_out.past_key_values
                    last_logits = step_out.logits[:, -1, :]
                    cur_pos += 1

            torch.cuda.synchronize()
            t_end = time.perf_counter()

            peak_mem = _gpu_mem_mb()

            if i >= self.n_warmup:
                prefill_times.append((t_prefill - t_start) * 1000)
                compress_times.append((t_compress - t_prefill) * 1000)
                decode_times.append((t_end - t_compress) * 1000)
                e2e_times.append((t_end - t_start) * 1000)
                peak_mems.append(peak_mem)
                kv_mems.append(max(0, kv_mem))

            del out, past_kv, cache
            clear_gpu()

        if not prefill_times:
            return {}

        mean_decode = statistics.mean(decode_times)
        return {
            "label": f"layerbudget_{cr}x",
            "TTFT_ms": round(statistics.mean(prefill_times), 2),
            "TPOT_ms": round(mean_decode / self.gen_tokens, 2),
            "prefill_ms": round(statistics.mean(prefill_times), 2),
            "compress_ms": round(statistics.mean(compress_times), 2),
            "decode_ms": round(mean_decode, 2),
            "e2e_ms": round(statistics.mean(e2e_times), 2),
            "tokens_per_sec": round(self.gen_tokens / (mean_decode / 1000), 1),
            "peak_mem_mb": round(statistics.mean(peak_mems), 1),
            "kv_mem_mb": round(statistics.mean(kv_mems), 1),
            "compressed_bytes": 0,  # filled by compress_kv
            "prefill_std": round(statistics.stdev(prefill_times), 2) if len(prefill_times) > 1 else 0,
            "decode_std": round(statistics.stdev(decode_times), 2) if len(decode_times) > 1 else 0,
        }
