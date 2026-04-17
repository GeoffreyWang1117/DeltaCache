"""PPL evaluation task — main quality metric (Table 1 in paper).

Extended from original: supports 512-8192 tokens, all 16 baselines,
multiple runs for error bars.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List

import torch

from ..config import ExperimentUnit, ModelSpec
from ..eval_utils import (
    extract_kv_and_attention, compress_kv, build_hf_cache,
    compute_ppl_from_cache, clear_gpu,
)
from .base import BaseTask


class PPLTask(BaseTask):
    name = "ppl"

    def __init__(self, n_chunks: int = 6, prefix_ratio: float = 0.6,
                 n_runs: int = 1):
        super().__init__()
        self.n_chunks = n_chunks
        self.prefix_ratio = prefix_ratio
        self.n_runs = n_runs
        self._chunks_cache: Dict[int, List] = {}  # seq_len -> chunks
        self._full_kv_ppl: Dict[int, float] = {}  # seq_len -> mean full_kv ppl
        # KV extraction cache: avoid redundant forward passes when
        # multiple methods run at the same (seq_len, chunk_idx).
        # Key: (seq_len, chunk_idx, need_attn)
        # Value: (full_k, full_v, attns)
        self._kv_cache: Dict = {}
        self._kv_cache_seq_len: int = -1  # current cached seq_len

    def setup(self, tokenizer, device: str) -> None:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        self._full_text = " ".join(x["text"] for x in ds if x["text"].strip())
        self._tokens = tokenizer.encode(self._full_text, add_special_tokens=False)
        self._tokenizer = tokenizer
        self._chunks_cache.clear()
        self._full_kv_ppl.clear()  # reset per-model cache
        self._data_loaded = True

    def _get_chunks(self, seq_len: int) -> List[List[int]]:
        """Get n_chunks of seq_len tokens, evenly spaced through the text."""
        if seq_len in self._chunks_cache:
            return self._chunks_cache[seq_len]

        total = len(self._tokens)
        if total < seq_len * self.n_chunks:
            # Not enough text: use overlapping chunks
            stride = max(1, (total - seq_len) // max(1, self.n_chunks - 1))
            chunks = [self._tokens[i * stride: i * stride + seq_len]
                      for i in range(self.n_chunks)]
        else:
            stride = total // self.n_chunks
            chunks = [self._tokens[i * stride: i * stride + seq_len]
                      for i in range(self.n_chunks)]

        self._chunks_cache[seq_len] = chunks
        return chunks

    def run_unit(
        self,
        unit: ExperimentUnit,
        model, tokenizer, spec: ModelSpec,
        gini_scores, importance_weights, device,
    ) -> Dict[str, Any]:
        seq_len = unit.seq_len or 1024
        chunks = self._get_chunks(seq_len)
        prefix_len = int(seq_len * self.prefix_ratio)

        ppls = []
        compress_times = []
        memories = []

        # Invalidate KV cache when seq_len changes
        if seq_len != self._kv_cache_seq_len:
            self._kv_cache.clear()
            self._kv_cache_seq_len = seq_len

        for chunk_idx, chunk_tokens in enumerate(chunks):
            if len(chunk_tokens) < seq_len:
                continue

            input_ids = torch.tensor(
                [chunk_tokens[:seq_len]], device=device)

            # Prefill: extract KV — cached across methods at same seq_len.
            # Only cache (full_k, full_v), NOT attention tensors (too large
            # at long seq: 32 heads × S² × 2B × 32 layers = 34GB @ 4096tok).
            need_attn = (unit.method_name not in
                         ("full_kv", "layer_budget", "kivi_uniform", "xquant",
                          "streaming_llm", "h2o_uniform", "duo_attention"))
            kv_key = (seq_len, chunk_idx)
            if kv_key in self._kv_cache:
                full_k, full_v = self._kv_cache[kv_key]
                # Re-extract attention only if needed (cheap: ~1 fwd pass)
                if need_attn:
                    _, _, attns = extract_kv_and_attention(
                        model, input_ids[:, :prefix_len], need_attention=True,
                    )
                else:
                    attns = None
            else:
                full_k, full_v, attns = extract_kv_and_attention(
                    model, input_ids[:, :prefix_len], need_attention=need_attn,
                )
                self._kv_cache[kv_key] = (full_k, full_v)

            # Compress
            layers, comp_ms, mem_bytes = compress_kv(
                unit.method_name, full_k, full_v, unit.compression_ratio,
                spec.num_layers, spec.num_kv_heads, spec.head_dim,
                attention_weights=attns,
                gini_scores=gini_scores,
                importance_weights=importance_weights,
                model_short_name=spec.short_name,
            )
            compress_times.append(comp_ms)
            memories.append(mem_bytes)

            # Build cache and evaluate PPL
            cache = build_hf_cache(layers, prefix_len, device, fill="mean")
            ppl = compute_ppl_from_cache(model, input_ids, cache, prefix_len)
            ppls.append(ppl)

            # Free compression results (KV is cached for next method)
            del layers, cache
            clear_gpu()

        if not ppls:
            return {"error": "no valid chunks"}

        mean_ppl = statistics.mean(ppls)

        # Get full-KV PPL for ratio computation (cached across methods)
        full_ppl = None
        if unit.method_name == "full_kv":
            # Cache this result for other methods
            self._full_kv_ppl[seq_len] = mean_ppl
        else:
            full_ppl = self._full_kv_ppl.get(seq_len)
            if full_ppl is None:
                # Compute full-KV PPL and cache it
                full_ppls = []
                for chunk_tokens in chunks:
                    if len(chunk_tokens) < seq_len:
                        continue
                    input_ids = torch.tensor(
                        [chunk_tokens[:seq_len]], device=device)
                    full_k, full_v, _ = extract_kv_and_attention(
                        model, input_ids[:, :prefix_len],
                        need_attention=False)
                    all_idx = torch.arange(prefix_len)
                    full_layers = [(full_k[l:l+1], full_v[l:l+1], all_idx)
                                   for l in range(spec.num_layers)]
                    full_cache = build_hf_cache(
                        full_layers, prefix_len, device, fill="mean")
                    fp = compute_ppl_from_cache(
                        model, input_ids, full_cache, prefix_len)
                    full_ppls.append(fp)
                    del full_k, full_v, full_layers, full_cache
                    clear_gpu()
                full_ppl = statistics.mean(full_ppls) if full_ppls else None
                if full_ppl is not None:
                    self._full_kv_ppl[seq_len] = full_ppl

        ppl_ratio = mean_ppl / full_ppl if full_ppl and full_ppl > 0 else None

        return {
            "task": "ppl",
            "model": spec.short_name,
            "method": unit.method_name,
            "compression_ratio": unit.compression_ratio,
            "seq_len": seq_len,
            "prefix_len": prefix_len,
            "n_chunks": len(ppls),
            "mean_ppl": round(mean_ppl, 4),
            "std_ppl": round(statistics.stdev(ppls), 4) if len(ppls) > 1 else 0,
            "ppl_ratio": round(ppl_ratio, 4) if ppl_ratio else None,
            "full_ppl": round(full_ppl, 4) if full_ppl else None,
            "mean_compress_ms": round(statistics.mean(compress_times), 2),
            "mean_memory_bytes": int(statistics.mean(memories)),
        }
