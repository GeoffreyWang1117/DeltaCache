"""Shared evaluation utilities for all tasks.

Contains the core compress-and-evaluate pipeline shared across tasks,
plus metric functions and efficient KV cache handling.
"""

from __future__ import annotations

import gc
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf
from baselines import REGISTRY

from .model_pool import clear_gpu


# ── KV Cache Operations ────────────────────────────────────────────

def extract_kv_and_attention(
    model, input_ids: Tensor, need_attention: bool = True,
) -> Tuple[Tensor, Tensor, Optional[List[Tensor]]]:
    """Run prefill and extract KV cache + optional attention weights.

    Uses hook-based profiling for memory efficiency when possible.

    Returns:
        full_keys: (num_layers, seq_len, num_kv_heads, head_dim)
        full_values: same shape
        attention_weights: list of per-layer attention tensors, or None
    """
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            output_attentions=need_attention,
            use_cache=True,
            return_dict=True,
        )

    full_k, full_v = hf_to_deltacache(out.past_key_values)
    attns = list(out.attentions) if need_attention and out.attentions else None

    # Free model outputs except what we need
    del out
    clear_gpu()

    return full_k, full_v, attns


def compress_kv(
    method_name: str,
    full_k: Tensor,
    full_v: Tensor,
    compression_ratio: float,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    attention_weights: Optional[List[Tensor]] = None,
    gini_scores: Optional[Dict[int, float]] = None,
    importance_weights: Optional[Dict[int, float]] = None,
    model_short_name: Optional[str] = None,
) -> Tuple[List[Tuple[Tensor, Tensor, Tensor]], float, int]:
    """Compress KV cache using the specified method.

    Returns:
        layers: list of (keys, values, indices) per layer
        compress_time_ms: compression time
        memory_bytes: actual compressed memory
    """
    seq_len = full_k.shape[1]

    if method_name == "full_kv":
        all_idx = torch.arange(seq_len)
        layers = [(full_k[l:l+1], full_v[l:l+1], all_idx)
                  for l in range(num_layers)]
        mem = 2 * full_k.numel() * full_k.element_size()
        return layers, 0.0, mem

    if method_name == "layer_budget":
        return _compress_layer_budget(
            full_k, full_v, compression_ratio, num_layers,
            num_kv_heads, head_dim, gini_scores, importance_weights,
        )

    # Use baseline registry
    if method_name not in REGISTRY:
        raise ValueError(f"Unknown method: {method_name}. "
                         f"Available: {list(REGISTRY.keys())}")

    baseline_cls = REGISTRY[method_name]
    extra_kwargs = {}
    if method_name == "duo_attention":
        extra_kwargs["model_short_name"] = model_short_name
    baseline = baseline_cls(num_layers, num_kv_heads, head_dim, **extra_kwargs)
    result = baseline.compress_timed(
        full_k, full_v, compression_ratio,
        attention_weights=attention_weights,
    )
    return result.layers, result.compress_time_ms, result.memory_bytes


def _compress_layer_budget(
    full_k, full_v, cr, num_layers, num_kv_heads, head_dim,
    gini_scores, importance_weights,
):
    """LayerBudget compression with greedy allocator."""
    seq_len = full_k.shape[1]
    full_bytes = 2 * num_layers * seq_len * num_kv_heads * head_dim * 2
    budget_bytes = int(full_bytes / cr)

    allocator = LayerBudgetAllocator(
        num_layers=num_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        available_bits=[4, 8, 16],
    )

    t0 = time.perf_counter()
    alloc_result = allocator.allocate(
        sparsity=gini_scores,
        importance=importance_weights,
        budget_bytes=budget_bytes,
        seq_len=seq_len,
    )
    compress_ms = (time.perf_counter() - t0) * 1000

    store = LayerKVStore(num_layers, num_kv_heads, head_dim)  # param name is num_heads
    store.store_from_full_cache(full_k, full_v, alloc_result.allocations)
    layers = store.get_all_layers()

    return layers, compress_ms, alloc_result.total_memory_bytes


def build_hf_cache(
    layers: List[Tuple[Tensor, Tensor, Tensor]],
    full_seq_len: int,
    device: str = "cuda:0",
    fill: str = "mean",
) -> List[Tuple[Tensor, Tensor]]:
    """Reconstruct HuggingFace-compatible KV cache from compressed layers.

    Args:
        fill: "mean" (paper default) or "zero" (deployment).
    """
    from transformers import DynamicCache

    cache = DynamicCache()
    for layer_idx, (k_compressed, v_compressed, indices) in enumerate(layers):
        n_heads = k_compressed.shape[2] if k_compressed.dim() == 4 else k_compressed.shape[1]
        head_dim = k_compressed.shape[-1]

        # Ensure 4D: (batch, heads, seq, dim)
        if k_compressed.dim() == 3:
            k_compressed = k_compressed.unsqueeze(0)
            v_compressed = v_compressed.unsqueeze(0)
        if k_compressed.shape[1] != n_heads:  # (batch, seq, heads, dim)
            k_compressed = k_compressed.transpose(1, 2)
            v_compressed = v_compressed.transpose(1, 2)

        # Build full-length tensors
        k_full = torch.zeros(1, n_heads, full_seq_len, head_dim,
                             dtype=k_compressed.dtype, device=device)
        v_full = torch.zeros(1, n_heads, full_seq_len, head_dim,
                             dtype=v_compressed.dtype, device=device)

        idx = indices.long().to(device)

        if fill == "mean":
            k_mean = k_compressed.mean(dim=2, keepdim=True).expand_as(k_full)
            v_mean = v_compressed.mean(dim=2, keepdim=True).expand_as(v_full)
            k_full.copy_(k_mean)
            v_full.copy_(v_mean)

        k_full[:, :, idx] = k_compressed
        v_full[:, :, idx] = v_compressed

        cache.update(k_full, v_full, layer_idx)

    return cache


# ── Metrics ─────────────────────────────────────────────────────────

def compute_ppl_from_cache(
    model, input_ids: Tensor, cache, prefix_len: int,
) -> float:
    """Compute perplexity on suffix tokens using cached prefix KV."""
    suffix_ids = input_ids[:, prefix_len:]
    if suffix_ids.shape[1] == 0:
        return float("inf")

    with torch.no_grad():
        out = model(
            input_ids=suffix_ids,
            past_key_values=cache,
            use_cache=False,
            return_dict=True,
        )

    logits = out.logits[:, :-1]  # predict next token
    targets = suffix_ids[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="mean",
    )
    return math.exp(min(loss.item(), 20))


def compute_next_token_logits(
    model, input_ids: Tensor, cache,
) -> Tensor:
    """Get logits for the next token given cached prefix."""
    with torch.no_grad():
        out = model(
            input_ids=input_ids[:, -1:],
            past_key_values=cache,
            use_cache=False,
            return_dict=True,
        )
    return out.logits[:, -1]  # (batch, vocab)


# ── Text metric helpers ─────────────────────────────────────────────

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gt_tokens) if gt_tokens else 0
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0)


def rouge_l(prediction: str, reference: str) -> float:
    """Simple ROUGE-L via longest common subsequence."""
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return 0.0
    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    p = lcs / m if m > 0 else 0
    r = lcs / n if n > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# ── Generation helpers ──────────────────────────────────────────────

def generate_from_cache(
    model, tokenizer, cache, prefix_len: int,
    suffix_ids: Optional[Tensor] = None,
    max_new_tokens: int = 64,
    device: str = "cuda:0",
    stop_patterns: Optional[List[str]] = None,
) -> str:
    """Generate text from a manually-built DynamicCache.

    Uses manual token-by-token decode with explicit position_ids,
    which is compatible with all transformers versions and avoids
    the cache_position issues with model.generate().

    Args:
        cache: DynamicCache built by build_hf_cache().
        prefix_len: Number of tokens in the cached prefix.
        suffix_ids: Optional additional input tokens after prefix (e.g., question).
                    Shape (1, suffix_len). If None, starts generating immediately.
        max_new_tokens: Max tokens to generate.
        device: CUDA device.

    Returns:
        Decoded generated text (excluding input).
    """
    eos_id = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_id = eos_id[0]

    cur_pos = prefix_len
    past_kv = cache

    # Step 1: Feed suffix tokens through cache if any
    if suffix_ids is not None and suffix_ids.shape[1] > 0:
        suffix_ids = suffix_ids.to(device)
        pos = torch.arange(
            cur_pos, cur_pos + suffix_ids.shape[1], device=device
        ).unsqueeze(0)
        with torch.no_grad():
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
        # Get logits for the last cached position
        # Feed a dummy token at position prefix_len-1
        # Actually we need to generate the next token from the cache
        dummy_input = torch.zeros(1, 1, dtype=torch.long, device=device)
        # We need the last token from the prefix — not available.
        # Instead, re-derive from cache: just do a forward with 0-length
        # Actually the standard trick: feed position cur_pos-1 again
        # Safer: caller should include at least the last input token as suffix
        return ""

    # Step 2: Autoregressive decode
    generated_ids = []
    # Check for answer pattern every N tokens to avoid decoding overhead
    _stop_check_interval = 8
    for step in range(max_new_tokens):
        next_token = last_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
        token_id = next_token.item()

        if eos_id is not None and token_id == eos_id:
            break

        generated_ids.append(token_id)

        # Early stop: check if generated text contains an answer pattern
        if stop_patterns and step > 0 and step % _stop_check_interval == 0:
            partial = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if any(p in partial for p in stop_patterns):
                break

        pos = torch.tensor([[cur_pos]], device=device)
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past_kv,
                position_ids=pos,
                return_dict=True,
                use_cache=True,
            )
        past_kv = out.past_key_values
        last_logits = out.logits[:, -1, :]
        cur_pos += 1

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
