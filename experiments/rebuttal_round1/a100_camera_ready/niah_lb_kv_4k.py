"""NIAH at 4096 ctx for LayerBudget-KV — A100 camera-ready.

Per-cell checkpointing means a pre-emption only loses the in-progress cell.
Rerun the script and completed cells are skipped.

Smoke mode (env DC_SMOKE=1): runs only Mistral-7B at CR=4× (~5 min) for
sanity-checking on a fresh instance before committing to the full sweep.

Outputs:
  results/niah_4k/<model>__<cr>__<method>.json   (per-cell)
  results/niah_lb_kv_4k_<timestamp>.json         (aggregated bundle)
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import random
import statistics
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

from _common import setup_paths, CellCheckpoint, Timer, announce  # noqa: E402

THIS = Path(__file__).resolve()
REPO = setup_paths(__file__)

import layer_budget_kv  # noqa: F401, E402
from baselines.base import REGISTRY  # noqa: E402

assert "layer_budget_kv" in REGISTRY, "layer_budget_kv not registered"

from suite.eval_utils import (  # noqa: E402
    extract_kv_and_attention, compress_kv, build_hf_cache, clear_gpu,
    generate_from_cache,
)
from deltacache.core.layer_profiler import LayerAttentionProfiler  # noqa: E402
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator  # noqa: E402

OUT_DIR = THIS.parent / "results" / "niah_4k"
SMOKE = os.environ.get("DC_SMOKE", "0") == "1"

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

if SMOKE:
    MODELS = [("mistralai/Mistral-7B-Instruct-v0.2", "mistral_7b", 32, 8, 128)]
    CRS = [4.0]
    METHODS = ["layer_budget_kv"]
    SEQ_LEN = 4096
    N_DEPTHS = 3
    N_REPEATS = 1
else:
    MODELS = [
        ("mistralai/Mistral-7B-Instruct-v0.2", "mistral_7b", 32, 8, 128),
        ("meta-llama/Llama-2-7b-chat-hf", "llama2_7b", 32, 32, 128),
    ]
    CRS = [2.0, 4.0, 6.0]
    METHODS = ["layer_budget", "layer_budget_kv"]
    SEQ_LEN = 4096
    N_DEPTHS = 10
    N_REPEATS = 3


def build_haystack(target_tokens, tokenizer):
    rng = random.Random(42)
    sentences = []
    est = 0
    while est < target_tokens:
        s = rng.choice(HAYSTACK_SENTENCES)
        sentences.append(s); est += len(s.split()) * 1.3
    text = " ".join(sentences)
    tokens = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(tokens)


def run_cell(model, tokenizer, NL, NH, HD, method, cr, device):
    target_len = SEQ_LEN
    haystack_budget = target_len - 100
    haystack = build_haystack(haystack_budget, tokenizer)
    hay_tokens = tokenizer.encode(haystack, add_special_tokens=False)
    depths = [i / max(N_DEPTHS - 1, 1) for i in range(N_DEPTHS)]
    depth_results = []
    oom_total = 0
    err_total = 0
    for depth in depths:
        correct = 0
        total = 0
        for rep in range(N_REPEATS):
            needle, ans, q = NEEDLES[rep % len(NEEDLES)]
            insert = int(len(hay_tokens) * depth)
            needle_tok = tokenizer.encode(f" {needle} ", add_special_tokens=False)
            combined = hay_tokens[:insert] + needle_tok + hay_tokens[insert:]
            ctx = tokenizer.decode(combined[:haystack_budget])
            prompt = (f"Read the following text carefully.\n\n{ctx}\n\n"
                      f"Based on the text above, answer: {q}\nAnswer:")
            input_ids = tokenizer.encode(
                prompt, return_tensors="pt", max_length=target_len,
                truncation=True, add_special_tokens=True,
            ).to(device)
            seq_len = input_ids.shape[1]
            prefix_len = int(seq_len * 0.85)
            suffix_ids = input_ids[:, prefix_len:]
            try:
                full_k, full_v, attns = extract_kv_and_attention(
                    model, input_ids[:, :prefix_len], need_attention=True,
                )
                gini = importance = None
                if method == "layer_budget":
                    profiler = LayerAttentionProfiler()
                    pr = profiler.profile_from_attention_weights(attns)
                    gini = pr.gini_scores()
                    importance = LayerBudgetAllocator.compute_importance_weights(NL)
                layers, _, _ = compress_kv(
                    method, full_k, full_v, cr, NL, NH, HD,
                    attention_weights=attns, gini_scores=gini, importance_weights=importance,
                )
                cache = build_hf_cache(layers, prefix_len, device, fill="mean")
                out = generate_from_cache(
                    model, tokenizer, cache, prefix_len,
                    suffix_ids=suffix_ids, max_new_tokens=32, device=device,
                )
                if ans.lower() in out.lower():
                    correct += 1
                total += 1
            except torch.cuda.OutOfMemoryError:
                oom_total += 1; clear_gpu()
                if oom_total >= 5: break
            except Exception as e:
                err_total += 1
                print(f"  [err {err_total}] {type(e).__name__}: {e}", flush=True)
                clear_gpu()
            finally:
                # Drop the references before clear_gpu(), which starts with gc.collect():
                # the collector can only reclaim what nothing points at any more.
                # This was exec("del <name>") in a loop, which cannot work. exec gets a
                # copy of the function's locals, so the del applied to the copy and every
                # tensor stayed alive until the frame exited. Rebinding does drop them,
                # and unlike del it is safe when a name was never assigned.
                full_k = full_v = attns = layers = cache = None
                clear_gpu()
        if oom_total >= 5: break
        if total > 0:
            depth_results.append({"depth": depth, "correct": correct, "total": total,
                                  "accuracy": round(correct/total, 4)})
    accs = [d["accuracy"] for d in depth_results]
    return {
        "method": method, "cr": cr, "seq_len": SEQ_LEN,
        "n_depths": N_DEPTHS, "n_repeats": N_REPEATS,
        "overall_accuracy": round(statistics.mean(accs), 4) if accs else 0,
        "depths": depth_results,
        "oom_count": oom_total,
        "err_count": err_total,
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CellCheckpoint(OUT_DIR)

    if SMOKE:
        announce("SMOKE MODE: Mistral-7B / CR=4× / LB-KV / 3 depths × 1 rep")
    else:
        announce(f"NIAH @ 4K — {len(MODELS)} models × {len(CRS)} CRs × {len(METHODS)} methods")

    overall_t = Timer("total run")
    for hf_name, short, NL, NH, HD in MODELS:
        announce(f"Model: {short} ({hf_name})")
        # Plan cells
        cells_to_run = []
        for cr in CRS:
            for method in METHODS:
                key = f"{short}__cr{cr}__{method}"
                if ckpt.is_done(key):
                    print(f"[skip] {key} already done", flush=True)
                    continue
                cells_to_run.append((cr, method, key))
        if not cells_to_run:
            print(f"[skip] all {short} cells already done", flush=True)
            continue

        # Load model only if there are cells to run
        load_t = Timer(f"load {short}")
        try:
            tok = AutoTokenizer.from_pretrained(hf_name)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(
                hf_name, quantization_config=bnb, device_map={"": "cuda:0"},
                attn_implementation="eager", trust_remote_code=True,
            )
            model.eval()
            device = next(model.parameters()).device
        except Exception as e:
            traceback.print_exc()
            ckpt.save(f"{short}__model_load_FAILED",
                      {"model": short, "error": str(e), "stage": "model_load"})
            continue
        finally:
            load_t.end()

        for cr, method, key in cells_to_run:
            cell_t = Timer(f"{key}")
            try:
                res = run_cell(model, tok, NL, NH, HD, method, cr, device)
                res["model"] = short
                ckpt.save(key, res)
                print(f"  → accuracy={res['overall_accuracy']:.3f} (OOM={res['oom_count']}, err={res['err_count']})", flush=True)
            except Exception as e:
                traceback.print_exc()
                ckpt.save(f"{key}_FAILED",
                          {"model": short, "cr": cr, "method": method, "error": str(e)})
            cell_t.end()

        del model, tok
        clear_gpu()

    bundle_path = OUT_DIR.parent / f"niah_lb_kv_4k_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    ckpt.aggregate(bundle_path)
    overall_t.end()
    print(f"\n[DONE] Aggregated → {bundle_path}", flush=True)


if __name__ == "__main__":
    main()
