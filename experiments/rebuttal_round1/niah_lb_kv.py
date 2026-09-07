"""NIAH retrieval evaluation for LayerBudget-KV at CR=4× and 6× on Mistral-7B + Llama-2-7B.

Mirrors the suite/tasks/niah.py logic but is standalone so it can register the
LayerBudget-KV baseline before evaluating.

Configuration: 4096 token context, n_depths=5, n_repeats=2 (10 questions per cell).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import json  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

DELTACACHE_ROOT = Path("/home/coder-gw/Projects/DeltaCache")
sys.path.insert(0, str(DELTACACHE_ROOT))
sys.path.insert(0, str(DELTACACHE_ROOT / "experiments"))

# Register layer_budget_kv before importing eval_utils
import layer_budget_kv  # noqa: F401, E402
from baselines.base import REGISTRY  # noqa: E402

assert "layer_budget_kv" in REGISTRY

from suite.eval_utils import (  # noqa: E402
    extract_kv_and_attention, compress_kv, build_hf_cache, clear_gpu,
    generate_from_cache,
)
from deltacache.core.layer_profiler import LayerAttentionProfiler  # noqa: E402
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator  # noqa: E402

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.2", "mistral_7b", 32, 8, 128),
    ("meta-llama/Llama-2-7b-chat-hf", "llama2_7b", 32, 32, 128),
]
CRS = [4.0, 6.0]
SEQ_LEN = 1024
N_DEPTHS = 5
N_REPEATS = 2


def build_haystack(target_tokens, tokenizer):
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


def run_niah_cell(model, tokenizer, num_layers, num_kv_heads, head_dim, method, cr, device):
    target_len = SEQ_LEN
    haystack_budget = target_len - 100
    haystack_text = build_haystack(haystack_budget, tokenizer)
    haystack_tokens = tokenizer.encode(haystack_text, add_special_tokens=False)
    depths = [i / (N_DEPTHS - 1) for i in range(N_DEPTHS)]
    depth_results = []
    oom_total = 0
    for depth in depths:
        correct = 0
        total = 0
        for rep in range(N_REPEATS):
            needle_fact, needle_answer, question = NEEDLES[rep % len(NEEDLES)]
            insert_pos = int(len(haystack_tokens) * depth)
            needle_tokens = tokenizer.encode(f" {needle_fact} ", add_special_tokens=False)
            combined = haystack_tokens[:insert_pos] + needle_tokens + haystack_tokens[insert_pos:]
            context = tokenizer.decode(combined[:haystack_budget])
            prompt = (f"Read the following text carefully.\n\n"
                      f"{context}\n\nBased on the text above, answer: {question}\nAnswer:")
            input_ids = tokenizer.encode(
                prompt, return_tensors="pt",
                max_length=target_len, truncation=True, add_special_tokens=True,
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
                    importance = LayerBudgetAllocator.compute_importance_weights(num_layers)
                layers, _, _ = compress_kv(
                    method, full_k, full_v, cr, num_layers, num_kv_heads, head_dim,
                    attention_weights=attns,
                    gini_scores=gini,
                    importance_weights=importance,
                )
                cache = build_hf_cache(layers, prefix_len, device, fill="mean")
                output = generate_from_cache(
                    model, tokenizer, cache, prefix_len,
                    suffix_ids=suffix_ids, max_new_tokens=32, device=device,
                )
                if needle_answer.lower() in output.lower():
                    correct += 1
                total += 1
            except torch.cuda.OutOfMemoryError:
                oom_total += 1
                clear_gpu()
                if oom_total >= 5:
                    break
            except Exception as e:
                print(f"  [err] {type(e).__name__}: {e}", flush=True)
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
        if oom_total >= 5:
            break
        if total > 0:
            depth_results.append({"depth": depth, "correct": correct, "total": total,
                                  "accuracy": round(correct / total, 4)})
    accs = [d["accuracy"] for d in depth_results]
    return {
        "method": method, "cr": cr,
        "overall_accuracy": round(statistics.mean(accs), 4) if accs else 0.0,
        "depths": depth_results,
        "oom_count": oom_total,
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    overall = {"date": datetime.now().isoformat(), "results": []}
    for hf_name, short, NL, NH, HD in MODELS:
        print(f"\n=== {short} ===", flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(hf_name)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(
                hf_name, quantization_config=bnb, device_map={"": "cuda:0"},
                attn_implementation="eager", trust_remote_code=True,
            )
            model.eval()
            device = next(model.parameters()).device
            cells = []
            for cr in CRS:
                for method in ["layer_budget", "layer_budget_kv"]:
                    print(f"  CR={cr} {method} ...", flush=True)
                    res = run_niah_cell(model, tok, NL, NH, HD, method, cr, device)
                    cells.append(res)
                    print(f"    overall_accuracy={res['overall_accuracy']:.3f} (OOM={res['oom_count']})", flush=True)
            overall["results"].append({"model": short, "cells": cells})
            del model, tok
            clear_gpu()
        except Exception as e:
            print(f"FAILED {short}: {e}")
            import traceback; traceback.print_exc()
            overall["results"].append({"model": short, "error": str(e)})
    out_path = OUT_DIR / f"niah_lb_kv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
