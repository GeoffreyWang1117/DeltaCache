#!/usr/bin/env python3
"""Expanded MMLU evaluation: 4 questions per subject × 57 subjects = 228 questions.

Compares LayerBudget vs Full KV, H2O, KIVI at 3x compression.
Uses hook-based profiler for Llama-2-7B (4-bit).

Usage:
    python run_mmlu_expanded.py                         # default: Llama-2-7B, 3x
    python run_mmlu_expanded.py --model mistralai/Mistral-7B-Instruct-v0.2 --model-short Mistral-7B
    python run_mmlu_expanded.py --cr 4.0                # test at 4x compression
    python run_mmlu_expanded.py --n-per-subject 2       # fewer questions (faster)
"""

import gc
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from baselines import REGISTRY
from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "mmlu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_4bit(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"\nLoading {model_name} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        ),
        device_map={"": "cuda:0"},
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_mmlu_subset(n_per_subject=4, seed=42):
    """Load stratified MMLU subset: n questions per subject."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subject = defaultdict(list)
    for item in ds:
        by_subject[item["subject"]].append(item)

    random.seed(seed)
    subset = []
    for subject in sorted(by_subject.keys()):
        items = by_subject[subject]
        n = min(n_per_subject, len(items))
        sampled = random.sample(items, n)
        subset.extend(sampled)

    print(f"  MMLU subset: {len(subset)} questions from {len(by_subject)} subjects "
          f"({n_per_subject}/subject)")
    return subset


def build_cache(layers_data, full_seq_len, device, fill="mean"):
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for li, (k, v, idx) in enumerate(layers_data):
        k, v = k.to(device), v.to(device)
        if k.dim() == 3:
            k, v = k.unsqueeze(0), v.unsqueeze(0)
        nt, nh, hd = k.shape[1], k.shape[2], k.shape[3]
        if nt == full_seq_len:
            cache.update(k.transpose(1, 2), v.transpose(1, 2), li)
            continue
        if fill == "mean":
            km = k[0].mean(dim=0, keepdim=True)
            vm = v[0].mean(dim=0, keepdim=True)
            kf = km.expand(full_seq_len, -1, -1).clone().unsqueeze(0)
            vf = vm.expand(full_seq_len, -1, -1).clone().unsqueeze(0)
        else:
            kf = torch.zeros(1, full_seq_len, nh, hd, dtype=k.dtype, device=device)
            vf = torch.zeros(1, full_seq_len, nh, hd, dtype=v.dtype, device=device)
        ix = idx.long().to(device)
        valid = ix[ix < full_seq_len]
        if valid.numel() > 0:
            kf[0, valid] = k[0, : valid.numel()]
            vf[0, valid] = v[0, : valid.numel()]
        cache.update(kf.transpose(1, 2), vf.transpose(1, 2), li)
    return cache


def do_compress_lb(full_k, full_v, gini_scores, prefix_len, cr, nl, nh, hd):
    allocator = LayerBudgetAllocator(nl, nh, hd)
    store = LayerKVStore(nl, nh, hd)
    imp = allocator.compute_importance_weights(nl)
    fm = allocator.full_memory(prefix_len)
    alloc = allocator.allocate(gini_scores, imp, int(fm / cr), prefix_len)
    store.store_from_full_cache(full_k, full_v, alloc.allocations)
    return store.get_all_layers(), store.memory_usage()


def run_mmlu_expanded(model, tokenizer, model_short, cr=3.0, n_per_subject=4):
    print(f"\n{'='*70}")
    print(f"  Expanded MMLU: {model_short} @ {cr}x (inverted importance, mean-fill)")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    questions = load_mmlu_subset(n_per_subject=n_per_subject)
    methods = ["full_kv", "layer_budget", "h2o_uniform", "kivi_uniform"]

    # Get answer token IDs
    choice_labels = ["A", "B", "C", "D"]
    answer_tokens = {}
    for label in choice_labels:
        toks = tokenizer.encode(f" {label}", add_special_tokens=False)
        answer_tokens[label] = toks[-1]

    profiler = LayerAttentionProfiler()

    correct = {m: 0 for m in methods}
    total = {m: 0 for m in methods}
    per_subject = {m: defaultdict(lambda: {"correct": 0, "total": 0}) for m in methods}

    t0 = time.time()
    for qi, item in enumerate(questions):
        q = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]  # 0-3
        subject = item["subject"]
        answer_label = choice_labels[answer_idx]

        # Format prompt
        choices_str = "\n".join(f"{choice_labels[i]}) {c}" for i, c in enumerate(choices))
        prompt = f"Question: {q}\n{choices_str}\nAnswer:"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        sl = ids.shape[1]
        pl = sl - 1

        if pl < 2:
            continue

        # Profile
        try:
            profile = profiler.profile(ids[:, :pl], model, device=str(device))
            gini = profile.gini_scores()
        except Exception:
            continue

        # Get KV
        with torch.no_grad():
            out = model(input_ids=ids[:, :pl], return_dict=True, use_cache=True)
        fk, fv = hf_to_deltacache(out.past_key_values)
        del out
        clear_gpu()

        for m in methods:
            try:
                if m == "full_kv":
                    ly = [
                        (fk[l : l + 1], fv[l : l + 1], torch.arange(pl))
                        for l in range(nl)
                    ]
                elif m == "layer_budget":
                    ly, _ = do_compress_lb(fk, fv, gini, pl, cr, nl, nh, hd)
                elif m in REGISTRY:
                    cls = REGISTRY[m]
                    bl = cls(nl, nh, hd)
                    ly = bl.compress(fk, fv, cr)
                else:
                    continue

                pkv = build_cache(ly, pl, device, fill="mean")
                last_tok = ids[:, -1:]
                pos = torch.tensor([[pl]], device=device)
                with torch.no_grad():
                    logits = model(
                        input_ids=last_tok,
                        past_key_values=pkv,
                        position_ids=pos,
                        return_dict=True,
                    ).logits

                pred_id = logits[0, -1, list(answer_tokens.values())].argmax().item()
                pred = choice_labels[pred_id]
                is_correct = int(pred == answer_label)
                correct[m] += is_correct
                total[m] += 1
                per_subject[m][subject]["correct"] += is_correct
                per_subject[m][subject]["total"] += 1
                del pkv, ly
            except Exception:
                total[m] += 1
                per_subject[m][subject]["total"] += 1

        del fk, fv
        clear_gpu()

        if (qi + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (qi + 1) / elapsed
            remaining = (len(questions) - qi - 1) / rate
            acc_lb = correct["layer_budget"] / max(total["layer_budget"], 1)
            acc_fk = correct["full_kv"] / max(total["full_kv"], 1)
            print(
                f"  [{qi+1}/{len(questions)}] "
                f"FullKV={acc_fk:.1%} LB={acc_lb:.1%} "
                f"({remaining/60:.0f}min left)"
            )

    elapsed = time.time() - t0

    # Print results
    print(f"\n  {'Method':20s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>9s}")
    print(f"  {'-'*46}")
    rows = []
    for m in methods:
        acc = correct[m] / total[m] if total[m] > 0 else 0
        marker = " ***" if m == "layer_budget" else ""
        print(f"  {m:20s} {correct[m]:>8d} {total[m]:>6d} {acc:>8.1%}{marker}")
        rows.append({
            "method": m,
            "correct": correct[m],
            "total": total[m],
            "accuracy": round(acc, 4),
        })

    # Per-subject breakdown for LayerBudget
    print(f"\n  Per-subject accuracy (LayerBudget vs Full KV):")
    subject_data = []
    for subj in sorted(per_subject["layer_budget"].keys()):
        lb = per_subject["layer_budget"][subj]
        fk = per_subject["full_kv"][subj]
        lb_acc = lb["correct"] / lb["total"] if lb["total"] > 0 else 0
        fk_acc = fk["correct"] / fk["total"] if fk["total"] > 0 else 0
        gap = lb_acc - fk_acc
        flag = " DROP" if gap < -0.25 else ""
        subject_data.append({
            "subject": subj,
            "lb_acc": round(lb_acc, 4),
            "fk_acc": round(fk_acc, 4),
            "gap": round(gap, 4),
        })
    # Show worst subjects
    subject_data.sort(key=lambda x: x["gap"])
    for s in subject_data[:5]:
        print(f"    {s['subject']:40s} LB={s['lb_acc']:.0%} FK={s['fk_acc']:.0%} gap={s['gap']:+.0%}")
    print(f"    ... ({len(subject_data)} subjects total)")

    # Save
    output = {
        "metadata": {
            "model": model_short,
            "cr": cr,
            "n_per_subject": n_per_subject,
            "total_questions": len(questions),
            "n_subjects": len(per_subject["layer_budget"]),
            "importance": "inverted",
            "fill": "mean",
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        },
        "summary": rows,
        "per_subject": subject_data,
    }
    safe = model_short.replace("-", "_").replace("/", "_").lower()
    path = RESULTS_DIR / f"mmlu_expanded_{safe}_{cr}x_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {path}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-short", default="Llama-2-7B")
    parser.add_argument("--cr", type=float, default=3.0)
    parser.add_argument("--n-per-subject", type=int, default=4)
    args = parser.parse_args()

    model, tokenizer = load_model_4bit(args.model)
    run_mmlu_expanded(model, tokenizer, args.model_short, args.cr, args.n_per_subject)
