#!/usr/bin/env python3
"""#18: LongBench evaluation for LayerBudget.

Tasks: SingleDoc QA (qasper, multifieldqa_en), Summarization (gov_report),
       Few-shot (trec), Code (lcc).

Compares: Full KV, LayerBudget (inverted, 2x/4x), H2O (2x/4x), KIVI.
Uses F1 score for QA, ROUGE-L for summarization, accuracy for classification.

Runs on Mistral-7B-Instruct-v0.2 (4-bit) with context up to 4K tokens.
"""

import argparse, gc, json, math, re, statistics, string, sys, time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "longbench"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def load_model_4bit(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"\nLoading {model_name} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        device_map={"": "cuda:0"}, attn_implementation="eager", trust_remote_code=True)
    model.eval()
    return model, tokenizer


def inverted_importance(nl):
    sigmoid = LayerBudgetAllocator.compute_importance_weights(nl)
    return {l: sigmoid[nl - 1 - l] for l in range(nl)}


# ── Metrics ──

def normalize_answer(s):
    """Normalize for F1 computation."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gt_tokens) if gt_tokens else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def rouge_l(prediction, reference):
    """Simple ROUGE-L (longest common subsequence)."""
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref: return 0.0
    m, n = len(ref), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / n if n > 0 else 0
    rec = lcs / m if m > 0 else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


# ── Generation with KV compression ──

def generate_with_compression(model, tokenizer, prompt, method, cr, nl, nh, hd, device,
                               max_new_tokens=50):
    """Generate text with compressed KV cache prefix."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True,
                                  max_length=3500).to(device)
    seq_len = input_ids.shape[1]

    if method == "full_kv" or seq_len < 64:
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0, seq_len:], skip_special_tokens=True)

    # Use ~80% as prefix for compression, generate from rest
    prefix_len = int(seq_len * 0.8)
    prefix_ids = input_ids[:, :prefix_len]
    suffix_ids = input_ids[:, prefix_len:]

    # Prefill with attention
    with torch.no_grad():
        out = model(input_ids=prefix_ids, output_attentions=True, return_dict=True)
    full_k, full_v = hf_to_deltacache(out.past_key_values)
    attn = list(out.attentions)
    del out; clear_gpu()

    if method == "layer_budget":
        profiler = LayerAttentionProfiler()
        pr = profiler.profile_from_attention_weights(attn)
        sp = pr.gini_scores()
        imp = inverted_importance(nl)
        allocator = LayerBudgetAllocator(nl, nh, hd)
        store = LayerKVStore(nl, nh, hd)
        fm = allocator.full_memory(prefix_len)
        alloc = allocator.allocate(sp, imp, int(fm / cr), prefix_len)
        store.store_from_full_cache(full_k, full_v, alloc.allocations)
        layers = store.get_all_layers()
    elif method == "h2o":
        from baselines import REGISTRY
        cls = REGISTRY["h2o_uniform"]
        bl = cls(nl, nh, hd)
        layers = bl.compress(full_k, full_v, cr, attention_weights=attn)
    elif method == "kivi":
        from baselines import REGISTRY
        cls = REGISTRY["kivi_uniform"]
        bl = cls(nl, nh, hd)
        layers = bl.compress(full_k, full_v, cr)
    else:
        raise ValueError(f"Unknown method: {method}")

    del full_k, full_v, attn

    # Build mean-fill cache
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for li, (k, v, idx) in enumerate(layers):
        k, v = k.to(device), v.to(device)
        if k.dim() == 3: k, v = k.unsqueeze(0), v.unsqueeze(0)
        nt, nh_, hd_ = k.shape[1], k.shape[2], k.shape[3]
        if nt == prefix_len:
            cache.update(k.transpose(1,2), v.transpose(1,2), li)
        else:
            km = k[0].mean(dim=0, keepdim=True)
            vm = v[0].mean(dim=0, keepdim=True)
            kf = km.expand(prefix_len, -1, -1).clone().unsqueeze(0)
            vf = vm.expand(prefix_len, -1, -1).clone().unsqueeze(0)
            ix = idx.long().to(device); valid = ix[ix < prefix_len]
            if valid.numel() > 0:
                kf[0, valid] = k[0, :valid.numel()]
                vf[0, valid] = v[0, :valid.numel()]
            cache.update(kf.transpose(1,2), vf.transpose(1,2), li)
    del layers

    # Continue from suffix + generate
    with torch.no_grad():
        pos = torch.arange(prefix_len, prefix_len + suffix_ids.shape[1], device=device).unsqueeze(0)
        out = model(input_ids=suffix_ids, past_key_values=cache, position_ids=pos,
                    return_dict=True, use_cache=True)
        past_kv = out.past_key_values
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = out.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id: break
            out = model(input_ids=next_token, past_key_values=past_kv,
                       position_ids=torch.tensor([[generated.shape[1]-1]], device=device),
                       return_dict=True, use_cache=True)
            past_kv = out.past_key_values

    del cache, past_kv; clear_gpu()
    return tokenizer.decode(generated[0, seq_len:], skip_special_tokens=True)


# ── Main evaluation ──

def evaluate_task(model, tokenizer, task_name, method, cr, nl, nh, hd, device,
                  max_samples=20, max_new_tokens=50):
    """Evaluate one LongBench task with one compression method."""
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench", task_name, split="test", trust_remote_code=True)

    # Filter to reasonable lengths (< 4K tokens)
    samples = []
    for item in ds:
        ctx_len = len(tokenizer.encode(item["context"], truncation=True, max_length=4096))
        if 512 <= ctx_len <= 3500:
            samples.append(item)
        if len(samples) >= max_samples:
            break

    if not samples:
        print(f"    No suitable samples for {task_name}")
        return {"task": task_name, "method": method, "cr": cr, "score": 0, "n": 0}

    scores = []
    for idx, item in enumerate(samples):
        prompt = f"Context:\n{item['context']}\n\nQuestion: {item['input']}\nAnswer:"

        try:
            pred = generate_with_compression(model, tokenizer, prompt, method, cr,
                                              nl, nh, hd, device, max_new_tokens)
        except Exception as e:
            clear_gpu()
            continue

        # Score
        answers = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]
        if task_name in ["gov_report", "multi_news"]:
            score = max(rouge_l(pred, a) for a in answers)
        elif task_name in ["trec"]:
            score = 1.0 if any(normalize_answer(a) in normalize_answer(pred) for a in answers) else 0.0
        else:
            score = max(f1_score(pred, a) for a in answers)

        scores.append(score)
        clear_gpu()

    avg = statistics.mean(scores) if scores else 0
    return {"task": task_name, "method": method, "cr": cr,
            "score": round(avg, 4), "n": len(scores)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--model-short", default=None)
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()

    model_short = args.model_short or args.model.split("/")[-1]
    model, tokenizer = load_model_4bit(args.model)
    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    tasks = ["qasper", "multifieldqa_en", "gov_report", "trec"]
    methods = [
        ("full_kv", 1.0),
        ("layer_budget", 2.0),
        ("layer_budget", 4.0),
        ("h2o", 2.0),
        ("h2o", 4.0),
        ("kivi", 2.0),
    ]

    print(f"\n{'='*70}")
    print(f"  LongBench Evaluation ({model_short})")
    print(f"  Tasks: {tasks}")
    print(f"  Methods: {[(m,c) for m,c in methods]}")
    print(f"{'='*70}")

    all_results = []

    for task_name in tasks:
        print(f"\n  --- {task_name} ---")
        for method, cr in methods:
            label = f"{method}@{cr:.0f}x" if cr > 1 else method
            print(f"    {label:25s}", end=" ", flush=True)
            result = evaluate_task(model, tokenizer, task_name, method, cr,
                                    nl, nh, hd, device, args.max_samples)
            all_results.append(result)
            print(f"score={result['score']:.3f} (n={result['n']})")

    # Summary table
    print(f"\n  {'Task':20s}", end="")
    for method, cr in methods:
        label = f"{method}@{cr:.0f}x" if cr > 1 else method
        print(f" {label:>12s}", end="")
    print()
    print(f"  {'-'*(20 + 12*len(methods))}")
    for task in tasks:
        print(f"  {task:20s}", end="")
        for method, cr in methods:
            res = [r for r in all_results if r["task"] == task and r["method"] == method and r["cr"] == cr]
            score = res[0]["score"] if res else 0
            print(f" {score:>11.3f}", end="")
        print()

    # Save
    safe = model_short.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"longbench_{safe}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump({"metadata": {"model": args.model, "model_short": model_short,
                                "max_samples": args.max_samples,
                                "timestamp": datetime.now().isoformat()},
                   "results": all_results}, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
