#!/usr/bin/env python3
"""#19: RULER-style evaluation for KV cache compression.

Implements three core RULER tasks synthetically:
  1. Needle-in-a-Haystack (NIAH): Find a specific fact buried in padding text
  2. Multi-Key Retrieval (MKR): Retrieve values for multiple keys from a KV store
  3. Variable Tracking (VT): Track variable assignments through a sequence

Tests whether compressed KV cache retains fine-grained retrieval ability.
Runs on Mistral-7B at 2K-4K contexts with LayerBudget vs H2O vs KIVI vs Full KV.
"""

import argparse, gc, json, math, random, re, statistics, sys, time
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

RESULTS_DIR = Path(__file__).parent / "results" / "ruler"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ── Haystack filler text ──

FILLER_SENTENCES = [
    "The weather in coastal regions varies significantly throughout the year.",
    "Modern computing relies on advances in semiconductor technology.",
    "Renewable energy sources include solar, wind, and hydroelectric power.",
    "The human genome contains approximately three billion base pairs.",
    "International trade agreements shape global economic relationships.",
    "Machine learning algorithms require large datasets for effective training.",
    "Ancient civilizations developed sophisticated systems of governance.",
    "Biodiversity conservation efforts focus on protecting endangered species.",
    "The periodic table organizes elements by their atomic properties.",
    "Urban planning addresses transportation, housing, and public spaces.",
    "Quantum mechanics describes behavior at the subatomic level.",
    "Agricultural practices have evolved dramatically over millennia.",
    "The internet has transformed communication and information sharing.",
    "Climate models predict changes in temperature and precipitation patterns.",
    "Philosophical inquiry explores fundamental questions about existence.",
    "Medical research continues to develop new treatments for diseases.",
    "Artistic expression takes many forms across cultures and time periods.",
    "Economic indicators help measure the health of national economies.",
    "Space exploration has expanded our understanding of the universe.",
    "Educational systems vary widely between different countries and cultures.",
]


def make_haystack(tokenizer, target_tokens, needle_position_frac=0.5):
    """Create a haystack of filler text with approximately target_tokens tokens."""
    text_parts = []
    current_tokens = 0
    while current_tokens < target_tokens:
        sentence = random.choice(FILLER_SENTENCES)
        text_parts.append(sentence)
        current_tokens += len(tokenizer.encode(sentence, add_special_tokens=False))
    return " ".join(text_parts)


# ── Task generators ──

def generate_niah(tokenizer, context_len, n_samples=15):
    """Needle-in-a-Haystack: find a specific fact in padding text."""
    NEEDLES = [
        ("The secret code is 7392.", "7392"),
        ("The magic number is 4815.", "4815"),
        ("The password is alpha-bravo-charlie.", "alpha-bravo-charlie"),
        ("The answer to the question is 42.", "42"),
        ("The hidden city is called Atlantis.", "Atlantis"),
        ("The key ingredient is saffron.", "saffron"),
        ("The launch date is March 15th.", "March 15"),
        ("The winning score was 97 points.", "97"),
        ("The treasure is buried under the oak tree.", "oak tree"),
        ("The final answer is approximately 3.14159.", "3.14159"),
        ("The captain's name is Rodriguez.", "Rodriguez"),
        ("The temperature was exactly 37 degrees.", "37"),
        ("The book was published in 1984.", "1984"),
        ("The correct option is choice B.", "B"),
        ("The signal frequency is 440 Hz.", "440"),
    ]

    samples = []
    for i in range(min(n_samples, len(NEEDLES))):
        needle_text, answer = NEEDLES[i]
        # Place needle at varying depths
        depth = (i / max(n_samples - 1, 1))  # 0.0 to 1.0
        haystack = make_haystack(tokenizer, context_len - 100)

        # Split haystack and insert needle
        sentences = haystack.split(". ")
        insert_idx = max(1, int(len(sentences) * depth))
        sentences.insert(insert_idx, needle_text)
        context = ". ".join(sentences)

        prompt = f"{context}\n\nBased on the text above, what is the specific value or answer mentioned? Answer concisely:"
        samples.append({"prompt": prompt, "answer": answer, "depth": round(depth, 2)})

    return samples


def generate_mkr(tokenizer, context_len, n_samples=12):
    """Multi-Key Retrieval: retrieve values for keys from a KV store in text."""
    KEYS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
            "iota", "kappa", "lambda", "mu"]
    samples = []

    for i in range(n_samples):
        # Create random KV pairs
        n_pairs = 6
        pairs = {}
        for j in range(n_pairs):
            key = KEYS[j]
            value = str(random.randint(100, 999))
            pairs[key] = value

        # Build context with KV pairs scattered in filler
        haystack = make_haystack(tokenizer, context_len - 200)
        sentences = haystack.split(". ")
        for j, (k, v) in enumerate(pairs.items()):
            insert_pos = max(1, int(len(sentences) * (j + 1) / (n_pairs + 1)))
            sentences.insert(insert_pos, f"Record: {k} = {v}")

        context = ". ".join(sentences)

        # Ask for a specific key
        query_key = KEYS[i % n_pairs]
        query_value = pairs[query_key]
        prompt = f"{context}\n\nWhat is the value associated with the key '{query_key}'? Answer with just the number:"
        samples.append({"prompt": prompt, "answer": query_value, "key": query_key})

    return samples


def generate_vt(tokenizer, context_len, n_samples=12):
    """Variable Tracking: track variable assignments through reassignments."""
    VARS = ["x", "y", "z", "w", "a", "b"]
    samples = []

    for i in range(n_samples):
        var = VARS[i % len(VARS)]
        # Chain of assignments
        n_assignments = 4
        values = [str(random.randint(10, 99)) for _ in range(n_assignments)]

        haystack = make_haystack(tokenizer, context_len - 200)
        sentences = haystack.split(". ")

        for j, val in enumerate(values):
            insert_pos = max(1, int(len(sentences) * (j + 1) / (n_assignments + 1)))
            sentences.insert(insert_pos, f"Set {var} = {val}")

        context = ". ".join(sentences)
        final_value = values[-1]  # Last assignment wins

        prompt = f"{context}\n\nAfter all the assignments above, what is the final value of {var}? Answer with just the number:"
        samples.append({"prompt": prompt, "answer": final_value, "variable": var})

    return samples


# ── Generation with compression ──

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


def generate_compressed(model, tokenizer, prompt, method, cr, nl, nh, hd, device,
                        max_new_tokens=30):
    """Generate with compressed KV cache prefix."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True,
                                  max_length=3800).to(device)
    seq_len = input_ids.shape[1]

    if method == "full_kv" or seq_len < 64:
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0, seq_len:], skip_special_tokens=True).strip()

    prefix_len = int(seq_len * 0.85)
    suffix_ids = input_ids[:, prefix_len:]

    with torch.no_grad():
        out = model(input_ids=input_ids[:, :prefix_len], output_attentions=True, return_dict=True)
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
        bl = REGISTRY["h2o_uniform"](nl, nh, hd)
        layers = bl.compress(full_k, full_v, cr, attention_weights=attn)
    elif method == "kivi":
        from baselines import REGISTRY
        bl = REGISTRY["kivi_uniform"](nl, nh, hd)
        layers = bl.compress(full_k, full_v, cr)
    else:
        raise ValueError(f"Unknown method: {method}")

    del full_k, full_v, attn

    # Mean-fill cache
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for li, (k, v, idx) in enumerate(layers):
        k, v = k.to(device), v.to(device)
        if k.dim() == 3: k, v = k.unsqueeze(0), v.unsqueeze(0)
        nt, nh_, hd_ = k.shape[1], k.shape[2], k.shape[3]
        if nt == prefix_len:
            cache.update(k.transpose(1, 2), v.transpose(1, 2), li)
        else:
            km = k[0].mean(dim=0, keepdim=True)
            vm = v[0].mean(dim=0, keepdim=True)
            kf = km.expand(prefix_len, -1, -1).clone().unsqueeze(0)
            vf = vm.expand(prefix_len, -1, -1).clone().unsqueeze(0)
            ix = idx.long().to(device); valid = ix[ix < prefix_len]
            if valid.numel() > 0:
                kf[0, valid] = k[0, :valid.numel()]
                vf[0, valid] = v[0, :valid.numel()]
            cache.update(kf.transpose(1, 2), vf.transpose(1, 2), li)
    del layers

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
                       position_ids=torch.tensor([[generated.shape[1] - 1]], device=device),
                       return_dict=True, use_cache=True)
            past_kv = out.past_key_values

    del cache, past_kv; clear_gpu()
    return tokenizer.decode(generated[0, seq_len:], skip_special_tokens=True).strip()


def check_answer(prediction, ground_truth):
    """Check if prediction contains the ground truth answer."""
    pred_lower = prediction.lower().strip()
    gt_lower = ground_truth.lower().strip()
    return 1.0 if gt_lower in pred_lower else 0.0


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--model-short", default=None)
    parser.add_argument("--context-len", type=int, default=2048,
                        help="Target context length in tokens")
    parser.add_argument("--n-samples", type=int, default=12)
    args = parser.parse_args()

    model_short = args.model_short or args.model.split("/")[-1]
    model, tokenizer = load_model_4bit(args.model)
    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    methods = [
        ("full_kv", 1.0),
        ("layer_budget", 2.0),
        ("layer_budget", 4.0),
        ("h2o", 2.0),
        ("h2o", 4.0),
        ("kivi", 2.0),
    ]

    tasks = {
        "niah": generate_niah(tokenizer, args.context_len, args.n_samples),
        "mkr": generate_mkr(tokenizer, args.context_len, args.n_samples),
        "vt": generate_vt(tokenizer, args.context_len, args.n_samples),
    }

    print(f"\n{'='*70}")
    print(f"  RULER Evaluation ({model_short}, ~{args.context_len} tokens)")
    print(f"  Tasks: {list(tasks.keys())} ({sum(len(v) for v in tasks.values())} total samples)")
    print(f"{'='*70}")

    all_results = []

    for task_name, samples in tasks.items():
        print(f"\n  --- {task_name} ({len(samples)} samples) ---")
        for method, cr in methods:
            label = f"{method}@{cr:.0f}x" if cr > 1 else method
            scores = []
            for si, sample in enumerate(samples):
                try:
                    pred = generate_compressed(model, tokenizer, sample["prompt"],
                                               method, cr, nl, nh, hd, device)
                    score = check_answer(pred, sample["answer"])
                    scores.append(score)
                except Exception as e:
                    clear_gpu()
                    continue

            acc = statistics.mean(scores) if scores else 0
            print(f"    {label:25s} acc={acc:.1%} ({sum(scores):.0f}/{len(scores)})")
            all_results.append({
                "task": task_name, "method": method, "cr": cr,
                "accuracy": round(acc, 4), "correct": sum(scores), "total": len(scores),
            })

    # Summary
    print(f"\n  {'Task':>6s}", end="")
    for method, cr in methods:
        label = f"{method}@{cr:.0f}x" if cr > 1 else method
        print(f" {label:>12s}", end="")
    print()
    print(f"  {'-'*(6 + 12*len(methods))}")
    for task_name in tasks:
        print(f"  {task_name:>6s}", end="")
        for method, cr in methods:
            res = [r for r in all_results if r["task"] == task_name and r["method"] == method and r["cr"] == cr]
            acc = res[0]["accuracy"] if res else 0
            print(f" {acc:>11.1%}", end="")
        print()

    # Average
    print(f"  {'AVG':>6s}", end="")
    for method, cr in methods:
        accs = [r["accuracy"] for r in all_results if r["method"] == method and r["cr"] == cr]
        avg = statistics.mean(accs) if accs else 0
        print(f" {avg:>11.1%}", end="")
    print()

    safe = model_short.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"ruler_{safe}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump({"metadata": {"model": args.model, "model_short": model_short,
                                "context_len": args.context_len, "n_samples": args.n_samples,
                                "timestamp": datetime.now().isoformat()},
                   "results": all_results}, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
