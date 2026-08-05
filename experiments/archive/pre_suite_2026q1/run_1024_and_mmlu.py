#!/usr/bin/env python3
"""#27: Llama-2-7B@1024 with hook-based profiler (no OOM).
#28: MMLU re-run with inverted importance.

Hook-based profiler captures only the last-token attention row per layer
via forward hooks — O(S) memory per layer instead of O(S^2) from
output_attentions=True. This avoids the OOM on 32-head MHA at 1024 tokens.
"""

import gc, json, math, statistics, sys, time
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

RESULTS_DIR = Path(__file__).parent / "results" / "main_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def get_wikitext_chunks(tokenizer, target_len, n=6):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    all_ids = tokenizer.encode(all_text)
    return [all_ids[i:i+target_len] for i in range(0, len(all_ids)-target_len, target_len)][:n]


def build_cache(layers_data, full_seq_len, device, fill="mean"):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for li, (k, v, idx) in enumerate(layers_data):
        k, v = k.to(device), v.to(device)
        if k.dim() == 3: k, v = k.unsqueeze(0), v.unsqueeze(0)
        nt, nh, hd = k.shape[1], k.shape[2], k.shape[3]
        if nt == full_seq_len:
            cache.update(k.transpose(1,2), v.transpose(1,2), li)
            continue
        if fill == "mean":
            km = k[0].mean(dim=0, keepdim=True)
            vm = v[0].mean(dim=0, keepdim=True)
            kf = km.expand(full_seq_len, -1, -1).clone().unsqueeze(0)
            vf = vm.expand(full_seq_len, -1, -1).clone().unsqueeze(0)
        else:
            kf = torch.zeros(1, full_seq_len, nh, hd, dtype=k.dtype, device=device)
            vf = torch.zeros(1, full_seq_len, nh, hd, dtype=v.dtype, device=device)
        ix = idx.long().to(device); valid = ix[ix < full_seq_len]
        if valid.numel() > 0:
            kf[0, valid] = k[0, :valid.numel()]
            vf[0, valid] = v[0, :valid.numel()]
        cache.update(kf.transpose(1,2), vf.transpose(1,2), li)
    return cache


def compute_ppl(model, input_ids, past_kv, prefix_len, device):
    suffix = input_ids[:, prefix_len:]
    if suffix.shape[1] <= 1: return 1.0
    with torch.no_grad():
        pos = torch.arange(prefix_len, prefix_len + suffix.shape[1], device=device).unsqueeze(0)
        out = model(input_ids=suffix, past_key_values=past_kv, position_ids=pos, return_dict=True)
    logits = out.logits[:, :-1, :].contiguous()
    labels = suffix[:, 1:].contiguous()
    return math.exp(min(F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1)).item(), 20))


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


def do_compress_lb(full_k, full_v, gini_scores, prefix_len, cr, nl, nh, hd):
    """LayerBudget compression with inverted importance (now default)."""
    allocator = LayerBudgetAllocator(nl, nh, hd)
    store = LayerKVStore(nl, nh, hd)
    imp = allocator.compute_importance_weights(nl)  # inverted by default now
    fm = allocator.full_memory(prefix_len)
    alloc = allocator.allocate(gini_scores, imp, int(fm / cr), prefix_len)
    store.store_from_full_cache(full_k, full_v, alloc.allocations)
    return store.get_all_layers(), store.memory_usage()


# ═══════════════════════════════════════════════════════════════════
#  #27: Llama-2-7B @ 1024 with hook-based profiler
# ═══════════════════════════════════════════════════════════════════

def run_1024(model, tokenizer, model_short, n_texts=4):
    """Run LayerBudget at 1024 tokens using hook-based profiler."""
    print(f"\n{'='*70}")
    print(f"  #27: {model_short} @ 1024tok (hook-based profiler)")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, 1024, n_texts)
    CRS = [2.0, 3.0, 4.0, 6.0]
    METHODS = ["full_kv", "layer_budget", "h2o_uniform", "kivi_uniform"]
    results = {}

    for tidx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        prefix_len = int(1024 * 0.6)
        print(f"  [{tidx+1}/{len(chunks)}] prefix={prefix_len}", end=" ", flush=True)

        # Hook-based profiling: no output_attentions, O(S) memory
        profiler = LayerAttentionProfiler()
        profile = profiler.profile(input_ids[:, :prefix_len], model, device=str(device))
        gini = profile.gini_scores()

        # Get KV without attention weights (saves memory)
        clear_gpu()
        with torch.no_grad():
            out = model(input_ids=input_ids[:, :prefix_len], return_dict=True, use_cache=True)
        full_k, full_v = hf_to_deltacache(out.past_key_values)
        del out; clear_gpu()

        ref_kv = deltacache_to_hf(full_k, full_v, add_batch_dim=True)
        ref_ppl = compute_ppl(model, input_ids, ref_kv, prefix_len, device)
        del ref_kv; clear_gpu()
        print(f"ref={ref_ppl:.2f}", end=" ", flush=True)

        for cr in CRS:
            for m in METHODS:
                key = f"{m}@{cr}"
                if key not in results:
                    results[key] = {"m": m, "cr": cr, "ppls": [], "ratios": [], "mems": []}
                try:
                    actual_cr = 1.0 if m == "full_kv" else cr
                    if m == "full_kv":
                        layers = [(full_k[l:l+1], full_v[l:l+1], torch.arange(prefix_len)) for l in range(nl)]
                        mem = full_k.numel() * 2 * 2
                    elif m == "layer_budget":
                        layers, mem = do_compress_lb(full_k, full_v, gini, prefix_len, actual_cr, nl, nh, hd)
                    elif m in REGISTRY:
                        cls = REGISTRY[m]
                        bl = cls(nl, nh, hd)
                        # No attention_weights available with hook-based profiler
                        layers = bl.compress(full_k, full_v, actual_cr)
                        mem = bl.memory_bytes(layers)
                    else:
                        continue

                    for fill in ["mean"]:
                        pkv = build_cache(layers, prefix_len, device, fill=fill)
                        ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
                        results[key]["ppls"].append(ppl)
                        results[key]["ratios"].append(ppl / max(ref_ppl, 1e-6))
                        results[key]["mems"].append(mem)
                        del pkv
                    del layers
                except Exception as e:
                    print(f"ERR:{m}@{cr}:{e}", end=" ")
                clear_gpu()

        del full_k, full_v; clear_gpu()
        print("done", flush=True)

    # Print
    print(f"\n  {'Method':25s} {'CR':>4s} {'Ratio':>8s}")
    print(f"  {'-'*40}")
    summary = []
    for cr in CRS:
        entries = [(k, v) for k, v in results.items() if v["cr"] == cr and v["ratios"]]
        entries.sort(key=lambda x: statistics.mean(x[1]["ratios"]))
        for key, v in entries:
            avg_r = statistics.mean(v["ratios"])
            marker = " ***" if "layer_budget" in v["m"] else ""
            print(f"  {v['m']:25s} {cr:>4.0f}x {avg_r:>8.4f}{marker}")
            summary.append({"method": v["m"], "cr": cr, "mean_ratio": round(avg_r, 4), "n": len(v["ratios"])})
        print()

    path = RESULTS_DIR / f"main_v2_1024tok_{model_short.lower().replace('-','_')}.json"
    with open(path, "w") as f:
        json.dump({"metadata": {"model": model_short, "seq_len": 1024, "profiler": "hook-based",
                                "importance": "inverted", "fill": "mean",
                                "timestamp": datetime.now().isoformat()},
                   "summary": summary}, f, indent=2)
    print(f"  Saved: {path.name}")
    return summary


# ═══════════════════════════════════════════════════════════════════
#  #28: MMLU with inverted importance
# ═══════════════════════════════════════════════════════════════════

MMLU_QUESTIONS = {
    "abstract_algebra": [
        ("Find the degree of the extension Q(sqrt(2), sqrt(3)) over Q.", "A) 2\nB) 4\nC) 6\nD) 8", "B"),
        ("Statement 1: Every group of order p^2 is abelian. Statement 2: Every group of order p is cyclic.", "A) True, True\nB) False, False\nC) True, False\nD) False, True", "A"),
        ("The symmetric group S_3 has order", "A) 3\nB) 4\nC) 6\nD) 12", "C"),
    ],
    "computer_science": [
        ("Which sorting algorithm has O(n log n) average case?", "A) Bubble sort\nB) Merge sort\nC) Selection sort\nD) Insertion sort", "B"),
        ("In a BST, worst case search is", "A) O(1)\nB) O(log n)\nC) O(n)\nD) O(n log n)", "C"),
        ("TCP operates at which OSI layer?", "A) Network\nB) Transport\nC) Session\nD) Application", "B"),
        ("What is the time complexity of binary search?", "A) O(1)\nB) O(log n)\nC) O(n)\nD) O(n^2)", "B"),
        ("Which is NOT a type of join in SQL?", "A) INNER\nB) OUTER\nC) CROSS\nD) PARALLEL", "D"),
    ],
    "physics": [
        ("F = ma is Newton's", "A) First law\nB) Second law\nC) Third law\nD) Law of gravitation", "B"),
        ("Speed of light approximately", "A) 3x10^6 m/s\nB) 3x10^8 m/s\nC) 3x10^10 m/s\nD) 3x10^12 m/s", "B"),
        ("Entropy in isolated system", "A) Decreases\nB) Stays constant\nC) Increases or stays constant\nD) Oscillates", "C"),
    ],
    "biology": [
        ("DNA replication is", "A) Conservative\nB) Dispersive\nC) Semi-conservative\nD) Non-conservative", "C"),
        ("Powerhouse of the cell is", "A) Nucleus\nB) Ribosome\nC) Mitochondria\nD) Golgi", "C"),
        ("mRNA to protein is called", "A) Transcription\nB) Translation\nC) Replication\nD) Transformation", "B"),
    ],
    "mathematics": [
        ("Derivative of x^3", "A) x^2\nB) 3x^2\nC) 3x\nD) x^3/3", "B"),
        ("Integral of 1/x", "A) x\nB) ln|x| + C\nC) 1/x^2\nD) x^2/2", "B"),
        ("lim(x->0) sin(x)/x", "A) 0\nB) 1\nC) infinity\nD) undefined", "B"),
    ],
}


def run_mmlu(model, tokenizer, model_short, cr=3.0):
    """MMLU with inverted importance."""
    print(f"\n{'='*70}")
    print(f"  #28: MMLU ({model_short}, {cr}x, inverted importance)")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    methods = ["full_kv", "layer_budget", "h2o_uniform", "kivi_uniform"]
    correct = {m: 0 for m in methods}
    total = {m: 0 for m in methods}

    all_q = [(subj, q, c, a) for subj, qs in MMLU_QUESTIONS.items() for q, c, a in qs]
    print(f"  {len(all_q)} questions")

    answer_tokens = None
    profiler = LayerAttentionProfiler()

    for qi, (subj, q_text, choices, answer) in enumerate(all_q):
        prompt = f"Question: {q_text}\n{choices}\nAnswer:"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        sl = ids.shape[1]
        pl = sl - 1

        if answer_tokens is None:
            answer_tokens = {
                "A": tokenizer.encode(" A", add_special_tokens=False)[-1],
                "B": tokenizer.encode(" B", add_special_tokens=False)[-1],
                "C": tokenizer.encode(" C", add_special_tokens=False)[-1],
                "D": tokenizer.encode(" D", add_special_tokens=False)[-1],
            }

        # Hook-based profiling
        profile = profiler.profile(ids[:, :pl], model, device=str(device))
        gini = profile.gini_scores()

        with torch.no_grad():
            out = model(input_ids=ids[:, :pl], return_dict=True, use_cache=True)
        fk, fv = hf_to_deltacache(out.past_key_values)
        del out; clear_gpu()

        for m in methods:
            try:
                if m == "full_kv":
                    ly = [(fk[l:l+1], fv[l:l+1], torch.arange(pl)) for l in range(nl)]
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
                    logits = model(input_ids=last_tok, past_key_values=pkv, position_ids=pos, return_dict=True).logits
                pred_id = logits[0, -1, list(answer_tokens.values())].argmax().item()
                pred = list(answer_tokens.keys())[pred_id]
                correct[m] += int(pred == answer)
                total[m] += 1
                del pkv, ly
            except:
                total[m] += 1
        del fk, fv; clear_gpu()

    print(f"\n  {'Method':20s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>8s}")
    print(f"  {'-'*45}")
    rows = []
    for m in methods:
        acc = correct[m] / total[m] if total[m] > 0 else 0
        print(f"  {m:20s} {correct[m]:>8d} {total[m]:>6d} {acc:>8.1%}")
        rows.append({"method": m, "correct": correct[m], "total": total[m], "accuracy": round(acc, 4)})

    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-short", default="Llama-2-7B")
    parser.add_argument("--exps", nargs="+", default=["27", "28"], choices=["27", "28"])
    args = parser.parse_args()

    model, tokenizer = load_model_4bit(args.model)

    output = {"metadata": {"model": args.model, "model_short": args.model_short,
                           "importance": "inverted (default)",
                           "timestamp": datetime.now().isoformat()}}

    if "27" in args.exps:
        output["main_1024"] = run_1024(model, tokenizer, args.model_short, n_texts=4)

    if "28" in args.exps:
        output["mmlu"] = run_mmlu(model, tokenizer, args.model_short, cr=3.0)

    safe = args.model_short.replace("-", "_").lower()
    path = RESULTS_DIR / f"1024_mmlu_{safe}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")
