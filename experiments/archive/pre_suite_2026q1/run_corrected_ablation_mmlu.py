#!/usr/bin/env python3
"""Corrected ablation + MMLU using zero-fill for evicted positions."""

import gc
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from baselines import REGISTRY
from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_cache_CORRECTED(layers_data, full_seq_len, device):
    """ZERO-FILL for evicted positions."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for li, (k, v, idx) in enumerate(layers_data):
        k, v = k.to(device), v.to(device)
        if k.dim() == 3:
            k, v = k.unsqueeze(0), v.unsqueeze(0)
        nt, nh, hd = k.shape[1], k.shape[2], k.shape[3]
        if nt == full_seq_len:
            cache.update(k.transpose(1, 2), v.transpose(1, 2), li)
        else:
            kf = torch.zeros(1, full_seq_len, nh, hd, dtype=k.dtype, device=device)
            vf = torch.zeros(1, full_seq_len, nh, hd, dtype=v.dtype, device=device)
            ix = idx.long().to(device)
            valid = ix[ix < full_seq_len]
            vc = valid.shape[0]
            if vc > 0:
                kf[0, valid] = k[0, :vc]
                vf[0, valid] = v[0, :vc]
            cache.update(kf.transpose(1, 2), vf.transpose(1, 2), li)
    return cache


def compute_ppl(model, input_ids, past_kv, prefix_len, device):
    suffix = input_ids[:, prefix_len:]
    if suffix.shape[1] <= 1:
        return 1.0
    with torch.no_grad():
        pos = torch.arange(prefix_len, prefix_len + suffix.shape[1], device=device).unsqueeze(0)
        out = model(input_ids=suffix, past_key_values=past_kv, position_ids=pos, return_dict=True)
    logits = out.logits[:, :-1, :].contiguous()
    labels = suffix[:, 1:].contiguous()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean")
    return math.exp(min(loss.item(), 20))


def get_wikitext_chunks(tokenizer, target_len, n=4):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    all_ids = tokenizer.encode(all_text)
    return [all_ids[i:i + target_len] for i in range(0, len(all_ids) - target_len, target_len)][:n]


def load_model_4bit(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"\nLoading {model_name} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        device_map="auto", attn_implementation="eager", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# =========================================================
#  ABLATION (corrected)
# =========================================================

def run_ablation(model, tokenizer, model_short, seq_len=512, cr=3.0, n_texts=4):
    print(f"\n{'='*70}")
    print(f"  CORRECTED ABLATION: {model_short} @ {seq_len}tok, {cr}x")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    chunks = get_wikitext_chunks(tokenizer, seq_len, n_texts)
    results = {}

    for tidx, tokens in enumerate(chunks):
        input_ids = torch.tensor([tokens], device=device)
        prefix_len = int(seq_len * 0.6)
        print(f"  [{tidx+1}/{len(chunks)}]", end=" ", flush=True)

        with torch.no_grad():
            out = model(input_ids=input_ids[:, :prefix_len], output_attentions=True, return_dict=True)
        full_k, full_v = hf_to_deltacache(out.past_key_values)
        attn = list(out.attentions)
        del out; clear_gpu()

        ref_kv = deltacache_to_hf(full_k, full_v, add_batch_dim=True)
        ref_ppl = compute_ppl(model, input_ids, ref_kv, prefix_len, device)
        del ref_kv

        profiler = LayerAttentionProfiler()
        pr = profiler.profile_from_attention_weights(attn)
        sparsity = pr.gini_scores()
        importance = LayerBudgetAllocator.compute_importance_weights(nl)

        def run_variant(name, use_sparsity=True, use_importance=True, bits_set=None,
                        evict_only=False, quant_only=False, use_per_input=True):
            allocator = LayerBudgetAllocator(nl, nh, hd, available_bits=bits_set or [4, 8, 16])
            store = LayerKVStore(nl, nh, hd)
            fm = allocator.full_memory(prefix_len)
            budget = int(fm / cr)

            sp = sparsity if use_sparsity else {l: 0.5 for l in range(nl)}
            imp = importance if use_importance else {l: 0.5 for l in range(nl)}
            if not use_per_input:
                sp = {l: 0.7 for l in range(nl)}

            alloc = allocator.allocate(sp, imp, budget, prefix_len)

            if evict_only:
                for a in alloc.allocations:
                    a.quant_bits = 16
            elif quant_only:
                for a in alloc.allocations:
                    a.token_budget = prefix_len

            store.store_from_full_cache(full_k, full_v, alloc.allocations)
            layers = store.get_all_layers()
            # CORRECTED: zero-fill
            pkv = build_cache_CORRECTED(layers, prefix_len, device)
            ppl = compute_ppl(model, input_ids, pkv, prefix_len, device)
            del pkv, layers
            return ppl, ppl / max(ref_ppl, 1e-6), store.memory_usage()

        variants = {
            "eviction_only": dict(evict_only=True),
            "quant_only": dict(quant_only=True),
            "joint": dict(),
            "sparsity_only": dict(use_importance=False),
            "importance_only": dict(use_sparsity=False),
            "combined": dict(),
            "bits_4_16": dict(bits_set=[4, 16]),
            "bits_4_8_16": dict(),
            "per_input": dict(),
            "fixed_profile": dict(use_per_input=False),
        }

        for vname, kwargs in variants.items():
            if vname not in results:
                results[vname] = {"ppls": [], "ratios": [], "mems": []}
            try:
                ppl, ratio, mem = run_variant(vname, **kwargs)
                results[vname]["ppls"].append(ppl)
                results[vname]["ratios"].append(ratio)
                results[vname]["mems"].append(mem)
            except Exception as e:
                print(f"ERR:{vname}", end=" ", flush=True)

        del full_k, full_v, attn; clear_gpu()
        print(f"ref={ref_ppl:.2f} done", flush=True)

    print(f"\n  {'Variant':25s} {'PPL':>8s} {'Ratio':>8s} {'Mem(KB)':>8s}")
    print(f"  {'-'*55}")
    rows = []
    for vname, v in results.items():
        if not v["ppls"]:
            continue
        r = {"variant": vname, "mean_ppl": round(statistics.mean(v["ppls"]), 4),
             "mean_ratio": round(statistics.mean(v["ratios"]), 4),
             "mean_mem_kb": round(statistics.mean(v["mems"]) / 1024, 1)}
        rows.append(r)
        print(f"  {vname:25s} {r['mean_ppl']:>8.2f} {r['mean_ratio']:>8.4f} {r['mean_mem_kb']:>7.1f}KB")

    path = RESULTS_DIR / f"corrected_ablation_{model_short.lower().replace('-', '_')}.json"
    with open(path, "w") as f:
        json.dump({"metadata": {"model": model_short, "seq_len": seq_len, "cr": cr,
                                "eval": "CORRECTED_zero_fill", "timestamp": datetime.now().isoformat()},
                   "results": rows}, f, indent=2)
    print(f"  Saved: {path.name}")


# =========================================================
#  MMLU (corrected)
# =========================================================

MMLU_QUESTIONS = {
    "abstract_algebra": [
        ("Find the degree of the extension Q(sqrt(2), sqrt(3)) over Q.", "A) 2\nB) 4\nC) 6\nD) 8", "B"),
        ("Statement 1: Every group of order p^2 is abelian. Statement 2: Every group of order p is cyclic.", "A) True, True\nB) False, False\nC) True, False\nD) False, True", "A"),
        ("The symmetric group S_3 has order", "A) 3\nB) 4\nC) 6\nD) 12", "C"),
        ("How many elements of order 2 are in Z_8?", "A) 0\nB) 1\nC) 2\nD) 4", "B"),
        ("Which of the following is a field?", "A) Z_6\nB) Z_7\nC) Z_8\nD) Z_9", "B"),
    ],
    "computer_science": [
        ("Which sorting algorithm has O(n log n) average case?", "A) Bubble sort\nB) Merge sort\nC) Selection sort\nD) Insertion sort", "B"),
        ("In a BST, worst case search is", "A) O(1)\nB) O(log n)\nC) O(n)\nD) O(n log n)", "C"),
        ("FIFO ordering is used by", "A) Stack\nB) Queue\nC) Tree\nD) Graph", "B"),
        ("TCP operates at which OSI layer?", "A) Network\nB) Transport\nC) Session\nD) Application", "B"),
        ("What is the time complexity of binary search?", "A) O(1)\nB) O(log n)\nC) O(n)\nD) O(n^2)", "B"),
        ("Which is NOT a type of join in SQL?", "A) INNER\nB) OUTER\nC) CROSS\nD) PARALLEL", "D"),
        ("A deadlock requires all EXCEPT", "A) Mutual exclusion\nB) Preemption\nC) Hold and wait\nD) Circular wait", "B"),
        ("Worst case for quicksort is", "A) O(n)\nB) O(n log n)\nC) O(n^2)\nD) O(2^n)", "C"),
    ],
    "world_history": [
        ("The French Revolution began in", "A) 1776\nB) 1789\nC) 1804\nD) 1815", "B"),
        ("Treaty of Westphalia (1648) ended", "A) Hundred Years War\nB) Seven Years War\nC) Thirty Years War\nD) War of Spanish Succession", "C"),
        ("Industrial Revolution originated in", "A) France\nB) Germany\nC) United States\nD) Great Britain", "D"),
        ("The Berlin Wall fell in", "A) 1985\nB) 1989\nC) 1991\nD) 1993", "B"),
        ("Ottoman Empire fell after", "A) World War I\nB) World War II\nC) Korean War\nD) Napoleonic Wars", "A"),
        ("Magna Carta was signed in", "A) 1066\nB) 1215\nC) 1453\nD) 1492", "B"),
    ],
    "physics": [
        ("F = ma is Newton's", "A) First law\nB) Second law\nC) Third law\nD) Law of gravitation", "B"),
        ("Speed of light approximately", "A) 3x10^6 m/s\nB) 3x10^8 m/s\nC) 3x10^10 m/s\nD) 3x10^12 m/s", "B"),
        ("Relativistic mass as object approaches light speed", "A) Decreases\nB) Stays same\nC) Increases\nD) Becomes zero", "C"),
        ("Entropy in isolated system", "A) Decreases\nB) Stays constant\nC) Increases or stays constant\nD) Oscillates", "C"),
        ("Photoelectric effect demonstrates light has", "A) Wave nature\nB) Particle nature\nC) No mass\nD) Infinite speed", "B"),
        ("Heisenberg limits measurement of", "A) Mass and charge\nB) Position and momentum\nC) Energy and time\nD) Both B and C", "D"),
    ],
    "biology": [
        ("DNA replication is", "A) Conservative\nB) Dispersive\nC) Semi-conservative\nD) Non-conservative", "C"),
        ("Powerhouse of the cell is", "A) Nucleus\nB) Ribosome\nC) Mitochondria\nD) Golgi", "C"),
        ("Photosynthesis occurs in", "A) Mitochondria\nB) Chloroplasts\nC) Nucleus\nD) Cell membrane", "B"),
        ("NOT a nucleotide base in DNA?", "A) Adenine\nB) Uracil\nC) Guanine\nD) Thymine", "B"),
        ("mRNA to protein is called", "A) Transcription\nB) Translation\nC) Replication\nD) Transformation", "B"),
        ("Universal donor blood type?", "A) A\nB) B\nC) AB\nD) O", "D"),
    ],
    "mathematics": [
        ("Derivative of x^3", "A) x^2\nB) 3x^2\nC) 3x\nD) x^3/3", "B"),
        ("Integral of 1/x", "A) x\nB) ln|x| + C\nC) 1/x^2\nD) x^2/2", "B"),
        ("Determinant of [[a,b],[c,d]]", "A) ab-cd\nB) ad-bc\nC) ac-bd\nD) ab+cd", "B"),
        ("lim(x->0) sin(x)/x", "A) 0\nB) 1\nC) infinity\nD) undefined", "B"),
        ("Sum of interior angles of hexagon", "A) 540\nB) 720\nC) 900\nD) 1080", "B"),
    ],
    "chemistry": [
        ("Water's formula", "A) H2O\nB) CO2\nC) NaCl\nD) O2", "A"),
        ("pH of neutral solution", "A) 0\nB) 5\nC) 7\nD) 14", "C"),
        ("Noble gas with 8 electrons", "A) Helium\nB) Neon\nC) Argon\nD) Krypton", "B"),
        ("Bond involving sharing electrons", "A) Ionic\nB) Covalent\nC) Metallic\nD) Hydrogen", "B"),
    ],
    "economics": [
        ("When demand increases, supply constant, price", "A) Decreases\nB) Same\nC) Increases\nD) Zero", "C"),
        ("GDP stands for", "A) Gross Direct Product\nB) Gross Domestic Product\nC) General Domestic Product\nD) Gross Domestic Price", "B"),
        ("Inflation measured by", "A) GDP\nB) CPI\nC) GNP\nD) PPP", "B"),
    ],
}


def run_mmlu(model, tokenizer, model_short, device, cr=3.0):
    print(f"\n{'='*70}")
    print(f"  CORRECTED MMLU: {model_short} @ {cr}x")
    print(f"{'='*70}")

    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads

    methods = ["full_kv", "h2o_uniform", "cake", "kvtuner", "kivi_uniform", "layer_budget"]
    correct = {m: 0 for m in methods}
    total = {m: 0 for m in methods}

    all_q = [(subj, q, c, a) for subj, qs in MMLU_QUESTIONS.items() for q, c, a in qs]
    print(f"  {len(all_q)} questions, {len(MMLU_QUESTIONS)} subjects")

    answer_tokens = None
    for qi, (subj, q_text, choices, answer) in enumerate(all_q):
        if qi % 10 == 0:
            print(f"  [{qi}/{len(all_q)}]", flush=True)

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

        with torch.no_grad():
            out = model(input_ids=ids[:, :pl], output_attentions=True, return_dict=True)
        fk, fv = hf_to_deltacache(out.past_key_values)
        attn = list(out.attentions)
        del out

        for m in methods:
            try:
                if m == "full_kv":
                    ly = [(fk[l:l+1], fv[l:l+1], torch.arange(pl)) for l in range(nl)]
                elif m == "layer_budget":
                    profiler = LayerAttentionProfiler()
                    allocator = LayerBudgetAllocator(nl, nh, hd)
                    store = LayerKVStore(nl, nh, hd)
                    pr = profiler.profile_from_attention_weights(attn)
                    sp = pr.gini_scores()
                    imp = allocator.compute_importance_weights(nl)
                    fm = allocator.full_memory(pl)
                    alloc = allocator.allocate(sp, imp, int(fm / cr), pl)
                    store.store_from_full_cache(fk, fv, alloc.allocations)
                    ly = store.get_all_layers()
                elif m in REGISTRY:
                    cls = REGISTRY[m]
                    bl = cls(nl, nh, hd)
                    kw = {"attention_weights": attn} if cls.requires_attention else {}
                    ly = bl.compress(fk, fv, cr, **kw)
                else:
                    continue

                # CORRECTED: zero-fill
                pkv = build_cache_CORRECTED(ly, pl, device)
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
        del fk, fv, attn; clear_gpu()

    print(f"\n  {'Method':20s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>8s}")
    print(f"  {'-'*45}")
    rows = []
    for m in methods:
        acc = correct[m] / total[m] if total[m] > 0 else 0
        print(f"  {m:20s} {correct[m]:>8d} {total[m]:>6d} {acc:>8.1%}")
        rows.append({"method": m, "correct": correct[m], "total": total[m], "accuracy": round(acc, 4)})

    path = RESULTS_DIR / f"corrected_mmlu_{model_short.lower().replace('-', '_')}.json"
    with open(path, "w") as f:
        json.dump({"metadata": {"model": model_short, "cr": cr, "n_questions": len(all_q),
                                "eval": "CORRECTED_zero_fill",
                                "timestamp": datetime.now().isoformat()}, "results": rows}, f, indent=2)
    print(f"  Saved: {path.name}")


if __name__ == "__main__":
    models = [
        ("meta-llama/Llama-2-7b-chat-hf", "Llama-2-7B"),
        ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B"),
    ]
    for model_name, model_short in models:
        model, tokenizer = load_model_4bit(model_name)
        device = next(model.parameters()).device
        run_ablation(model, tokenizer, model_short, seq_len=512, cr=3.0)
        run_mmlu(model, tokenizer, model_short, device, cr=3.0)
        del model, tokenizer; clear_gpu()
    print("\n\nAll corrected ablation + MMLU complete!")
