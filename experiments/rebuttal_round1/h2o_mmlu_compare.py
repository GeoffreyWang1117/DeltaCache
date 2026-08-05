"""MMLU comparison: h2o_uniform (current) vs h2o_faithful (per-head, recent_ratio=0.1) on Llama-2-7B at CR=4×.

Goal: determine whether our reimpl's 36.5% MMLU is faithful to the official H2O algorithm
(which uses per-head heavy-hitter selection + 0.1×seq recent window).

Strategy: load Llama-2-7B 4-bit, run MMLU directly (avoiding full suite plumbing).
For each question:
  1. Forward + extract attention
  2. Compress with h2o_uniform → predict → record
  3. Compress with h2o_faithful → predict → record
  4. (sanity) full-KV → predict → record

Subset: 200 MMLU questions across all 57 subjects (3-4 per subject, balanced).

Outputs results/h2o_mmlu_compare.json with per-method accuracy.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DELTACACHE_ROOT = Path("/home/coder-gw/Projects/DeltaCache")
sys.path.insert(0, str(DELTACACHE_ROOT))
sys.path.insert(0, str(DELTACACHE_ROOT / "experiments"))

import torch  # noqa: E402

# Import order: register h2o_faithful FIRST so the registry contains it
import h2o_faithful  # noqa: F401, E402

from baselines.base import REGISTRY  # noqa: E402
from suite.eval_utils import compress_kv, extract_kv_and_attention, build_hf_cache, compute_next_token_logits  # noqa: E402
from suite.tasks.mmlu import MMLU_CHOICES, format_mmlu_question  # noqa: E402

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Available methods: {sorted(REGISTRY.keys())}")
print(f"h2o_faithful registered: {'h2o_faithful' in REGISTRY}")

# --- Load model ---
print("\nLoading Llama-2-7B 4-bit on cuda:0...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

HF_NAME = "meta-llama/Llama-2-7b-chat-hf"
tok = AutoTokenizer.from_pretrained(HF_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    HF_NAME, quantization_config=bnb, device_map={"": "cuda:0"},
    attn_implementation="eager", trust_remote_code=True,
)
model.eval()
device = next(model.parameters()).device

cfg = model.config
nl, nh, hd = cfg.num_hidden_layers, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads), cfg.hidden_size // cfg.num_attention_heads
print(f"Model loaded. nl={nl}, nh={nh}, hd={hd}")

# --- Load MMLU ---
print("\nLoading MMLU...")
from datasets import load_dataset  # noqa: E402

ds = load_dataset("cais/mmlu", "all", split="test")
# Group by subject
by_subject = defaultdict(list)
for item in ds:
    by_subject[item["subject"]].append(item)
print(f"  {len(by_subject)} subjects, {sum(len(v) for v in by_subject.values())} total Q.")

# Subset: 4 questions per subject = 228 total
N_PER_SUBJECT = 4
SUBSET = []
import random
random.seed(42)
for subject, items in sorted(by_subject.items()):
    SUBSET.extend(random.sample(items, min(N_PER_SUBJECT, len(items))))
print(f"  using subset of {len(SUBSET)} questions ({N_PER_SUBJECT} per subject)")

# Token IDs for A, B, C, D
choice_ids = [tok.encode(c, add_special_tokens=False)[-1] for c in MMLU_CHOICES]

# --- MMLU loop with two methods ---
methods = ["full_kv", "h2o_uniform", "h2o_faithful", "kivi_uniform"]
CR = 4.0
results = {m: {"correct": 0, "total": 0, "per_subject": defaultdict(lambda: {"c": 0, "t": 0})} for m in methods}

def reformat_q_dict(item: dict, subject: str) -> dict:
    """Convert HF MMLU item to suite format."""
    return {
        "subject": subject,
        "question": item["question"],
        "choices": item["choices"],
        "answer": item["answer"],
    }

t0 = time.time()
for i, item in enumerate(SUBSET):
    subject = item["subject"]
    q_dict = reformat_q_dict(item, subject)
    prompt = format_mmlu_question(q_dict, subject)
    input_ids = tok.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    seq_len = input_ids.shape[1]

    # Forward to get full KV + attention
    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True, return_dict=True)
    # KV: out.past_key_values is tuple; convert to (nl, S, H, D) layout used by compress_kv
    full_k = torch.stack([kv[0].squeeze(0).transpose(0, 1) for kv in out.past_key_values])  # (nl, S, H, D)
    full_v = torch.stack([kv[1].squeeze(0).transpose(0, 1) for kv in out.past_key_values])
    attns = list(out.attentions)
    del out

    answer = item["answer"]

    for method in methods:
        try:
            layers, _, _ = compress_kv(
                method, full_k, full_v, CR, nl, nh, hd,
                attention_weights=attns,
            )
            cache = build_hf_cache(layers, seq_len, device, fill="mean")
            logits = compute_next_token_logits(model, input_ids, cache)
            choice_logits = logits[0, choice_ids]
            pred = int(choice_logits.argmax().item())
            ok = int(pred == answer)
            results[method]["correct"] += ok
            results[method]["total"] += 1
            results[method]["per_subject"][subject]["c"] += ok
            results[method]["per_subject"][subject]["t"] += 1
            del cache, logits
        except Exception as e:  # noqa: BLE001
            print(f"  [Q{i}] {method} FAILED: {type(e).__name__}: {e}")

    if (i + 1) % 20 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(SUBSET) - i - 1)
        msg = "  ".join(f"{m}={results[m]['correct']}/{results[m]['total']}" for m in methods)
        print(f"[{i+1}/{len(SUBSET)}] elapsed {elapsed:.0f}s ETA {eta:.0f}s  {msg}", flush=True)

    del full_k, full_v, attns
    torch.cuda.empty_cache()

# --- Summary ---
elapsed = time.time() - t0
print(f"\n=== Done in {elapsed:.0f}s ===")
print(f"{'Method':<18} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
for m in methods:
    r = results[m]
    acc = r["correct"] / max(r["total"], 1)
    print(f"{m:<18} {r['correct']:>8} {r['total']:>6} {acc:>8.4f}")

# Save (convert defaultdict for JSON)
out = {
    "model": HF_NAME, "compression_ratio": CR, "n_questions": len(SUBSET),
    "n_per_subject": N_PER_SUBJECT, "elapsed_s": round(elapsed, 1),
    "methods": {m: {"correct": r["correct"], "total": r["total"], "accuracy": r["correct"] / max(r["total"], 1),
                    "per_subject": {s: dict(d) for s, d in r["per_subject"].items()}}
                for m, r in results.items()},
    "timestamp": datetime.now().isoformat(),
}
out_path = OUT_DIR / f"h2o_mmlu_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nWrote {out_path}")
