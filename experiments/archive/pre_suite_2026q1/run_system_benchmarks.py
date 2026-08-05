#!/usr/bin/env python3
"""#20: System benchmarks — latency/throughput/memory curves.

Measures 5 metrics across seq_len 256/512/1024/2048:
  - Peak GPU memory
  - Prefill latency
  - Decode latency (per token)
  - Tokens/sec
  - End-to-end throughput

Three deployment modes:
  A: Fixed profile (zero marginal cost)
  B: Online profiling (full overhead)
  C: No compression (baseline)

Uses inverted importance on Mistral-7B.
"""

import argparse, gc, json, math, statistics, sys, time
from datetime import datetime
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deltacache.integrations.hf_cache import LayerBudgetCache

RESULTS_DIR = Path(__file__).parent / "results" / "system"
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


def get_wikitext_text(tokenizer, target_len):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = " ".join([t for t in ds["text"] if len(t.strip()) > 100])
    all_ids = tokenizer.encode(all_text)
    return all_ids[:target_len]


def run_benchmarks(model, tokenizer, model_short):
    print(f"\n{'='*70}")
    print(f"  #20: System Benchmarks ({model_short})")
    print(f"{'='*70}")

    device = next(model.parameters()).device
    nl = model.config.num_hidden_layers
    nh = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hd = model.config.hidden_size // model.config.num_attention_heads
    max_pos = min(getattr(model.config, "max_position_embeddings", 4096), 2048)

    seq_lens = [s for s in [256, 512, 1024, 2048] if s <= max_pos]
    gen_tokens = 32

    # Compute auto budget for inverted importance
    kv_per_token = 2 * nl * nh * hd * 2
    full_kv_1024_mb = kv_per_token * 1024 / (1024**2)
    budget_mb = max(1.0, round(full_kv_1024_mb * 0.25, 1))

    print(f"  Config: {nl}L {nh}H {hd}D, budget={budget_mb}MB, gen={gen_tokens}tok")
    print(f"  Seq lens: {seq_lens}")

    gen_kw = dict(max_new_tokens=gen_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    results = []

    # Warmup
    warm_ids = tokenizer.encode("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(warm_ids, max_new_tokens=4, pad_token_id=tokenizer.eos_token_id)
    clear_gpu()

    for sl in seq_lens:
        tokens = get_wikitext_text(tokenizer, sl)
        input_ids = torch.tensor([tokens], device=device)
        actual_len = input_ids.shape[1]

        for mode in ["baseline", "layerbudget"]:
            clear_gpu()
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / 1024**2

            # Prefill timing
            t_start = time.perf_counter()
            with torch.no_grad():
                if mode == "baseline":
                    out = model.generate(input_ids, **gen_kw)
                else:
                    cache = LayerBudgetCache(max_memory_mb=budget_mb, compression_ratio=0.25)
                    out = model.generate(input_ids, past_key_values=cache, **gen_kw)
            t_total = time.perf_counter() - t_start

            peak_mem = torch.cuda.max_memory_allocated() / 1024**2 - mem_before
            gen_len = out.shape[1] - actual_len
            total_ms = t_total * 1000
            per_token_ms = total_ms / max(gen_len, 1)
            tokens_per_sec = gen_len / max(t_total, 1e-6)

            compressed = getattr(cache, '_compressed', False) if mode == "layerbudget" else False

            row = {
                "seq_len": actual_len, "mode": mode,
                "total_ms": round(total_ms, 1),
                "per_token_ms": round(per_token_ms, 1),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "peak_mem_mb": round(peak_mem, 1),
                "gen_tokens": gen_len,
                "compressed": compressed,
            }
            results.append(row)

        clear_gpu()

    # Print summary
    print(f"\n  {'SeqLen':>7s} {'Mode':>14s} {'Total(ms)':>10s} {'ms/tok':>8s} {'tok/s':>8s} {'PeakMem':>9s} {'Compr':>6s}")
    print(f"  {'-'*65}")
    for sl in seq_lens:
        for mode in ["baseline", "layerbudget"]:
            rows = [r for r in results if r["seq_len"] == sl and r["mode"] == mode]
            if not rows: continue
            r = rows[0]
            print(f"  {r['seq_len']:>7d} {mode:>14s} {r['total_ms']:>9.1f}ms {r['per_token_ms']:>7.1f} "
                  f"{r['tokens_per_sec']:>7.1f} {r['peak_mem_mb']:>8.1f}MB {str(r['compressed']):>6s}")
        print()

    # Compute speedup/savings
    print(f"  {'SeqLen':>7s} {'Latency':>10s} {'MemSave':>9s}")
    print(f"  {'-'*30}")
    for sl in seq_lens:
        base = [r for r in results if r["seq_len"] == sl and r["mode"] == "baseline"]
        lb = [r for r in results if r["seq_len"] == sl and r["mode"] == "layerbudget"]
        if base and lb:
            speedup = base[0]["total_ms"] / max(lb[0]["total_ms"], 0.1)
            mem_save = 1.0 - lb[0]["peak_mem_mb"] / max(base[0]["peak_mem_mb"], 0.1)
            print(f"  {sl:>7d} {speedup:>9.2f}x {mem_save:>8.1%}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--model-short", default=None)
    args = parser.parse_args()

    model_short = args.model_short or args.model.split("/")[-1]
    model, tokenizer = load_model_4bit(args.model)
    results = run_benchmarks(model, tokenizer, model_short)

    safe = model_short.replace("/", "_").replace("-", "_").lower()
    path = RESULTS_DIR / f"system_{safe}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w") as f:
        json.dump({"metadata": {"model": args.model, "model_short": model_short,
                                "timestamp": datetime.now().isoformat(),
                                "gpu": torch.cuda.get_device_name(0)},
                   "results": results}, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
