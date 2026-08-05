"""Per-model F(b) fidelity calibration.

For each model in the eval suite, compute mean cosine similarity between
FP16 KV tensors and their INT8 / INT4 quantized counterparts, averaged
across layers and a small WikiText-2 calibration corpus.

Outputs results/f_b_calibration.json with per-model {F(8), F(4)} values
to compare against the Mistral-7B headline (F(8)=0.9999, F(4)=0.9964).

Run: python f_b_calibration.py
GPU: ~0.5h on A100, ~1h on RTX 3090.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.2", "mistral_7b"),  # baseline anchor
    ("meta-llama/Llama-2-7b-chat-hf", "llama2_7b"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama3.1_8b"),
    ("Qwen/Qwen3-8B", "qwen3_8b"),
    ("meta-llama/Llama-2-13b-chat-hf", "llama2_13b"),
    ("Qwen/Qwen2.5-14B-Instruct", "qwen2.5_14b"),
    # 72B intentionally skipped — fidelity should not depend on scale
]

NUM_CALIB_SAMPLES = 8
CALIB_SEQ_LEN = 512
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def cosine_per_token(x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean cosine similarity over the last dim (token-major)."""
    x_flat = x.reshape(-1, x.shape[-1]).float()
    y_flat = y.reshape(-1, y.shape[-1]).float()
    return float(F.cosine_similarity(x_flat, y_flat, dim=-1).mean().item())


def calibrate_one_model(hf_name: str, short: str) -> dict:
    print(f"\n=== {short} ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, quantization_config=bnb, device_map={"": "cuda:0"},
        attn_implementation="eager", trust_remote_code=True,
    )
    model.eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t) > 200][:NUM_CALIB_SAMPLES]

    fp16_int8_sims = []
    fp16_int4_sims = []
    for txt in texts:
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=CALIB_SEQ_LEN).input_ids
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model(ids, use_cache=True, output_attentions=False)
        # out.past_key_values: tuple of (key, value) per layer
        for k, v in out.past_key_values:
            for prec, store in [(QuantPrecision.INT8, fp16_int8_sims), (QuantPrecision.INT4, fp16_int4_sims)]:
                q = KVQuantizer(precision=prec)
                qkv = q.quantize(k.contiguous(), v.contiguous())
                k_dq, v_dq = q.dequantize(qkv)
                # average key + value cos sim
                cos_k = cosine_per_token(k, k_dq)
                cos_v = cosine_per_token(v, v_dq)
                store.append(0.5 * (cos_k + cos_v))

    F8 = sum(fp16_int8_sims) / len(fp16_int8_sims)
    F4 = sum(fp16_int4_sims) / len(fp16_int4_sims)
    elapsed = time.time() - t0
    print(f"  F(8)={F8:.4f}  F(4)={F4:.4f}  ({elapsed:.0f}s, {len(fp16_int8_sims)} layer-samples)")

    del model
    torch.cuda.empty_cache()
    return {"hf_name": hf_name, "short": short, "F8": F8, "F4": F4, "n_samples": len(fp16_int8_sims)}


def main() -> None:
    out = {"calibration_corpus": "wikitext-2-raw-v1 test", "seq_len": CALIB_SEQ_LEN, "num_samples": NUM_CALIB_SAMPLES, "models": []}
    for hf_name, short in MODELS:
        try:
            out["models"].append(calibrate_one_model(hf_name, short))
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {short}: {e}")
            out["models"].append({"short": short, "error": str(e)})
    out_path = RESULT_DIR / "f_b_calibration.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    # Pretty summary
    print("\n=== Summary (compare to Mistral baseline F(8)=0.9999, F(4)=0.9964) ===")
    print(f"{'Model':<20} {'F(8)':>8} {'F(4)':>8}  Δ-from-Mistral")
    for m in out["models"]:
        if "error" in m:
            continue
        df8 = m["F8"] - 0.9999
        df4 = m["F4"] - 0.9964
        print(f"{m['short']:<20} {m['F8']:>8.4f} {m['F4']:>8.4f}  Δ8={df8:+.4f} Δ4={df4:+.4f}")


if __name__ == "__main__":
    main()
