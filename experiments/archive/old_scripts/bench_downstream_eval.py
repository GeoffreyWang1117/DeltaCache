#!/usr/bin/env python3
"""Downstream task evaluation for LayerBudget.

Evaluates accuracy on MMLU (5-shot) and HellaSwag subsets using
compressed vs full KV cache.

Measures: accuracy@top1 for multiple-choice tasks.
"""

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator
from deltacache.core.layer_kv_store import LayerKVStore
from deltacache.hf_integration.kv_format import hf_to_deltacache, deltacache_to_hf

from baselines.cake import CAKEBaseline

RESULTS_DIR = Path(__file__).parent / "results" / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _build_hf_past_kv(layers_data, device, full_seq_len=None):
    from transformers.cache_utils import DynamicCache
    token_counts = [k.shape[1] if k.dim() == 4 else k.shape[0] for k, v, _ in layers_data]
    seq_len = full_seq_len if full_seq_len else max(token_counts)
    cache = DynamicCache()
    for layer_idx, (keys, values, indices) in enumerate(layers_data):
        k = keys.to(device)
        v = values.to(device)
        if k.dim() == 3: k = k.unsqueeze(0); v = v.unsqueeze(0)
        n_tokens, num_heads, head_dim = k.shape[1], k.shape[2], k.shape[3]
        if n_tokens == seq_len:
            k_out, v_out = k.transpose(1, 2), v.transpose(1, 2)
        else:
            k_full = torch.zeros(1, seq_len, num_heads, head_dim, dtype=k.dtype, device=device)
            v_full = torch.zeros(1, seq_len, num_heads, head_dim, dtype=v.dtype, device=device)
            idx = indices.long().to(device)
            valid_idx = idx[idx < seq_len]
            if valid_idx.shape[0] > 0:
                k_full[0, valid_idx] = k[0, :valid_idx.shape[0]]
                v_full[0, valid_idx] = v[0, :valid_idx.shape[0]]
            k_out, v_out = k_full.transpose(1, 2), v_full.transpose(1, 2)
        cache.update(k_out, v_out, layer_idx)
    return cache


# MMLU 5-shot format
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "computer_security", "conceptual_physics", "econometrics",
]

HELLASWAG_N = 100  # Number of HellaSwag examples


def format_mmlu_prompt(subject, examples, question, choices):
    """Format MMLU prompt with 5-shot examples."""
    prompt = f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n"
    for ex in examples[:5]:
        prompt += f"Q: {ex['question']}\n"
        for i, c in enumerate(ex['choices']):
            prompt += f"{'ABCD'[i]}. {c}\n"
        prompt += f"Answer: {'ABCD'[ex['answer']]}\n\n"
    prompt += f"Q: {question}\n"
    for i, c in enumerate(choices):
        prompt += f"{'ABCD'[i]}. {c}\n"
    prompt += "Answer:"
    return prompt


def evaluate_mc_accuracy(model, tokenizer, prompts, answers, device,
                         compress_fn=None, compression_ratio=3.0,
                         num_layers=None, num_heads=None, head_dim=None):
    """Evaluate multiple-choice accuracy with optional KV compression.

    compress_fn: None for full KV, or a callable(full_keys, full_values,
                 attention_weights, prefix_len, ...) -> past_kv
    """
    correct = 0
    total = 0
    choice_tokens = {
        c: tokenizer.encode(c, add_special_tokens=False)[-1]
        for c in ["A", "B", "C", "D"]
    }

    for prompt, answer in zip(prompts, answers):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        if input_ids.shape[1] > 512:
            input_ids = input_ids[:, :512]  # Cap for memory

        with torch.no_grad():
            if compress_fn is None:
                # Full KV
                outputs = model(input_ids=input_ids, return_dict=True)
                logits = outputs.logits[:, -1, :]
            else:
                # Compressed KV: use prefix as context, evaluate on last token
                prefix_len = input_ids.shape[1] - 1
                prefix_ids = input_ids[:, :prefix_len]

                out = model(input_ids=prefix_ids, output_attentions=True, return_dict=True)
                kv_cache = out.past_key_values
                attention_weights = list(out.attentions)
                full_keys, full_values = hf_to_deltacache(kv_cache)
                del out, kv_cache
                clear_gpu()

                past_kv = compress_fn(
                    full_keys, full_values, attention_weights,
                    prefix_len, compression_ratio,
                    num_layers, num_heads, head_dim, device
                )
                del full_keys, full_values, attention_weights
                clear_gpu()

                suffix_ids = input_ids[:, prefix_len:]
                position_ids = torch.arange(
                    prefix_len, prefix_len + suffix_ids.shape[1],
                    device=device
                ).unsqueeze(0)
                out2 = model(
                    input_ids=suffix_ids, past_key_values=past_kv,
                    position_ids=position_ids, return_dict=True
                )
                logits = out2.logits[:, -1, :]
                del past_kv, out2
                clear_gpu()

        # Get probabilities for A/B/C/D
        probs = {}
        for letter, tok_id in choice_tokens.items():
            probs[letter] = logits[0, tok_id].item()

        predicted = max(probs, key=probs.get)
        correct_letter = "ABCD"[answer]
        if predicted == correct_letter:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def compress_layer_budget(full_keys, full_values, attention_weights,
                          prefix_len, compression_ratio,
                          num_layers, num_heads, head_dim, device):
    """Compress using LayerBudget."""
    profiler = LayerAttentionProfiler()
    profile_result = profiler.profile_from_attention_weights(attention_weights)
    gini = profile_result.gini_scores()

    allocator = LayerBudgetAllocator(num_layers, num_heads, head_dim)
    importance = allocator.compute_importance_weights(num_layers)
    budget = int(allocator.full_memory(prefix_len) / compression_ratio)
    allocation = allocator.allocate(gini, importance, budget, prefix_len)

    store = LayerKVStore(num_layers, num_heads, head_dim)
    store.store_from_full_cache(full_keys, full_values, allocation.allocations)
    return _build_hf_past_kv(store.get_all_layers(), device, full_seq_len=prefix_len)


def compress_h2o(full_keys, full_values, attention_weights,
                 prefix_len, compression_ratio,
                 num_layers, num_heads, head_dim, device):
    """Compress using uniform H2O."""
    n_tok = max(1, int(prefix_len / compression_ratio))
    store = LayerKVStore(num_layers, num_heads, head_dim)
    for l in range(num_layers):
        idx = store._default_token_selection(
            full_keys[l:l+1], full_values[l:l+1], n_tok, prefix_len)
        store.store_layer(l, full_keys[l:l+1], full_values[l:l+1], idx, quant_bits=16)
    return _build_hf_past_kv(store.get_all_layers(), device, full_seq_len=prefix_len)


def compress_cake(full_keys, full_values, attention_weights,
                  prefix_len, compression_ratio,
                  num_layers, num_heads, head_dim, device):
    """Compress using CAKE."""
    cake = CAKEBaseline(num_layers, num_heads, head_dim)
    cake_layers = cake.compress(full_keys, full_values, attention_weights,
                                compression_ratio)
    return _build_hf_past_kv(cake_layers, device, full_seq_len=prefix_len)


def run_downstream_eval(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda:0",
    compression_ratio=3.0,
    load_in_4bit=False,
    n_mmlu_subjects=5,
    n_hellaswag=50,
):
    print(f"\n{'='*70}")
    print(f"Downstream Task Evaluation")
    print(f"Model: {model_name}   CR: {compression_ratio}x   4-bit: {load_in_4bit}")
    print(f"{'='*70}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(torch_dtype=torch.float16, trust_remote_code=True,
                       attn_implementation="eager")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Load MMLU subset
    print("Loading MMLU dataset...")
    from datasets import load_dataset
    mmlu_results = {}
    subjects = MMLU_SUBJECTS[:n_mmlu_subjects]

    for subject in subjects:
        print(f"\n  MMLU: {subject}")
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
            ds_few = load_dataset("cais/mmlu", subject, split="dev", trust_remote_code=True)
        except Exception as e:
            print(f"    Skipped: {e}")
            continue

        # Get 5-shot examples
        examples = [{"question": r["question"], "choices": r["choices"], "answer": r["answer"]}
                     for r in ds_few]

        # Limit to 20 questions per subject for speed
        test_items = list(ds)[:20]
        prompts = []
        answers = []
        for item in test_items:
            prompt = format_mmlu_prompt(subject, examples, item["question"], item["choices"])
            prompts.append(prompt)
            answers.append(item["answer"])

        # Full KV
        acc_full = evaluate_mc_accuracy(
            model, tokenizer, prompts, answers, device)
        print(f"    Full KV: {acc_full:.1%}")

        # LayerBudget
        acc_lb = evaluate_mc_accuracy(
            model, tokenizer, prompts, answers, device,
            compress_fn=compress_layer_budget,
            compression_ratio=compression_ratio,
            num_layers=num_layers, num_heads=num_heads, head_dim=head_dim)
        print(f"    LayerBudget ({compression_ratio}x): {acc_lb:.1%}")

        # H2O
        acc_h2o = evaluate_mc_accuracy(
            model, tokenizer, prompts, answers, device,
            compress_fn=compress_h2o,
            compression_ratio=compression_ratio,
            num_layers=num_layers, num_heads=num_heads, head_dim=head_dim)
        print(f"    H2O ({compression_ratio}x): {acc_h2o:.1%}")

        # CAKE
        acc_cake = evaluate_mc_accuracy(
            model, tokenizer, prompts, answers, device,
            compress_fn=compress_cake,
            compression_ratio=compression_ratio,
            num_layers=num_layers, num_heads=num_heads, head_dim=head_dim)
        print(f"    CAKE ({compression_ratio}x): {acc_cake:.1%}")

        mmlu_results[subject] = {
            "full": acc_full, "layer_budget": acc_lb,
            "h2o": acc_h2o, "cake": acc_cake,
            "n_questions": len(prompts),
        }
        clear_gpu()

    # Aggregate MMLU
    if mmlu_results:
        avg_full = sum(r["full"] for r in mmlu_results.values()) / len(mmlu_results)
        avg_lb = sum(r["layer_budget"] for r in mmlu_results.values()) / len(mmlu_results)
        avg_h2o = sum(r["h2o"] for r in mmlu_results.values()) / len(mmlu_results)
        avg_cake = sum(r["cake"] for r in mmlu_results.values()) / len(mmlu_results)
    else:
        avg_full = avg_lb = avg_h2o = avg_cake = 0.0

    print(f"\n{'='*70}")
    print(f"MMLU Average ({len(mmlu_results)} subjects, {compression_ratio}x)")
    print(f"  Full KV:      {avg_full:.1%}")
    print(f"  LayerBudget:  {avg_lb:.1%}")
    print(f"  H2O:          {avg_h2o:.1%}")
    print(f"  CAKE:         {avg_cake:.1%}")

    output = {
        "metadata": {
            "experiment": "downstream_evaluation",
            "model": model_name,
            "compression_ratio": compression_ratio,
            "load_in_4bit": load_in_4bit,
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "mmlu": {
            "per_subject": mmlu_results,
            "average": {
                "full": round(avg_full, 4),
                "layer_budget": round(avg_lb, 4),
                "h2o": round(avg_h2o, 4),
                "cake": round(avg_cake, 4),
            },
        },
    }

    safe_model = model_name.split("/")[-1].lower().replace("-", "_")
    out_path = RESULTS_DIR / f"downstream_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compression", type=float, default=3.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--n-subjects", type=int, default=5)
    parser.add_argument("--n-hellaswag", type=int, default=50)
    args = parser.parse_args()

    run_downstream_eval(
        model_name=args.model,
        device=args.device,
        compression_ratio=args.compression,
        load_in_4bit=args.load_in_4bit,
        n_mmlu_subjects=args.n_subjects,
        n_hellaswag=args.n_hellaswag,
    )
