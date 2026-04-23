# LayerBudget

**Per-layer joint token-precision optimization for KV cache compression in LLM inference.**

LayerBudget jointly allocates per-layer token retention budgets and quantization precision for KV cache compression. Three design principles drive its effectiveness:

1. **Quantize first, evict last** — the greedy allocator exhausts INT4/INT8 quantization headroom before evicting any tokens, because quantization preserves attention structure while eviction disrupts it.
2. **Protect early layers** — leave-one-out analysis reveals early transformer layers are the primary quality bottleneck under eviction. Inverted importance weights (early-high) improve 6x compression by 60–94% over conventional late-high weighting.
3. **Mean-fill** — evicted positions are filled with the mean of retained tokens, reducing output KL divergence by 96%.

## Key Results (NeurIPS 2026 experiment suite — 1423 checkpoints, 6 models, 17 baselines)

### PPL Ratio @ 1024 tokens

| Compression | LayerBudget | H2O | SnapKV | StreamingLLM | KIVI |
|:-----------:|:-----------:|:---:|:------:|:------------:|:----:|
| 2x | **0.999–1.001** | 1.02–1.17 | 1.06–1.12 | 1.17–1.29 | 1.01–1.02 |
| 4x | **1.01–1.03** | 1.07–1.21 | 1.13–1.24 | 1.28–1.41 | 1.01–1.02 |
| 6x | **1.03–1.25** | 1.11–1.24 | 1.18–1.35 | 1.32–1.64 | 1.01–1.02 |

### Long Context (16K–32K tokens, Mistral-7B)

| Method | 16K CR=2x | 16K CR=4x | 32K CR=2x | 32K CR=4x |
|--------|:---------:|:---------:|:---------:|:---------:|
| **LayerBudget** | **1.001** | **1.006** | **1.000** | **1.009** |
| KIVI | 1.006 | 1.006 | 1.008 | 1.008 |
| H2O | 1.039 | **34.9** | 1.057 | **83.4** |
| StreamingLLM | 1.013 | 1.035 | 1.014 | 1.026 |

### Downstream Tasks

| Benchmark | Full KV | LB@4x | H2O@4x | KIVI@4x |
|-----------|:-------:|:-----:|:-------:|:-------:|
| MMLU (1140q, 4 models) | 47–73% | **47–73%** | 36–52% | 47–73% |
| NIAH@4096 (4 models) | 87–100% | **73–100%** | 40–100% | 73–100% |
| LongBench (16 tasks) | 0.037–0.040 | **0.038–0.039** | 0.032–0.036 | 0.032–0.037 |

### System Metrics (KV cache memory savings)

| Architecture | Memory Savings |
|:------------:|:--------------:|
| GQA 4-head (Qwen2.5-14B) | **72%** |
| GQA 8-head (Mistral/Llama-3.1/Qwen3) | **43–72%** |
| MHA 32-head (Llama-2-7B) | **24%** |

Evaluated on 6 models (7B–14B): Llama-2-7B, Llama-2-13B, Llama-3.1-8B, Mistral-7B, Qwen3-8B, Qwen2.5-14B.
17 baselines: H2O, KIVI, SnapKV, StreamingLLM, MiniKV, DuoAttention, PyramidKV, D2O, CAKE, AdaKV, DynamicKV, LAVa, EvolKV, KVTuner, SqueezeAttention, XQuant.

## Project Structure

```
deltacache/                            # Core library
  core/
    layer_budget_allocator.py          # Greedy (n_l, b_l) solver, inverted importance default
    layer_profiler.py                  # Online attention Gini profiler (hook-based, O(S) memory)
    layer_kv_store.py                  # Per-layer KV storage with mixed precision
    kv_quantizer.py                    # KIVI-style INT8/INT4 quantization
    memory_monitor.py                  # GPU watermark-based memory tracking
  eviction/                            # Eviction policies (LRU, LFU, layer-aware)
  hf_integration/                      # HuggingFace model adapters + KV format conversion
  integrations/
    hf_cache.py                        # LayerBudgetCache (DynamicCache subclass, drop-in)
  vllm_integration/
    cache_engine.py                    # DeltaCacheEngine (vLLM CacheEngine replacement)
    layer_budget_block_manager.py      # Block-level LayerBudget for PagedAttention
    scheduler_hook.py                  # Prefix-aware scheduling hints

experiments/
  suite/                               # Unified experiment runner (NeurIPS 2026)
    run_all.py                         # CLI entry: phased execution by model size + hardware
    config.py                          # Model zoo, experiment matrix, cost estimation
    runner.py                          # Orchestrator: checkpoint/resume, OOM recovery
    eval_utils.py                      # Shared compress-and-evaluate pipeline
    model_pool.py                      # Smart model loading with Gini profiling
    checkpoint.py                      # Per-unit JSON checkpointing for resume
    tasks/                             # 8 evaluation tasks
      ppl.py                           # Perplexity (main table, 17 baselines)
      mmlu.py                          # MMLU (57 subjects, 1140 questions)
      gsm8k.py                         # GSM8K math reasoning (200 samples)
      math.py                          # MATH competition problems (200 samples)
      longbench.py                     # LongBench (16 English tasks)
      niah.py                          # Needle-In-A-Haystack retrieval
      ruler.py                         # RULER: MKR + variable tracking
      throughput.py                    # Latency / memory / tokens-per-sec
  baselines/                           # 16 baseline implementations (unified interface)
  results/suite/                       # Checkpointed results by run_id

paper/
  main.tex                             # Paper (ICML 2026 format, 9 pages content)
  references.bib                       # 48 references

tests/                                 # 189 unit tests (all passing)
```

## Quick Start

```bash
pip install -e ".[all]"

# End-to-end benchmark on any model
python experiments/bench_e2e_real.py --model Qwen/Qwen2-0.5B
python experiments/bench_e2e_real.py --model mistralai/Mistral-7B-Instruct-v0.2 --load-in-4bit

# Main results table (inverted importance, 12 baselines)
python experiments/run_main_results_v2.py

# Llama-2@1024 + MMLU (hook-based profiler, no OOM)
python experiments/run_1024_and_mmlu.py

# LongBench evaluation
python experiments/run_longbench.py --model mistralai/Mistral-7B-Instruct-v0.2

# RULER-style retrieval
python experiments/run_ruler.py --model mistralai/Mistral-7B-Instruct-v0.2

# Drop-in usage with HuggingFace model.generate()
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltacache.integrations.hf_cache import LayerBudgetCache

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', dtype='float16', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B')

cache = LayerBudgetCache(max_memory_mb=8.0)
inputs = tokenizer('The meaning of life is', return_tensors='pt').to('cuda')
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=50)
print(tokenizer.decode(out[0]))
"

# Run tests
pytest tests/ -v
```

## Design Decisions

### 1. Inverted Importance (early layers protected)

Leave-one-out analysis at 6x on Mistral-7B:
- Restoring layers 0–1: PPL improves by +1.8–2.0 (CRITICAL)
- Restoring layers 14+: near-zero effect

Inverted importance `w_l = sigmoid(k * (1 - l/L - tau))` protects early layers → **60–94% improvement** on both GQA and MHA models.

### 2. Quantize First, Evict Last

At 2–3.5x, the greedy allocator quantizes all layers to INT4/INT8 **without evicting a single token**. Eviction begins only at 4x+ when quantization budget is exhausted.

### 3. Mean-Fill

Evicted positions filled with mean of retained KV:
- Logits KL: 0.249 → 0.009 (**96% reduction**)
- Llama-2 6x PPL ratio: 45.09 → 1.13 (**97.5% reduction**)

### 4. Theoretical Guarantee

Quality function is monotone submodular → greedy achieves (1-1/e) approximation (Sviridenko 2004). Empirically: **118–119%** of 500-trial random search.

## Models Validated (NeurIPS 2026 suite — 1423 OK checkpoints)

| Model | Arch | Params | PPL | MMLU | GSM8K | MATH | LB | NIAH | RULER | Thru | Context |
|-------|:-----|:------:|:---:|:----:|:-----:|:----:|:--:|:----:|:-----:|:----:|:-------:|
| Llama-2-7B | MHA | 7B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 4K |
| Llama-2-13B | MHA | 13B | ✅* | — | — | — | ✅ | ✅ | ✅ | — | 4K |
| Llama-3.1-8B | GQA | 8B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 32K |
| Mistral-7B | GQA | 7B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 32K |
| Qwen3-8B | GQA | 8B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅* | 16K |
| Qwen2.5-14B | GQA | 14B | ✅* | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅* | 4K |

✅ = complete, ✅* = partial (some methods OOM at long seq), — = OOM on 3090

---

## Remaining Work

### Experiments complete (Apr 23, 2026)

- [x] **1423 OK checkpoints** across 6 models, 17 baselines, 8 tasks
- [x] **PPL** — 6 models × 8 methods × 4 CRs × up to 4 seq_lens + 16K/32K long context
- [x] **MMLU** — 5 models × 8 methods × 4 CRs (1140 questions, 57 subjects)
- [x] **GSM8K/MATH** — 5 models × 5 methods × 2 CRs (200 samples each)
- [x] **LongBench** — 6 models × 5 methods × 2 CRs (16 English tasks)
- [x] **NIAH/RULER** — 6 models × 5 methods × 2 CRs × 3 seq_lens
- [x] **Throughput** — 4 models (24–72% KV memory savings)
- [x] **16K/32K context** — 3 GQA models (LB ≤1.001 at 2x, H2O collapses at 83x)
- [x] **13B models** — Llama-2-13B (MHA) + Qwen2.5-14B (GQA)

### Before submission

- [ ] **Regenerate figures** — from 6-model, 17-baseline data
- [ ] **Final paper polish** — update abstract numbers, check page budget

## Baselines (16 methods, all reimplemented)

| Category | Method | Venue | Per-layer |
|----------|--------|-------|:---------:|
| Eviction | H2O | NeurIPS 2023 | No |
| Eviction | SnapKV | NeurIPS 2024 | No |
| Eviction | PyramidKV | COLM 2025 | Yes |
| Eviction | D2O | ICLR 2025 | Yes |
| Eviction | SqueezeAttention | ICLR 2025 | Yes |
| Eviction | CAKE | ICLR 2025 | Yes |
| Eviction | Ada-KV | NeurIPS 2025 | Per-head |
| Eviction | DynamicKV | EMNLP 2025 | Yes |
| Eviction | LAVa | EMNLP 2025 | Yes |
| Eviction | EvolKV | EMNLP 2025 | Yes |
| Per-head | DuoAttention | ICLR 2025 | Per-head |
| Per-head | MiniKV | NeurIPS 2024 | Per-head |
| Quantization | KIVI | ICML 2024 | No |
| Quantization | KVTuner | ICML 2025 | Yes |
| Quantization | XQuant | EMNLP 2025 | Yes |
| **Joint** | **LayerBudget** | — | **Yes** |

## Hardware

- 2x NVIDIA RTX 3090 (24GB), 64GB CPU RAM
- CUDA 12.1, PyTorch 2.x, Transformers 4.x

## Key Files

| File | Purpose |
|------|---------|
| `paper/neurips2026/main.tex` | NeurIPS 2026 paper draft (7.5p content + appendix) |
| `deltacache/core/layer_budget_allocator.py` | Core algorithm (inverted importance default) |
| `deltacache/integrations/hf_cache.py` | LayerBudgetCache drop-in (DynamicCache subclass) |
| `experiments/suite/run_all.py` | Unified experiment CLI (phased by model size + hardware) |
| `experiments/suite/config.py` | Model zoo, baselines, experiment matrix |
| `experiments/suite/runner.py` | Orchestrator: checkpoint/resume, OOM recovery |
| `experiments/results/suite/` | All checkpointed results by run_id |
| `tests/` | 189 unit tests (all passing) |

## Target Venues

- **ICML 2026** — under review, notification ~Apr 30
- **NeurIPS 2026** — DDL May 4 (abstract) / May 6 (full), paper ready at paper/neurips2026/
- **ICLR 2027** — fallback (~Sep/Oct 2026)

## License

Apache 2.0
