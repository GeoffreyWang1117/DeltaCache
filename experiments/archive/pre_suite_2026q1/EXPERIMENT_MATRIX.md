# Experiment Matrix — COMPLETE

## Quality (PPL) Experiments — WikiText-2, prefix=60%, 12 baselines × 4 CRs

| Model | Params | Layers | KV Heads | 512 tok | 1024 tok | 2048 tok |
|-------|--------|--------|----------|:-------:|:--------:|:--------:|
| TinyLlama-1.1B | 1.1B | 22 | 4 | ✅ | ✅ | — |
| Llama-2-7B-Chat | 6.7B | 32 | 32 (MHA) | ✅ | ✅ | ✅ |
| Mistral-7B-Instruct-v0.2 | 7.2B | 32 | 8 (GQA) | ✅ | ✅ | ✅ |

**Total: 7 configs × 12 methods × 4 CRs × 6 texts = 2,016 evaluations**

## LayerBudget PPL Ratio Summary

| Config | 2x | 3x | 4x | 6x | Beats KIVI+KVTuner |
|--------|:--:|:--:|:--:|:--:|:--:|
| Llama-2-7B@512 | 1.000 | **0.999** | 1.002 | **0.996** | 4/4 |
| Llama-2-7B@1024 | **1.000** | 1.004 | 1.005 | 1.002 | 4/4 |
| Llama-2-7B@2048 | 1.000 | 1.002 | 1.006 | 1.005 | 3/4 |
| Mistral-7B@512 | 1.000 | 1.005 | 1.004 | 1.002 | 3/4 |
| Mistral-7B@1024 | 1.000 | 1.002 | 1.001 | **1.000** | 3/4 |
| Mistral-7B@2048 | **1.000** | 1.003 | 1.002 | 1.002 | 3/4 |

**Bold** = ratio ≤ 1.000 (matches or beats Full KV)
**Win rate: 20/24 (83%)** where LayerBudget beats both KIVI and KVTuner

## Result Files

| File | Model | Seq Len |
|------|-------|---------|
| `unified_quality_512tok_llama2_7b.json` | Llama-2-7B | 512 |
| `unified_quality_1024tok_llama_2_7b_chat_hf.json` | Llama-2-7B | 1024 |
| `unified_quality_2048tok_llama2_7b.json` | Llama-2-7B | 2048 |
| `unified_quality_512tok_mistral_7b_instruct_v0.2.json` | Mistral-7B | 512 |
| `unified_quality_1024tok_mistral_7b_instruct_v0.2.json` | Mistral-7B | 1024 |
| `unified_quality_2048tok_mistral_7b_instruct_v0.2.json` | Mistral-7B | 2048 |
| `unified_quality_1024tok_tinyllama.json` | TinyLlama | 1024 |

## Paper Figures

| Figure | File | Content |
|--------|------|---------|
| Fig 1 | `ppl_vs_compression_all.pdf` | 2×3 grid: PPL ratio vs CR for all configs |
| Fig 2 | `method_heatmap_comparison.pdf` | Side-by-side heatmaps: LB vs KIVI vs KVTuner |
| Fig 3 | `win_rate_chart.pdf` | Bar chart: LB win/tie/loss vs each baseline |
