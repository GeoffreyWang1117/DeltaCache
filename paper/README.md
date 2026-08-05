# DeltaCache Paper

## Current Status

**Active version**: `main.tex` — ~8 pages, IEEE double-column format
**Previous**: IJCNN 2025 version archived to `archive/old_drafts/main_ijcnn2025.tex`

### Key Change from IJCNN Version

The IJCNN paper's core claim ("66% improvement via layer-aware eviction") was based on simulated data. Real experiments (2026-03-04) showed:
- Layer-aware eviction has **no effect** due to monolithic per-sequence cache blocks
- LFU dominates (99% hit rate at 10% capacity) — not importance-aware eviction
- The paper is now reframed as a **prefix caching system paper** with honest reporting

### Key Metrics (all from real experiments)

| Metric | Value | Context |
|--------|-------|---------|
| Avg speedup | **5.06x** | Mistral-7B, targeted scenarios |
| Max speedup | **10.90x** | Code completion (cached) |
| Long context | **5.17x** | 1801 tokens, constant ~17ms cached |
| Correctness | **100%** | fp16 (max diff < 0.016) |
| Memory overhead | **17-22%** | Prefix tree + metadata |
| Prefix threshold | **>200 tokens** | Break-even point |
| vs vLLM APC | **10-16x** | Different measurement scope (caveat in paper) |
| Best eviction | **LFU 99%** | @ 10% capacity (real model) |
| Layer-aware eviction | **No effect** | Negative result (Section 6) |

### Tables (11 total, all real data)

| # | Content | Source |
|---|---------|--------|
| 1 | Correctness validation | Real GPU measurements |
| 2 | Mistral-7B targeted speedup | Real GPU measurements |
| 3 | TinyLlama targeted speedup | Real GPU measurements |
| 4 | Prefix length sensitivity | Real GPU measurements |
| 5 | Context length scaling | Real GPU measurements |
| 6 | vLLM APC comparison | `vllm_apc_comparison.json` |
| 7 | Tiered cache infrastructure | `gpu_integration_v2.json` |
| 8 | Eviction under pressure | `eviction_pressure_tinyllama_*.json` |
| 9 | Memory overhead | Real GPU measurements |
| 10 | Layer attention analysis | `layer_attention_tinyllama_*.json` |
| 11 | Layer-aware ablation (negative) | `layer_ablation_tinyllama_*.json` |

## Compilation

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Directory Structure

```
paper/
├── main.tex                # Active paper (prefix caching system)
├── references.bib          # Bibliography
├── fancyhdr.sty / natbib.sty  # Required style files
├── figures/                # TikZ source + compiled PDFs
│   ├── architecture.tex/pdf
│   ├── inference_flow.tex/pdf
│   └── layer_weighting.tex/pdf
├── README.md               # This file
└── archive/
    ├── README.md
    ├── old_drafts/          # IJCNN 2025 + ICML 2026 + early draft
    ├── old_templates/       # ICML style files, example papers
    ├── spot_iclr2026_rejected/  # SPOT submission (rejected)
    └── iclr2026_archived/       # Earlier ICLR attempt
```

## Priority Improvements for Future Submissions

### P0 — Must Fix
- Validate on 13B+ models (Llama-2-13B) to demonstrate scalability
- Add batched inference evaluation
- Implement per-layer storage to enable real layer-aware eviction

### P1 — Should Fix
- Integrate with vLLM or SGLang for fair apples-to-apples comparison
- Add fair KV-only timing comparison with vLLM internals
- Explore learned eviction policies (beyond LRU/LFU)

### P2 — Nice to Have
- Distributed KV cache sharing across GPUs
- Encoder-decoder model support
- Token-level eviction combination (H2O / SnapKV)

---

*Last updated: 2026-03-04*
