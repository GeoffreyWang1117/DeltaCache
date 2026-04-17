# DeltaCache: ICLR 2026 Workshop Submission

## Target Workshop
**SPOT: Scaling Post-Training for LLMs**
- Deadline: January 30, 2026
- Venue: ICLR 2026 Workshop
- Format: 9 pages main text + unlimited references

## Submission Files

### Main Paper
- `main_iclr2026.tex` - LaTeX source
- `main_iclr2026.pdf` - Compiled paper (6 pages)
- `references.bib` - Bibliography

### Style Files
- `iclr2026_conference.sty` - ICLR 2026 style
- `iclr2026_conference.bst` - Bibliography style
- `fancyhdr.sty`, `natbib.sty` - Required packages

## Key Contributions

1. **Attention-Aware Eviction**: Uses attention scores to prioritize important cache entries
   - Formula: I(e) = w_a·ᾱ_e + w_f·log(1+freq_e) + w_r·1/(1+log(1+t_e))

2. **Layer-Aware Caching**: Prioritizes late-layer (semantic) KV cache over early-layer (local pattern)
   - Formula: w_l = 1/(1 + e^(-5(l/L - 0.3)))
   - **Key Result**: 90% fewer evictions at 25% cache capacity vs LRU

3. **Hierarchical Memory Management**: GPU→CPU tiering with proactive offloading

## Main Experimental Results

| Metric | Value |
|--------|-------|
| Max Prefill Speedup | **9.7×** (1500 tokens) |
| Max TTFT Improvement | **19.5×** |
| RAG Speedup | 1.76-2.56× |
| Eviction Reduction (Layer-Aware vs LRU) | **90%** at 25% cache |
| Correctness | 100% at fp16 |

## Memory Pressure Experiment (Mistral-7B)

| Cache Capacity | Policy | Evictions | Hit Rate |
|----------------|--------|-----------|----------|
| 25% | LRU | 51 | 97.9% |
| 25% | Layer-Aware | **5** | 97.9% |
| 25% | Hierarchical | 7 | 97.9% |

## Checklist Before Submission

- [x] Anonymous submission (no author names visible)
- [x] Uses ICLR 2026 style
- [x] Main text ≤ 9 pages (current: 6 pages)
- [x] All citations resolved
- [x] PDF compiles without errors

## How to Compile

```bash
cd paper
pdflatex main_iclr2026.tex
bibtex main_iclr2026
pdflatex main_iclr2026.tex
pdflatex main_iclr2026.tex
```

## Code Availability

The DeltaCache implementation is available in the parent directory:
- `deltacache/` - Core library
- `experiments/` - Experiment scripts
- `experiments/results/paper/` - Experimental results

Key implementation files for novel contributions:
- `deltacache/eviction/advanced_policies.py` - Attention-aware and layer-aware policies
- `deltacache/core/hierarchical_memory.py` - Hierarchical memory manager
