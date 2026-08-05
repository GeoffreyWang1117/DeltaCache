# /scout draft — Refreshed §2 Related Work

**Date:** 2026-05-03
**Note:** Automated scout draft failed (LLM providers unavailable). Draft produced manually from scan data.

## Refreshed §2 (replaces or augments current text)

```latex
\section{Related Work}
\label{sec:related}

\textbf{Token eviction.}
H2O~\citep{zhang2024h2o} retains attention heavy-hitters plus recent tokens.
SnapKV~\citep{snapkv2024} uses observation-window voting.
PyramidKV~\citep{pyramidkv2024} varies budget by layer depth with a fixed pyramid.
D2O~\citep{d2o2025} uses per-layer diversity-based allocation.
SqueezeAttention~\citep{squeezeattention2025} classifies layers via residual cosine similarity.
CAKE~\citep{cake2025} allocates budgets proportional to attention entropy $\times$ temporal variance.
Ada-KV~\citep{adakv2025} operates at head granularity.
EvolKV~\citep{evolkv2025} uses evolutionary optimization.
LAVa~\citep{lava2025} derives importance from residual information loss.
SimLayerKV~\citep{simlayerkv2024} provides a layer-level KV reduction baseline.
All evict tokens at \emph{full precision}---none jointly optimize quantization.

\textbf{KV quantization.}
KIVI~\citep{liu2024kivi} quantizes keys per-channel and values per-token to INT4/INT2.
KVTuner~\citep{kvtuner2025} assigns per-layer mixed-precision via sensitivity analysis but requires offline calibration and no token eviction.
KVmix~\citep{kvmix2026} uses gradient-based layer importance for mixed-precision.
SVDq~\citep{svdq2025} achieves 1.25-bit key compression via SVD-based mixed precision.
All quantization methods operate independently of token eviction.

\textbf{Joint token-precision optimization.}
``More Tokens, Lower Precision''~\citep{moretokenslowerprecision2025} demonstrates the optimal token-count/bit-width trade-off but applies it \emph{uniformly} across layers.
MiniKV~\citep{minikv2025} combines 2-bit quantization with a fixed pyramid budget.
``No Token Left Behind''~\citep{notokenleftbehind2024} pairs importance-aware mixed-precision quantization with retention but does not co-optimize $n_l$ and $b_l$ jointly.
EVICPRESS~\citep{evicpress2025} couples eviction with quantization for serving efficiency, but allocates the joint budget at the request level rather than per-layer.
The closest concurrent work is \textbf{MoE-nD}~\citep{moend2026}, which independently arrives at the same $(eviction, K\text{-bits}, V\text{-bits})$ per-layer joint formulation under a global memory budget; their approach uses an offline-calibrated greedy solver and a mixture-of-experts routing abstraction targeting extreme compression (14$\times$) on LongBench-v1 and AIME reasoning, while \layerbudget uses an online marginal-gain solver with a closed-form coverage law (Eq.~\ref{eq:coverage}), reports the operating envelope from 2$\times$ to 6$\times$ across seven 7B--72B models, and contributes the architecture-dependence finding for importance direction (\S\ref{sec:method}). The two designs are complementary: MoE-nD demonstrates that the per-layer joint regime extrapolates to extreme CR; \layerbudget characterizes the moderate-CR regime that dominates production serving.
xKV~\citep{xkv2025} attacks a different axis (cross-layer SVD) and is composable with our per-layer allocation.
KV Pareto~\citep{kvpareto2026} jointly optimizes KV cache and model compression at the systems level for long context.

\textbf{Long-context KV inference.}
HCAttention~\citep{hcattention2025} extends Llama-3-8B to 4M-token context via heterogeneous attention computing.
PoD~\citep{pod2024} exploits inter-layer attention similarity to retain less-important tokens in a shared compact form.
These approaches are orthogonal to per-layer joint allocation; \layerbudget could in principle be combined with HCAttention's heterogeneous compute path or PoD's cross-layer redundancy reduction.

To our knowledge, no prior \emph{online, training-free} method jointly optimizes per-layer $(n_l, b_l)$ across the seven 7B--72B model architectures and 2--6$\times$ operating regimes we evaluate.
```

## Insertion guidance

The draft above expands the current §2 from ~30 lines to ~50 lines. To stay within page budget:

1. **Drop or condense** the "Long-context KV inference" paragraph if pages tight — HCAttention and PoD can move to a single sentence in the Joint paragraph.
2. **Drop SimLayerKV citation** if SnapKV/PyramidKV already cover the Layer-level theme.
3. **Drop SVDq** if KIVI is sufficient as the quantization headline.
4. **Keep MoE-nD differentiation paragraph at all costs** — this is the highest-leverage citation in the new set.

## Summary of changes vs original §2

- Added: SimLayerKV, SVDq, No Token Left Behind, EVICPRESS, **MoE-nD**, xKV, KV Pareto, HCAttention, PoD
- Modified: "Joint token-precision optimization" paragraph extended with MoE-nD differentiation
- New: "Long-context KV inference" paragraph

## Implementation status

- All 7 must-cite papers (MoE-nD, xKV, EVICPRESS, KV Pareto, No Token Left Behind, PoD, HCAttention) **already added to references.bib in this session**
- §2 differentiation paragraph for MoE-nD, EVICPRESS, xKV **already inserted in main.tex**
- KV Pareto, No Token Left Behind, PoD, HCAttention citations are in the bib but not yet woven into §2 — this draft above does the weaving
- Optional: SimLayerKV, SVDq, Expected Attention not yet in bib (4/5-relevance, low priority)
