# /scout complement — Papers Whose Limitations LayerBudget Addresses

**Date:** 2026-05-03
**Note:** Automated scout complement failed (LLM providers unavailable). Manual analysis from scan abstracts.

## Method

For each candidate paper from the deep scan, identify a stated limitation/future-work direction, then assess whether LayerBudget's contribution addresses it. Match strength on a 1-5 scale.

## Match Strength 5/5 (Strong complement, citable as positioning)

### 1. KIVI (Liu et al. 2024)
- **Their limitation:** Pure quantization; no token-dimension reduction. Cannot exploit the fact that some tokens are unimportant.
- **Our contribution:** Joint $(n_l, b_l)$ allocates the byte budget across both axes. At CR ≤ 4×, LayerBudget matches KIVI quality; at CR > 4× LayerBudget exceeds KIVI on retrieval (NIAH 100% vs 93% on Llama-2-7B at 4×).
- **Positioning sentence:** "KIVI delivers near-lossless quantization but caps compression at the INT4 floor; LayerBudget extends the byte budget by jointly compressing the token dimension."
- **Collaboration potential:** High. KIVI's per-channel/per-token quantization could be the F(b) calibration source for LayerBudget's cost model in a unified pipeline.

### 2. H2O (Zhang et al. 2024)
- **Their limitation:** Uniform heavy-hitter strategy across all layers; no architecture awareness.
- **Our contribution:** Per-layer budget $n_l$ + inverted importance (when applicable) protects layers H2O over-evicts.
- **Positioning sentence:** "H2O's uniform heavy-hitter eviction is layer-blind; LayerBudget's per-layer allocator routes the eviction budget away from layers that the leave-one-out probe identifies as bottlenecks."
- **Collaboration potential:** Medium. H2O's heavy-hitter signal could be plugged into LayerBudget's coverage model as an alternative to Gini.

### 3. PyramidKV (Cai et al. 2024)
- **Their limitation:** Fixed pyramid budget per layer (depth-decreasing); no adaptation to attention sparsity or quantization.
- **Our contribution:** Adaptive per-layer budget driven by Gini sparsity + importance + cost model.
- **Positioning sentence:** "PyramidKV's fixed pyramid is a special case of LayerBudget's allocation when $g_l$ and $w_l$ are taken to be monotone in depth; the adaptive allocator generalizes it to architecture-dependent regimes."
- **Collaboration potential:** Low (subsumed).

## Match Strength 4/5 (Good complement)

### 4. MoE-nD (Sun et al. 2026, concurrent)
- **Their limitation:** Offline-calibrated greedy solver (per-task LongBench-v1 + AIME); reported only at extreme CR (14×).
- **Our contribution:** Online cost model that runs in <1ms; reported at moderate CR (2-6×) across 7 model architectures and 8 evaluation tasks.
- **Positioning sentence:** "MoE-nD demonstrates that the per-layer joint regime extrapolates to extreme CR; LayerBudget characterizes the moderate-CR operating envelope and provides an online recipe."
- **Collaboration potential:** Very high — the two papers cover non-overlapping operating regimes of the same problem. A combined work could provide a unified moderate-to-extreme CR analysis.

### 5. EVICPRESS (Feng et al. 2025)
- **Their limitation:** Joint allocation at the request level, not per-layer.
- **Our contribution:** Per-layer granularity (32 layers on Mistral, 80 on Qwen2.5-72B).
- **Positioning sentence:** "EVICPRESS allocates eviction-vs-quant budget per request; LayerBudget refines the same trade-off per layer per request, exploiting layer-wise sparsity heterogeneity."
- **Collaboration potential:** Medium. The two granularities compose: request-level coarse allocation + per-layer fine allocation.

### 6. xKV (Chang et al. 2025)
- **Their limitation:** Cross-layer SVD operates on a different axis; does not address per-layer budget allocation.
- **Our contribution:** Orthogonal axis (per-layer) — directly composable with cross-layer SVD.
- **Positioning sentence:** "xKV reduces redundancy across layers; LayerBudget allocates budget within each layer. The two methods compose: apply xKV to share KV across layers, then run LayerBudget on the residual per-layer budget."
- **Collaboration potential:** High — composability is testable as a follow-on experiment.

## Match Strength 3/5 (Weaker complement)

### 7. HCAttention (Yang et al. 2025)
- **Their limitation:** Heterogeneous attention computing focuses on extreme context (4M tokens), uses an attention-routing strategy.
- **Our contribution:** Allocator-driven per-layer compression at moderate context (16K-32K).
- **Positioning sentence:** "HCAttention targets the extreme-context regime; LayerBudget's operating envelope is moderate-to-long context where the per-layer allocator's cost model is empirically calibrated."

### 8. KV Pareto (Patwari et al. EACL 2026)
- **Their limitation:** Joint KV-and-model compression at systems level; per-layer KV granularity not exposed.
- **Our contribution:** Per-layer KV granularity within LayerBudget.

### 9. SimLayerKV (Zhang et al. 2024)
- **Their limitation:** Layer-level reduction without quantization combination.
- **Our contribution:** Layer-level + quant joint.

## Potential collaborators (institutions)

| Group | Active papers | Why they would be interested |
|---|---|---|
| **NVIDIA team** (Behnam, Fu, Tsai, Zhao, Yu, Tumanov) | kvpress library + 3 KV-compression papers | Production deployment perspective; LayerBudget could land in kvpress as a per-layer allocator backend. |
| **Cornell + UW team** (Chang, Lin, Wu, Abdelfattah, Ceze) | xKV + DuoAttention | Direct composability with xKV; senior systems-ML support. |
| **MoE-nD team** (Sun, He, Harn, Qin) | MoE-nD | Concurrent paper, complementary regime — collaboration could merge moderate-to-extreme CR analysis. |
| **HCAttention team** (Yang et al.) | HCAttention 4M-token | Extreme-context perspective; potential composability for LayerBudget at long context. |

## Recommended use

1. **In §2 Related Work:** the "complementary" framing is already in main.tex for MoE-nD, EVICPRESS, xKV. The KIVI / H2O / PyramidKV positioning sentences could be folded into the existing eviction/quantization paragraphs as one-line ours-vs-theirs differentiations.

2. **In rebuttal/cover letter:** the collaboration potential ratings could inform whom to suggest as area chair or reviewer if NeurIPS allows author-suggested reviewers.

3. **For camera-ready experimental additions:**
   - MoE-nD on shared LongBench-v1 (their target) at 14× — high-leverage head-to-head
   - xKV composability test — apply LayerBudget on top of xKV's compressed cache
   - KIVI as the quant-only baseline (already in our experiments)
