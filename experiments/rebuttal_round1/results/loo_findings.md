# LOO Second-Model Findings — Round 1 Rebuttal

**Run:** 2026-05-02
**Method:** Per-layer leave-one-out at CR=6× zero-fill on WikiText-2 (2 chunks of 512 tokens). For each layer ℓ, restore that layer's KV to full precision and measure ΔPPL_ratio. Higher ΔPPL = restoring this layer recovers more quality = layer is more sensitive to eviction.

**Paper baseline (Mistral-7B, 32L GQA, §5):** Restoring layers 0–1 improves PPL by 1.8–2.0; restoring any layer after 14 has near-zero effect. Hence "early layers are the bottleneck" → inverted importance.

## Headline finding

**Three models, three different bottleneck locations.** The "early layers are the bottleneck" principle does not generalize.

| Model | Arch | L | Early LOO | Mid LOO | Late LOO | Verdict |
|---|---|---|---|---|---|---|
| Mistral-7B (paper) | GQA | 32 | high | mid | low | early-bottleneck |
| Llama-2-13B | MHA | 40 | +0.030 | +0.084 | **+0.093** | **late-bottleneck** |
| Qwen2.5-14B | GQA | 48 | +0.039 | (n/a) | +0.038 | **indeterminate** |

- On Llama-2-13B: late > early by 3× → **opposite** of Mistral.
- On Qwen2.5-14B: early ≈ late (no significant directional bias).
- On Mistral-7B (paper): early > late.

**The eviction bottleneck location is architecture-dependent.**

## Implications for the paper

**The "(2) protect early layers" design principle is Mistral-specific.** Possible explanations the data narrows down:

1. **MHA vs GQA does NOT cleanly explain it** — Qwen2.5-14B is GQA (like Mistral) but is indeterminate, not early-bottleneck. So architecture-class alone is not the pivot.
2. **Depth might matter** — Mistral 32L (early), Llama-2-13B 40L (late), Qwen2.5-14B 48L (flat). Could be that depth shifts the bottleneck monotonically from early → late as L grows. With only 3 data points this is suggestive, not conclusive.
3. **Model family / training corpus** — possible that Llama-2 (older training) and modern Qwen2.5 simply have different layer specialization. Single-anchor calibration cannot capture this.

Either way, **the paper's inverted-importance scheme (k=5, τ=0.3) applied uniformly across architectures is wrong on at least one model**. The fact that LB still scores 1.11 at 4× and 1.56 at 6× on Llama-2-13B (Table 19) suggests the cost-model's marginal-gain ranking is robust enough that suboptimal importance weights don't dominate — but the paper's framing of inverted-importance as a *principle* rather than a *Mistral-specific heuristic* must be revised.

## Recommended camera-ready edits

### Option A (honest scope-down)

In §3.2 (importance), replace:
> Critically, this assigns higher weight to early layers, reflecting our key finding (Section 5): leave-one-out analysis shows that early layers (0–10) are the primary quality bottleneck under eviction…

with:
> On Mistral-7B (32L GQA), leave-one-out shows early layers (0–10) dominate eviction sensitivity, motivating inverted importance with $k{=}5$, $\tau{=}0.3$. On Llama-2-13B (40L MHA), leave-one-out shows the opposite pattern: late layers dominate, with mean LOO improvement +0.093 vs +0.030 for early layers (Table~\ref{tab:loo-cross}). The inverted-importance scheme is therefore a *Mistral-tuned* default, not a universal principle. We retain it across our experimental matrix because the cost-model's marginal-gain ranking is robust to importance weight perturbations within a $\sim$2$\times$ window (sensitivity analysis in Appendix); a per-architecture tuned importance would close the remaining gap on Llama-2-13B at 6$\times$.

### Option B (camera-ready experiment)

Run LB on Llama-2-13B with **non-inverted (sigmoid late-high) importance** at CR=6×. If PPL improves significantly over the inverted scheme, that's a publishable positive result: "the cost model is correct, the importance direction is architecture-specific, and we provide a one-line tuning recipe."

This is a single Vast.ai run (~30 min on A100). I recommend it over Option A — it converts a weakness into a contribution.

## What this means for the abstract

The current abstract says:
> Two design principles drive its effectiveness: (1) quantize first, evict last … (2) protect early layers — leave-one-out analysis reveals that early layers are the quality bottleneck under eviction…

Suggested rewrite (preserves the headline numbers, scopes the principle):
> Two design principles drive its effectiveness: (1) quantize first, evict last; (2) the eviction bottleneck is layer-architecture-dependent — leave-one-out analysis on Mistral-7B identifies early layers as the bottleneck, motivating inverted importance, while Llama-2-13B's eviction bottleneck is in late layers; the greedy allocator absorbs both regimes through its marginal-gain ranking.
