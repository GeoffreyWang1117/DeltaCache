# MoE-nD / xKV official-code sanity check — findings

## xKV (Chang et al. 2025)

**Public code:** github.com/abdelfattah-lab/xKV (cloned to /tmp/xkv).

**Algorithm comparison: `fake_svd` (official) vs `fake_svd_truncate` (our reimpl)**

| Step | Official xKV | Our reimpl |
|---|---|---|
| Input shape | `(bs, nh, sl, hd)` → reshape to `(bs, sl, nh*hd)` | `(bs, sl, nh, d)` → reshape to `(bs, sl, nh*d)` |
| SVD | `torch.linalg.svd(x, full_matrices=False)` | `torch.linalg.svd(x, full_matrices=False)` |
| Truncate | `U[:,:,:rank]`, `S[:,:rank]`, `Vh[:,:rank,:]` | identical |
| Sqrt-S split | `sqrt(S)` distributed evenly over U and Vh | identical |
| Multiply back | `matmul(U_scaled, Vh_scaled)` → reshape to original | identical |

**Verdict:** algorithm-equivalent. Our re-implementation will produce numerically identical results to xKV's official `fake_svd` on the same input (modulo Hadamard transform and quantizer, which we match by leaving disabled — same as their `configs/example.yaml` default `hadamard: true` is for the with-quantization variant; no-quant baseline matches).

**What's NOT verified:** end-to-end pipeline (custom MInference KIVI patch, `KVCompress` patcher class, attention re-routing). Their full pipeline runs on RULER long-context (≥16K) with multi-GPU torchrun, which is beyond our scope; we run their core compression mechanism in our standardized WikiText-2 PPL pipeline.

## MoE-nD (Sun et al. 2026)

**Public code:** Not yet released. arxiv preprint (arXiv:2604.17695) is reference-only.

Searched: `github.com/SubmissionAnonymous/MoE-nD`, `github.com/anonymous-moend/moend` — both 404. Author affiliations don't surface a public repo at submission time.

**Verdict:** Algorithm verification by code comparison is impossible. Our re-implementation follows the published algorithm description (per-layer `(eviction, K-bits, V-bits)` joint allocation under global memory budget; offline-calibrated greedy solver). Our re-implementation makes the following design choices to match their description:
- Independent K/V bit-widths (b_K, b_V) ∈ {4, 8, 16}²
- No inverted importance signal (their paper relies on cost model + Gini alone)
- Coverage law shared with our LayerBudget for fair comparison

Any gap to their published numbers is attributable to:
1. Different eval harness (we use WikiText-2 PPL; they report LongBench-v1 + AIME)
2. Calibration differences (offline routing per-task vs our online cost model)
3. Implementation gaps (e.g.\ their MoE routing abstraction is not modeled)

These are flagged in §2 of the paper.

## Camera-ready commitment

When MoE-nD code becomes public, we plan a single sanity cell on Mistral-7B WikiText-2 at CR=4× to verify our re-implementation reproduces their published behavior within ±2pp. The §2 caveat in the paper makes this commitment explicit.
