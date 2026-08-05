# LayerBudget-KV — `(n_l, b_K_l, b_V_l)` extension findings

**Run:** 2026-05-03 (v2 with `coverage^2` matched to production)
**Setup:** 4-bit weights, WikiText-2 PPL, seq=1024, prefix=60%, n=6 chunks, mean-fill.

## Headline result

| | CR=2× | CR=3× | CR=4× | CR=6× |
|---|---|---|---|---|
| **Mistral-7B** | | | | |
| KIVI | 1.006 | 1.006 | 1.006 | 1.006 |
| MoE-nD framing | 1.028 | 1.030 | 1.029 | **1.038** |
| LayerBudget | **1.000** | **1.006** | **1.020** | 1.250 |
| **LayerBudget-KV** | 1.036 | 1.036 | 1.036 | 1.062 |
| **Llama-2-7B** | | | | |
| KIVI | 1.006 | 1.006 | 1.006 | 1.006 |
| MoE-nD framing | 1.027 | 1.046 | 1.088 | 1.137 |
| LayerBudget | **0.996** | **1.004** | **1.021** | 1.099 |
| **LayerBudget-KV** | 1.026 | 1.026 | 1.026 | **1.037** |

## Key finding

**LayerBudget-KV is essentially flat at ~1.03 across CRs from 2× to 6× on both architectures** — it sacrifices moderate-CR sharpness (3pp behind LayerBudget at CR=2-4×) for extreme-CR robustness. At CR=6× it improves over LayerBudget by **19pp on Mistral-7B (1.250→1.062)** and **6pp on Llama-2-7B (1.099→1.037)**, and **beats MoE-nD at CR=6× on Llama-2-7B (1.037 vs 1.137)**.

## Mechanism

Independent K/V bit-widths (`b_K_l ≠ b_V_l` per layer) give the solver a degree of freedom that the shared-`b_l` formulation lacks. Once budget tightens enough that `b_l = 4` becomes mandatory across most layers (CR=6× regime), the shared formulation cannot allocate `(b_K=8, b_V=4)` mixes that the independent formulation can.

## Trade-off ⇒ ship as variant, not replacement

LB wins at CR=2-3× by 3pp; LB-KV wins at CR=6× by 6-19pp. Pick by deployment regime:
- **Production / serving** (CR=2-4×): use LayerBudget (default).
- **Extreme compression / unstable CR** (CR=5-6×+): use LayerBudget-KV.

A unified framework presenting both as design points of the same `(n_l, b_l, ...)` allocator family is the cleanest framing.

## Implementation note

`coverage^2` (squared penalty matching production) is required — the unsquared coverage produced flat 1.062 at all CRs because the greedy converged before saturating budget on most layers. Once squared, allocations actually differ across CRs.
