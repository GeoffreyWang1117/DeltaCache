# Problem formulation: detectability of KV-cache eviction
**Date:** 2026-08-04 · Survey + mathematical model. No experiments run.

---

## 0. Correction: the framing I proposed two turns ago is largely taken

I proposed the "noise floor" idea — measure the output variation an *honest* implementation produces
(backend, batch size, GPU), then judge compression against that envelope rather than against zero —
and said no paper in this area reports that denominator.

**That is wrong.** [DiFR](https://arxiv.org/abs/2511.20621) (Karvonen, Reuter, Rinberg, Marks,
Garriga-Alonso, Warr — MATS / Harvard / FAR AI / **Anthropic**, Nov 2025) formalizes exactly this, §4.1:

> The verifier has access to a calibration set generated under specification φ on trusted hardware.
> This calibration set may come from a single "reference" deployment or **from a pool of
> implementations that are all deemed acceptable (for example, different GPU types or tensor-parallel
> configurations). Pooling scores from these honest configurations defines the range of benign
> numerical variation allowed under H₀**, and we then flag only those deployments whose statistics
> fall outside this pooled distribution.

That is the composite null I was going to propose as the contribution. They also ship an open-source
vLLM integration. Two adjacent works close the rest of the space I had in mind:

- **Thinking Machines**, *Defeating Nondeterminism in LLM Inference* (Sep 2025) — diagnoses the root
  cause as **batch-size dependence of reduction kernels**, not floating-point non-associativity, and
  gives batch-invariant kernels producing **bit-identical outputs across 1,000 runs** at ~61.5%
  throughput cost (~34% after SGLang's CUDA-graph integration). So the noise floor is not a fact of
  nature; it is a known engineering defect with a known fix.
- **RUT** ([arXiv 2506.06975](https://arxiv.org/abs/2506.06975)) — rank-based uniformity two-sample
  test for black-box API auditing; detects 4/8-bit weight quantization, hidden system prompts, SFT,
  full substitution. **IRIS** ([arXiv 2607.20860](https://arxiv.org/html/2607.20860v1), Jul 2026) —
  budgeted black-box auditing of substitution and routing dilution in gateways.

**And DiFR already tests KV-cache quantization.** Its Figure 1 evaluates four misconfigurations:
4-bit model quantization, **FP8 KV-cache quantization**, incorrect sampling seed, and temperature
1.1 vs 1.0.

So: the problem is modeled, the tooling exists, and the nearest threat class is already benchmarked.
Anything we do here is an **extension to an existing line**, not a new framework. I would rather say
that now than have it discovered in review.

---

## 1. What is actually left

DiFR tests **κ_quant**: bf16→fp8 on the cache. A *dense* perturbation — every cached element is
perturbed, at every position, from the first token.

Nobody has tested **κ_evict**: token eviction under a per-layer budget (H2O, SnapKV, PyramidKV,
Ada-KV, LayerBudget). Searching both literatures confirms the split — the KV-compression side audits
eviction *policies* structurally, the verification side tests weight and precision changes. Neither
asks whether eviction is detectable from outputs.

This is not merely an untested cell. **κ_evict is structurally different from every threat these
verifiers were designed against**, in two ways that matter statistically:

**(i) It has an exact zero region.** Every published eviction policy protects a sink + recent window
of size `n_min`. If the context is short enough that the budget is not binding, **nothing is evicted
and the deviation is identically zero** — not small, zero. No test on any number of tokens can detect
it, because there is nothing to detect. This is a hard threshold at `S₀`, not an asymptotic decay.
It also means the standard short-prompt audit probe is exactly the wrong probe.

**(ii) Its signal is sparse across positions, not homogeneous.** Eviction perturbs the output only at
generated positions whose attention mass would have fallen on evicted entries. Elsewhere the logits
are unchanged to numerical precision. κ_quant perturbs everywhere.

---

## 2. The model

### 2.1 Setup (DiFR's, extended)

Provider computes outputs claimed to conform to specification φ = (architecture, weights, numerical
precision, sampling hyperparameters, PRNG seed ρ). Verifier recomputes and scores.

- **H₀:** the provider followed φ; discrepancies are benign numerical noise.
- **H₁:** the provider deviated.

Per-token score (Token-DiFR), under a shared Gumbel seed σ, with `t*` the provider's claimed token
and `t̂ = argmax_i z_i` the verifier's:

$$\delta_{\text{logit}}(t^*,\hat t,\sigma) \;=\; \big(\ell[\hat t] + T\,g_\sigma[\hat t]\big) - \big(\ell[t^*] + T\,g_\sigma[t^*]\big), \qquad \text{score}_t = \min(\delta_{\text{logit}}, \Delta_{\max})$$

DiFR aggregates **uniformly** over a batch of `B` token positions and thresholds:
$$\bar S = \tfrac1B\sum_{t=1}^{B}\text{score}_t \;\gtrless\; \tau$$

### 2.2 The extension: sparse vs dense deviations

Let $s_t(\kappa,S) = D\big(p_\varphi(\cdot\mid x,y_{<t}) \,\|\, p_{\varphi_\kappa}(\cdot\mid x,y_{<t})\big)$ be the per-position signal at context length $S$.

**Dense (κ_quant):** $s_t \approx \mu$ for all $t$, and $\mu$ is essentially independent of $S$.

**Sparse (κ_evict):** with per-layer budget $n_l$ and protected window $n_{\min}$,

$$s_t(\kappa_{\text{evict}}, S) = \begin{cases} 0 \quad\text{exactly}, & S \le S_0 \;(\text{budget not binding}) \\[2pt] \mu \;\text{w.p.}\; \pi(S), \quad \approx 0 \;\text{w.p.}\; 1-\pi(S), & S > S_0 \end{cases}$$

where $\pi(S)$ is the fraction of positions whose attention mass intersected the evicted set. Under a
fixed *absolute* memory ceiling (the deployment reality), the eviction fraction $\rho(S) = 1 - n/S \to 1$,
so $\pi$ increases with $S$.

### 2.3 Why this breaks uniform aggregation — the crisp statement

Under H₁ with a sparse signal, the aggregate mean is $\pi\mu$ while the H₀ noise is $\sigma/\sqrt B$.
For a level-α test with power $1-\beta$:

$$B^*_{\text{uniform}}(\kappa,S) \;\approx\; \frac{(z_\alpha + z_\beta)^2\,\sigma^2}{\big(\pi(S)\,\mu(S)\big)^2}, \qquad B^*_{\text{targeted}} \;\approx\; \frac{(z_\alpha + z_\beta)^2\,\sigma^2}{\mu(S)^2}$$

$$\boxed{\;\frac{B^*_{\text{uniform}}}{B^*_{\text{targeted}}} \;=\; \frac{1}{\pi(S)^2}\;}$$

**Uniform averaging over token positions is statistically inefficient by $1/\pi^2$ against a sparse
deviation.** If $\pi = 0.05$, that is a factor of 400 in required audit budget. This is the classical
sparse-signal detection setting (Donoho–Jin higher criticism; the detection boundary for sparse
normal mixtures), and it prescribes the fix: a selective statistic — higher criticism, a max/top-k
statistic, or explicit budget allocation toward high-signal positions — instead of the mean.

### 2.4 The optimization problem

The auditor holds a budget $B$ of verifier forward-pass positions (each is a recomputation, so cost
is real and grows with $S$). Choose an allocation $\pi$ over cells $(S, t, \ell)$:

$$\max_{\pi \ge 0} \;\; \sum_{(S,t)} \pi(S,t)\, I(S,t) \qquad \text{s.t.} \qquad \sum_{(S,t)} \pi(S,t)\, c(S) \;\le\; B$$

with $I(S,t)$ the Chernoff information at that cell and $c(S)$ the verifier cost. This is a fractional
knapsack; greedy by **information-per-verifier-FLOP** is optimal for the LP relaxation.

That is *the same object* as LayerBudget's allocator — greedy marginal gain of a scarce budget across
layers — with "quality per byte" replaced by "information per verifier-FLOP". The existing solver
code transfers directly, and this is the one place where the retired work carries forward as
machinery rather than as a claim. It is also the same shape as ChainProve's Fisher-information
audit-budget triage, which allocates *which layers* to audit.

---

## 3. Quantitative targets

Everything below is measurable and each has a stated failure condition.

| # | Quantity | Prediction | Falsified if |
|---|---|---|---|
| **Q1** | $S_0$ — the context length below which eviction is exactly undetectable | $S_0 = n_{\min}\cdot\text{CR}$; a hard threshold, per method | no such threshold exists (i.e. methods evict even when the budget is not binding) |
| **Q2** | $\pi(S)$ — signal sparsity, for $S \in \{512, 1\text{K}, 4\text{K}, 16\text{K}, 32\text{K}\}$ × {H2O, SnapKV, LayerBudget, KIVI} × CR {2,4,6} | $\pi \ll 1$ for eviction, $\pi \approx 1$ for KIVI (dense quantization) | $\pi \approx 1$ for eviction too — then there is no sparsity and no efficiency gap |
| **Q3** | $B^*_{\text{uniform}}$ for κ_evict, versus DiFR's reported ~10² tokens (4-bit weight quant) and ~10³ (FP8 KV quant) at AUC>0.999 | $B^* \gg 10^3$, or ∞ below $S_0$ | $B^* \lesssim 10^3$ everywhere → **no blind spot, no paper** |
| **Q4** | Empirical gain of targeted vs uniform aggregation | approaches $1/\pi^2$ | gain < 2× → the fix is not worth the complexity |

**The headline claim, if it survives:** *existing inference verifiers have sample complexity
independent of context length; KV-cache eviction has sample complexity that is infinite below a
threshold and falls with context length, so a verifier calibrated on weight-quantization threats is
blind exactly where cache compression is deployed.*

**The falsification is cheap and comes first.** Q1 and Q2 are forward-pass-only measurements on a
single 7–8B model. If Q2 returns $\pi \approx 1$, the whole line dies in about a day, and that is the
correct outcome to buy first.

---

## 4. Honest sizing

This is an **extension to DiFR**, not a competing framework, and it should be written and pitched
that way:

- one threat class (eviction) that the existing benchmark omits,
- one structural property (sparsity + an exact-zero region) that breaks the existing aggregation,
- one algorithmic fix (selective/allocated aggregation) with a stated $1/\pi^2$ efficiency argument.

Realistic venues: **VerifAI@ICLR** (natural home, already accepts this author, and the ChainProve
line is there), a security B-tier, or as a direct contribution to the DiFR line — their code is open
and the natural move is to implement κ_evict inside their harness rather than rebuild one. That last
option is worth considering seriously; it is faster, it is honest about provenance, and a
collaboration with the DiFR authors is worth more than a solo workshop paper.

What this is *not*: a NeurIPS-main-track framework contribution. The framework exists and has
Anthropic authors on it.

---

## 5. What carries over from the dead project

| Asset | Use |
|---|---|
| Greedy marginal-gain allocator | becomes the audit-budget allocator in §2.4 |
| 17 baseline implementations | the κ_evict arms to test |
| `experiments/faithfulness/measure_output_divergence.py` | per-position $s_t$ measurement (must be fixed to score all positions, not `logits[:, -1, :]`) |
| Architecture-dependent layer-sensitivity finding | which layers to fingerprint in an Activation-DiFR-style scheme is architecture-dependent |
| The 1423 checkpoints | **not usable** — only aggregate metrics were stored, no logits |

## 6. Immediate next step

Not an experiment. **Read DiFR §4.3 and Appendix A–D properly**, and run their open-source
implementation on one eviction method. If Token-DiFR detects H2O at CR=4 in 300 tokens, Q3 is
falsified before we write a line of our own code. That is a day of work and it gates everything else.
