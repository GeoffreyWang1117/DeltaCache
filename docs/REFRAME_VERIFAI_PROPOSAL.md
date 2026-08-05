# Reframe proposal: from KV-compression method → attestation of optimized inference
**Date:** 2026-08-04 · Written after reading the two accepted papers directly (not from description).

---

## 1. What the two papers actually claim

### ChainProve — VerifAI@ICLR 2026 (accepted) + ICICS 2026 (accepted, short)
`/home/coder-gw/Projects/nanoZkinference/paper/icics226/main.tex`

A zero-knowledge proof system for **verifiable LLM inference**: a client or third-party auditor checks that a provider executed the advertised model on a committed input, without learning weights or activations. Layerwise decomposition into independently provable sub-circuits linked by a SHA-256 commitment chain; 3.2–3.7 KB per sub-circuit (~83 KB at L=12); Halo2 IPA, no trusted setup; soundness ε≈3×10⁻³⁷.

**Its threat model, verbatim (§2.1):**

> LLM-as-a-Service exposes two integrity-side risks that prompt this work: ***model substitution*, where a provider silently swaps in a cheaper model, applies aggressive quantization, or returns cached outputs**; and *computation forgery*, where the returned y does not actually correspond to the advertised f_θ(x). Today these are deterred only contractually.

**"applies aggressive quantization, or returns cached outputs" is a literal description of what this project builds.** LayerBudget/KIVI/H2O *are* the provider-side technology in ChainProve's adversary model. That is not a narrative bridge I constructed — it is the second clause of their threat definition.

### ROA-LLM / PSE — ICML 2026 (accepted, camera-ready)
`/home/coder-gw/Projects/ROA-LLM/paper_icml/sections/abstract.tex`

"Persistent Semantic Entities in Tool-Augmented LLM Systems." Implicit agent state that persists across sessions, activates on events, propagates across agent boundaries — **invisible to standard debugging**. 20 models / 9 families / 1.5B–1T. Findings: contamination is universal (20–100%); persistence depends on *type* not scale; **context-isolated self-verification** achieves 20–79% reduction while **keyword-based detection produces systematic false positives**; contamination compounds 3× across multi-agent pipelines.

---

## 2. The specific gap neither paper covers

ChainProve's commitment chain binds the **weights** `c_W` and the per-layer activations `h_ℓ`. But KV cache compression is **not a weight substitution** — the provider runs the genuine advertised weights. A provider serving LayerBudget-, KIVI-, or H2O-compressed generations passes any weight-commitment check trivially, because `c_W` is bit-identical.

What differs is the *cache*, and therefore the activations at decode time. The chain would catch that — **but only if the auditor already knows what the uncompressed `h_ℓ` should be**, which means re-running the inference, which is the cost ZK exists to avoid.

So: **KV cache compression sits in the blind spot between "same weights" and "same output."** ChainProve names it in the threat model and does not (yet) address it; the entire KV-compression literature addresses it and never asks whether it is attestable.

There is also a sharper technical coupling worth stating: **ChainProve's Fisher-information audit-budget triage and LayerBudget's per-layer importance allocation are the same object over different budgets.** ChainProve allocates *which layers to audit* under an audit budget from per-layer Fisher scores (validated at Spearman ρ=0.916 against perturbation impact); LayerBudget allocates *how many bytes each layer gets* under a memory budget from per-layer importance weights, motivated by leave-one-out perturbation impact. LayerBudget's LOO analysis measures precisely the quantity ChainProve validates Fisher against — and this project's most durable finding is that **that quantity's sign is architecture-dependent** (early-bottleneck on Mistral-7B, late on Llama-2-13B, flat on Qwen2.5-14B). If layer importance flips direction by architecture, a fixed Fisher-triage schedule inherits the same fragility. That is a direct, testable contribution back into ChainProve's efficiency tool.

---

## 3. The empirical payload is already in hand

**(a) Faithfulness metrics contradict each other.** From `experiments/rebuttal_round1/results/mean_fill_kl_20260502_173052.json`, same model, same CR=6×:

| fill | logits KL | **top-1 agreement with uncompressed** |
|---|---|---|
| zero | 0.6696 | **1.00** |
| mean | 0.0636 | **0.50** |

Mean-fill is 10.5× better on KL and agrees with the uncompressed model on **half** the top-1 tokens; zero-fill is 10× worse on KL and matches **100%**. For a compression paper this is a curiosity. For an attestation system it is the central question: *which predicate does the certificate bind?* Commit to token identity and the "better" method fails half the time; commit to the distribution and the other one fails. **"Near-lossless" is not a well-defined predicate.**

**(b) An honest provider is non-deterministic in ways the client cannot observe.** The matched-budget audit (arXiv 2607.11942) measures that swapping the attention backend **sdpa → eager shifts RULER accuracy by 0.221 — larger than most method gaps**, with identical weights and no compression. So even a fully honest provider running the advertised model produces materially different outputs depending on an unobservable backend flag.

**(c) Rankings invert under the deployment-order protocol.** Same audit: under query-agnostic compression (compress context, *then* append question — the real serving order), SnapKV falls below a "keep-start + recent-window" trivial baseline. What the literature certifies as SOTA is an artifact of the evaluation order.

**(d) Nobody checks what budget a baseline actually consumed.** From this project's own checkpoints: KIVI's `mean_memory_bytes` is byte-identical across all four requested CRs — it silently ignores the compression knob. Every "memory-matched" comparison in the literature that includes KIVI has the same exposure.

These four are one thesis: **claims of faithful optimized inference are, as currently reported, unverifiable.** (a) and (d) are this project's own data; (b) and (c) are external and citable.

---

## 4. Feasibility — this route avoids every broken harness

The core measurement is `experiments/run_layerwise_and_distortion.py:301–304`:

```python
kl = F.kl_div(comp_probs.log(), ref_probs, reduction="batchmean").item()
top1_match = (ref_logits.argmax(-1) == comp_logits.argmax(-1)).float().mean().item()
```

**Forward-pass only.** It never invokes `generate_from_cache`. LongBench, GSM8K, MMLU — all three broken harnesses — are *generation* harnesses on a different code path. PPL uses the same forward-pass path and works correctly. So the reframe's core experiment runs entirely on verified-working code, and the 5–7 week harness repair is **not on its critical path**.

What it currently is: 2 fills × 2–3 models × **n=2 texts**. What it needs to be: {17 methods} × {7 models} × {CR 2,3,4,6} × N texts, reporting KL, top-1 agreement, top-k rank correlation, and entropy shift. That is a sweep, not a rewrite.

**Blocker:** the Python environment is dead (`import transformers` → ModuleNotFoundError), flagged in `RESEARCH_PORTFOLIO_2026H2.md` as *"所有补实验的总闸"*. Rebuild is step 0 regardless of direction.

Also note: only aggregate metrics were checkpointed — no raw generations or logits were stored. Any detector work needs fresh runs.

---

## 5. Honest assessment of the two links

**ChainProve link: strong.** Their threat model names the behavior; the Fisher-triage / per-layer-importance coupling is a concrete technical bridge with a testable claim attached; and the gap (compression is invisible to weight commitment) is real and unclaimed by either literature.

**PSE link: weak — do not make it load-bearing.** The shared vocabulary is "invisible state" and "detection produces false positives," which is thematic, not technical. PSE is about agent-level semantic contamination; this is about numerical divergence in a serving stack. If a three-paper arc is wanted, the honest shape is: **ChainProve (cryptographic integrity of inference) ← DeltaCache-reframed (integrity of *optimized* inference, the layer the commitment chain can't see) — with PSE as a sibling on "unobservable state in deployed LLM systems," not as a link in the chain.** Overselling it will be caught; PSE's own defense-evaluation framing (context-isolated self-verification beats keyword detection) is the one methodological idea that genuinely transfers.

**Ceiling, stated plainly.** This is a measurement + position contribution, not a method contribution. As a pure audit paper its ceiling is a strong workshop (VerifAI is the natural home, and it is a venue already accepting this author) or a B-tier security venue. To reach A-tier it needs a *mechanism*: the obvious candidate is **a detector — can a client tell, from outputs alone, that a provider silently served from a compressed cache?** That is concrete, falsifiable, unclaimed, uses this project's infrastructure as its measurement substrate, and slots directly under ChainProve as the sub-weight-commitment attack surface. It is also the version most likely to fail, so it should be prototyped before it is promised.

---

## 6. Concrete first step (≈1 week after env rebuild)

Do **not** touch LongBench. Run the faithfulness sweep:

1. Rebuild the Python environment (blocks everything).
2. Extend `run_layerwise_and_distortion.py` to the full method × model grid; report KL, top-1 agreement, top-5 rank correlation, entropy shift; N ≥ 50 texts with bootstrap CIs (current n=2 is not reportable).
3. The falsifiable question: **is the KL/top-1 inversion general, or an artifact of mean-fill at CR=6× on two texts?** If it holds across methods and models, the position paper writes itself and the detector becomes plausible. If it collapses at n=50, the reframe loses its empirical core and the honest move is to stop.

Everything else — venue, framing, whether to attempt the detector — waits on that answer.
