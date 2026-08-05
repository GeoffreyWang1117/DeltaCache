# Experiment design: is "near-lossless optimized inference" a verifiable claim?
**Date:** 2026-08-04 · Design only — nothing run, nothing installed.

---

## 0. Correction to the previous proposal

I told you the KL/top-1 inversion was "empirical payload already in hand." **It is not, and I overstated it.** Reading `experiments/run_layerwise_and_distortion.py:271-302`:

```python
ref_logits = out_ref.logits[:, -1, :].float()   # ← a SINGLE position
top1_match = (ref_logits.argmax(-1) == comp_logits.argmax(-1)).float().mean().item()
```

`ref_logits` has shape `[1, vocab]`. Both KL and `top1_match` are **one measurement per document**, and `n_texts=2`. So `top1_match` can only be 0.0, 0.5 or 1.0 — the reported values are literally **"2 of 2" and "1 of 2"**. The 95% Wilson CI on 1/2 is ≈[0.09, 0.91]; it is statistically indistinguishable from both 0% and 100%.

The inversion is a **hypothesis with a plausible mechanism**, not a finding. It stays worth testing — it is cheap and it has a real mechanism (below) — but it cannot carry a reframe on its own, and I should not have implied it could. This is the same error class that killed the main paper (n=2 texts behind 4-digit ratios); repeating it while proposing the replacement would be indefensible.

**Mechanism worth testing:** mean-fill writes phantom KV entries whose keys equal the *mean of retained keys* — i.e. they look like a plausible typical token, so they attract real attention mass and dilute the true distribution, potentially moving the argmax. Zero-fill writes obviously-anomalous keys that produce near-zero attention weight and are effectively ignored. That predicts exactly the observed direction: mean-fill better on aggregate distributional distance, worse on token identity. Plausible, unverified.

---

## 1. The question, stated so it cannot return "nothing"

> **Is there any metric on which current KV-compression methods' "near-lossless" claims are simultaneously (a) satisfied and (b) distinguishable from the output variation an *honest* implementation produces by accident?**

Every KV-compression paper reports "our distortion is small." **Nobody reports the distortion you get from changing nothing that should matter** — attention backend, batch size, GPU, matmul precision. Without that denominator, "small" is meaningless.

Three outcomes, all publishable:

| Outcome | Meaning | Where it goes |
|---|---|---|
| **A** — methods clear the noise floor on every metric | The field is fine; claims are sound | Null result; stop, cheaply |
| **B** — methods clear the floor on some metrics but not others, and metric choice reorders them | "Near-lossless" is underspecified; a faithfulness predicate must be named | Position + measurement paper; propose the predicate |
| **C** — method distortion sits **below** the honest-implementation noise floor | "Near-lossless" claims are unfalsifiable as stated; output comparison cannot detect compression | Strongest result — and it *strengthens* ChainProve: if you can't distinguish compression from honest nondeterminism by looking at outputs, the cryptographic path is the only path |

Note that B and C point in opposite directions for the detector idea (§5), and that is fine — the experiment tells us which mechanism contribution is even available before we promise one.

---

## 2. Experiment 0 — the noise floor (the gate, and the novel part)

**This runs first and it is the contribution most likely to survive on its own.**

Reference config: model M, **fp16 weights** (no bitsandbytes — see §6), sdpa attention, batch=1, fixed seed, GPU 1, uncompressed cache.

Perturbations that are *semantically null* — an honest provider could choose any of them:

| ID | Perturbation | Why it matters |
|---|---|---|
| N0 | Re-run identical config | Pure run-to-run nondeterminism floor |
| N1 | `attn_implementation="eager"` instead of sdpa | The matched-budget audit measures sdpa→eager as **−0.221 RULER**, larger than most method gaps |
| N2 | batch size 4 instead of 1 (same sequence) | Changes matmul reduction order |
| N3 | GPU 0 instead of GPU 1 | Same SKU, different die |
| N4 | TF32 matmul on/off | A one-line config flag |
| N5 | `torch.use_deterministic_algorithms(True)` | The "careful" provider |

Metrics (per position, see §3): KL(ref‖var), top-1 disagreement rate, top-5 set overlap.

**Output:** the distribution of null distortion, and its 95th percentile — the *honest-implementation envelope*. Every method in Experiment 1 is then judged against that envelope rather than against zero.

Cost: 6 configs × N documents. Trivial. **If N1 alone produces distortion comparable to a 4× compression method, that single plot is the paper.**

---

## 3. Metrics — fixing what the current code does wrong

The existing measurement takes one position and calls it a result. Replace with:

**Per-position, all suffix positions.** Not `logits[:, -1, :]`.

**Four families:**
1. *Distributional* — KL(ref‖comp), reverse KL, Jensen-Shannon distance, total variation
2. *Token identity* — top-1 agreement, top-5 Jaccard, Spearman ρ over the top-50 logits
3. *Confidence* — entropy shift, max-probability shift (these are what a deferral/routing system consumes)
4. *Effective* — the byte budget the cell **actually** consumed, logged and asserted against nominal

Family 4 is not optional. This project's own data has KIVI reporting **byte-identical `mean_memory_bytes` across all four requested CRs** — it silently ignores the compression knob. Every "memory-matched" claim in the literature that includes KIVI carries that exposure. Log actual bytes per cell; assert |effective/nominal − 1| < 0.05; fail loudly otherwise.

**Entropy stratification (Experiment 3).** Bin positions by *reference* entropy quintile. Most positions in natural text are trivially predictable — a method that only breaks hard positions looks fine on average. Prediction: distortion concentrates in the top entropy quintile, i.e. exactly at the decision-relevant positions that averaged metrics hide. If true, this is a second independent finding and it is the one that matters for routing/deferral consumers.

**Statistics.** Cluster bootstrap **at the document level**, not the position level — adjacent positions are correlated and position-level resampling would fabricate precision. Report 95% CIs on every reported number. Pre-register N ≥ 200 documents; do not report anything at n<50.

---

## 4. Experiment 1 — method distortion, and Experiment 2 — metric disagreement

**Exp 1.** Same reference. Arms: {LayerBudget, LayerBudget-KV, KIVI, H2O, SnapKV, StreamingLLM, PyramidKV, + zero-fill/mean-fill as a crossed factor} × CR {2, 3, 4, 6}. Same metrics. Judged against the Exp-0 envelope.

**Exp 2 — the inversion test.** Within Exp-1 data, rank methods under each metric family and compute Kendall τ between the rankings. The seed hypothesis predicts τ is low, specifically across the fill factor. **This is where the 1-of-2 coin flip gets a real answer.**

Documents must span regimes, not just WikiText-2: wiki prose, code, dialogue, and a retrieval-structured input (NIAH-style). The whole thesis is that behavior differs by regime; a single-corpus sweep would beg the question.

---

## 5. Experiment 4 — detector, conditional on Exp-1

Run **only if** Exp 1 shows separation above the noise floor. Given only outputs from an unknown provider, classify compressed vs uncompressed; report AUC with CIs.

- If AUC is high → a cheap statistical attestation primitive exists. That is the mechanism contribution and the direct ChainProve tie-in (the sub-weight-commitment attack surface).
- If AUC ≈ 0.5 → outcome C: only cryptography can close this gap. Also a contribution, and a stronger motivation for ChainProve than ChainProve currently has.

**Note:** only aggregate metrics were ever checkpointed — no raw logits or generations were stored. The detector needs fresh runs; it cannot be prototyped from the existing 1423 checkpoints.

---

## 6. Feasibility — the real constraints

**Weights must be fp16.** Every prior experiment ran on 4-bit weight-quantized models, never disclosed as a caveat. A KL between a compressed-cache run and a reference run is only meaningful if the weights are bit-identical and unquantized. This removes bitsandbytes from the critical path — and it imposes the binding constraint:

| Model | fp16 size | Fits a 24 GB 3090? |
|---|---|---|
| Mistral-7B, Llama-2-7B | ~14 GB | Yes |
| Llama-3.1-8B, Qwen3-8B | ~16 GB | Yes, tight |
| Llama-2-13B | ~26 GB | **No** — needs both GPUs (`device_map="auto"`) |
| Qwen2.5-14B | ~28 GB | **No** — needs both GPUs |
| Qwen2.5-72B | ~144 GB | **Not local.** Rental, or drop |

**So the local sweep is capped at 7–8B in fp16.** 13B/14B need both 3090s — and both are currently occupied by a `fisherkd` job (6.5 GB on GPU 0, 5.1 GB on GPU 1, ~30% util). Free VRAM today: ~16.5 GB / ~19.4 GB. Plan for GPU 1 and 7–8B, or wait for that job.

**Attention-materialisation ceiling.** `output_attentions=True` under eager attention materialises O(S²) per layer: at S=1024, 32 layers × 32 heads ≈ 2.1 GB; at S=4096 ≈ 34 GB → OOM (this is the documented A100 gotcha). Methods needing attention weights (H2O, SnapKV, PyramidKV) are therefore capped near S=1024–2048 unless the profiler is refactored to stream per-layer. LayerBudget/KIVI/StreamingLLM do not need it and can go longer.

**Compute.** ~8 methods × 4 CRs + 6 null configs ≈ 38 configs × 200 documents ≈ 7,600 forward passes per model. At ~0.5–1 s per cell including compression overhead (the paper's own table gives compress_kv at 100–500 ms) → **1–2 hours per model**. Two or three 7–8B models is an overnight run. This is genuinely cheap; the expensive thing was never the compute.

---

## 7. Threats to validity — pre-registered

1. **The reference is a convention, not truth.** "fp16 / sdpa / batch=1" is itself an arbitrary choice. If the noise floor turns out large, there *is* no canonical reference — which is the finding, not a flaw, but it must be stated rather than discovered by a reviewer.
2. **Our reimplementations.** The same caveat that sank the main paper. Mitigation: route as many methods as possible through **NVIDIA KVPress** so at least some arms are third-party code. This adds an install dependency but converts "trust our 17 baselines" into "trust the community harness."
3. **Prefix/suffix split.** The existing code hardcodes 0.6. Arbitrary. Fix it, justify it, or make it a factor — do not inherit it silently.
4. **Multiple comparisons.** 38 configs × 4 metric families × 5 entropy bins is a lot of tests. Pre-register the primary endpoint (Exp-0 envelope vs Exp-1 distortion, on KL and top-1) and mark everything else exploratory.
5. **Corpus selection** drives the answer; hence the multi-regime requirement in §4.

---

## 8. Sequence, and the stop rule

| Step | Depends on | Decides |
|---|---|---|
| 0 | env rebuild (10–20 min, §`ENV_REBUILD_DIAGNOSIS.md`) | — |
| 1 | PPL smoke cell | that the forward path works |
| 2 | **Exp 0, one model, N=200** | the noise floor — **run this before anything else** |
| 3 | Exp 1 + 2, 2–3 models at 7–8B | outcome A / B / C |
| 4 | Exp 3 (entropy stratification) — same data, no new runs | second finding |
| 5 | Exp 4 (detector) — only if Exp 1 separates | mechanism contribution |

**Stop rule, committed in advance:** if Exp 0 shows a noise floor *near zero* AND Exp 1 shows all methods comfortably above it on every metric with agreeing rankings, that is outcome A — the field is fine, there is no paper, and the honest move is to stop and write nothing. I want that branch stated now, while it costs nothing to state, rather than discovered after three weeks of looking for a way to make the data say something.

Nothing about venue or framing should be decided before step 3.

---

## 9. What is NOT on this path

- LongBench / GSM8K / MMLU harness repair (5–7 weeks). All three are *generation* harnesses; every experiment above is forward-pass only. Not needed.
- Anything to do with beating KIVI. That question is settled and the answer is no.
- bitsandbytes, except for reproducing old cells.
