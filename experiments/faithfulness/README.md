# Faithfulness measurement — the current research direction

This directory holds the measurement the project is being rebuilt around: **how far does a
compressed-KV run's output distribution move away from an uncompressed reference, and is that
movement distinguishable from the variation an honest implementation produces by accident?**

Design and rationale: **`docs/EXPERIMENT_PLAN_FAITHFULNESS.md`**.
Why the direction changed: see the banner in the top-level `README.md`.

## What is here

### `measure_output_divergence.py`
Formerly `experiments/run_layerwise_and_distortion.py`. Forward-pass only — it never calls
`generate_from_cache`, so it does **not** touch the broken LongBench/GSM8K/MMLU generation
harnesses. It computes, between a reference (uncompressed) and a compressed run:

- logit-level KL divergence and top-1 agreement
- per-layer attention entropy shift and attention KL
- a leave-one-out variant (restore one layer at a time)

## ⚠ Known limitation — read before extending

At `measure_output_divergence.py:271`:

```python
ref_logits = out_ref.logits[:, -1, :].float()   # a SINGLE position
top1_match = (ref_logits.argmax(-1) == comp_logits.argmax(-1)).float().mean().item()
```

`ref_logits` is `[1, vocab]`, so both KL and `top1_match` are **one measurement per document**,
and the default is `--n-texts 2`. `top1_match` can therefore only be 0.0, 0.5 or 1.0.

The paper's mean-fill headline (96% / 90.5% / 95.7% KL reduction) and the "zero-fill preserves
top-1 while mean-fill does not" observation both come from **two single-position measurements**.
The 95% Wilson CI on 1/2 is roughly [0.09, 0.91]. Treat these as hypotheses, not results.

## What needs to change before this produces anything reportable

Per `docs/EXPERIMENT_PLAN_FAITHFULNESS.md`:

1. **Score every suffix position**, not `logits[:, -1, :]`.
2. **Add the noise-floor control (Experiment 0)** — measure divergence under semantically-null
   perturbations (sdpa↔eager, batch size, GPU, TF32, plain re-run) to get an
   *honest-implementation envelope*, and judge every method against that rather than against
   zero. No paper in this area reports this denominator; it is the most likely contribution.
3. **Stratify by reference entropy** — most positions in natural text are trivially predictable,
   so averaged metrics hide damage concentrated at decision-relevant positions.
4. **fp16 weights, no bitsandbytes.** Every prior experiment silently ran on 4-bit weights; a KL
   against a reference is only meaningful if weights are bit-identical. This caps local runs at
   7–8B on one 24 GB 3090.
5. **Log the bytes actually consumed** per cell and assert against nominal. This project's own
   data has KIVI reporting byte-identical memory across all four requested CRs.
6. **Cluster bootstrap at the document level**, N ≥ 200, CIs on everything. Report nothing at n<50.

A stop rule is pre-registered in the plan: if the noise floor is near zero and every method sits
comfortably above it with agreeing rankings, there is no paper and the correct action is to stop.
