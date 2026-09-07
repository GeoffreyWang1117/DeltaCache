# Byte audit: does the compression-ratio knob move the bytes?

A one-question gate. The KV-compression literature compares methods along a
shared "compression ratio" axis. This measures whether that axis means the same
thing for every method, and whether what a method reports is what it delivers.

Runs on CPU. No model, no tokenizer, no GPU, no downloads.

## The claims under test

**A. The axis claim.** Eviction methods have a continuous token-budget knob.
Quantization methods have a small set of bit widths. If a requested ratio of 3x
has no representation in a quantized cache, then a table comparing method X at
2x, 3x, 4x and 6x against a quantized baseline is comparing four budgets against
one, and the ranking it reports is not a ranking at a matched budget.

**B. The accounting claim.** What a method reports as its footprint is not what
its returned tensors occupy.

## Two tiers, and why the distinction is load-bearing

| tier | target | what it can support |
|---|---|---|
| A | the baselines in this repository | that the instrument works, and our own reported-vs-delivered gap |
| B | upstream implementations, through adapters | evidence about the field |

The baselines here dequantize to FP16 by design, for quality comparison, and
report an analytic footprint. So claim B is **true by construction at tier A**
and carries no information about anyone else's code. `analyze.py --tier a` says
so in its own verdict rather than letting the number be quoted out of context.
Only tier B decides the gate.

## Running it

```bash
E=~/miniconda3/envs/deltacache
export PATH="$E/bin:$PATH"   # quanto builds a C++ extension and torch needs
                             # the ninja executable on PATH, not just installed

$E/bin/python -m experiments.byte_audit.gate \
    --adapters experiments/byte_audit/adapters/hf_quantized_cache.py \
               experiments/byte_audit/adapters/h2o_official.py \
               experiments/byte_audit/adapters/transformers_sliding_window.py \
    --out experiments/byte_audit/results.jsonl

$E/bin/python -m experiments.byte_audit.analyze experiments/byte_audit/results.jsonl --tier a
$E/bin/python -m experiments.byte_audit.analyze experiments/byte_audit/results.jsonl --tier b
```

One run produces both tiers. Every cell records its `source`, and the analyzer keeps the
two apart, so this repository's own code can never be counted as evidence about the field.

## Reading the output

Three numbers per cell, kept apart on purpose:

- **reported** the method's own accounting, from its `memory_bytes`
- **delivered** what its returned tensors actually hold, one count per storage
  so that views are not double-charged. Reported alongside **logical** bytes,
  what those tensors address; the two diverge when a small tensor is a view
  pinning a large buffer, and `view_overhead` is their ratio
- **allocator** the CUDA peak across the call, or -1 on CPU where the platform
  cannot supply it

`delivered` and `reported` are each classified as FLAT, DISCRETE or CONTINUOUS
over the requested grid. The cell worth looking for is CONTINUOUS reported over
FLAT delivered: a smoothly varying claim about a footprint that never moved.

`storage` reads `simulated` when nearly every byte sits in a float dtype, which
means the method modelled quantization and kept the original layout.

## The positive control

A method that was never really invoked and a method with a genuinely flat
footprint produce identical byte readings. Every cell therefore records whether
the method did anything: dropped tokens, or changed values. Cells that did
nothing are excluded and the method is marked UNVERIFIED rather than FLAT.

This is why eviction methods show `1 no-op` at a requested ratio of 1.0. Nothing
is evicted when the budget is not binding, so the control correctly rejects that
cell. That is the control working, not a failure.

## The decision rule, fixed before the numbers

Written in `analyze.py::verdict` so it is read first.

- **GO** at tier B when the axis claim holds and at least half of the verified
  methods report under half what they deliver. Write the paper.
- **PARTIAL** when only the axis claim holds. Narrow to the axis claim, or stop.
- **NO-GO** when the axis claim fails. File a bug report against the single
  implementation that showed it and archive the project.

## Tier-B result, 2026-09-06

Three upstream targets, at 512 tokens over 8 heads of 64 dimensions. Two families, and
two of the three come from the same library release, so the split is not a difference
between projects.

| target | family | requested ratios | distinct footprints delivered |
|---|---|---|---|
| official H2O, `utils_real_drop` | eviction | 6 | 6 |
| transformers sliding-window layer | eviction | 6 | 6 |
| transformers quantized cache, quanto | quantization | 6 | **2** |

Eviction tracks the request within 1%: asking for 3x delivers 2.99x, asking for 6x
delivers 6.02x. Quantization collapses six requests onto two footprints. Requesting 1x,
2x, 3x or 4x all deliver the same 3.54x, because the cache has no setting that means
"off" and no setting between four bits and two. Nominal four-bit delivers 3.54x and
nominal two-bit delivers 6.33x; the shortfall is the FP16 scale and shift buffers at a
group size of 64.

Two further observations worth their own lines:

**Nobody states a footprint.** None of the three exposes a figure for how many bytes it
holds. Claim B is therefore untestable upstream rather than passing, and the analyzer now
says so instead of scoring an absent claim as a clean one.

**Sliding-window eviction pins the whole prefill for one step.** The retained window is a
view into the prefill buffer, so immediately after prefill all six window sizes hold the
full 1048576 bytes regardless of window. The next update copies and it drops to the
window. Every adapter therefore measures in steady state, after one decode step;
measuring at prefill reports one transient peak for every setting and makes the knob look
flat. This was caught by the instrument reporting storage and logical bytes separately.

Verdict: **PARTIAL**. The axis claim holds on upstream code from two independent projects.

## Scope

Synthetic tensors, so this measures storage and accounting, never quality. It
cannot tell you whether a method is good. It can tell you whether the axis two
methods were compared along was the same axis.
