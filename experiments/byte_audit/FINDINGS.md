# Byte audit: what KV-cache compression implementations actually deliver

**Date:** 2026-09-07 · **Gate:** GO · Six upstream implementations from four independent
projects, four context lengths, measured on CPU with synthetic tensors.

Reproduce with the commands in `README.md`. Raw cells in `results.jsonl`, verdicts in
`verdict_tierA.txt` and `verdict_tierB.txt`, published-axis survey in
`literature/crosscheck.txt`.

---

## The question

The KV-compression literature compares methods along a "compression ratio" axis. This asks
whether that axis means the same thing for every method, by measuring the bytes an
implementation really holds rather than the ratio it was asked for.

Two claims were registered before the numbers arrived, in `analyze.py::verdict`:

- **A, the axis claim.** Eviction exposes a continuous token budget; quantization exposes a
  handful of bit widths. If so, a table sweeping both families across one ratio axis is
  comparing a swept budget against a fixed one.
- **B', the bit-width claim.** Nobody publishes a footprint, but every quantization method
  states a nominal bit width, and bytes can be held against that.

## Result 1: the two families are not on the same axis

Distinct footprints delivered across a six-point ratio grid, counted within a single context
length. Identical at 256, 512, 1024 and 2048 tokens.

| implementation | family | distinct footprints | knob |
|---|---|---|---|
| transformers sliding window | eviction | 6 | window in tokens |
| official H2O, real-drop | eviction | 5 | heavy-hitter plus recent tokens |
| official SnapKV | eviction | 5 | `max_capacity_prompt` in tokens |
| official KIVI | quantization | 3 | bits in {2, 4, 8} |
| transformers quantized cache, quanto | quantization | 2 | bits in {2, 4} |
| FlexGen `compress`, shipped with H2O | quantization | **1** | bits, which change nothing |

Eviction tracks the request to within one percent: asking for 3x delivers 2.99x, asking for
6x delivers 6.02x. Quantization cannot. The eviction counts read 5 rather than 6 because a
requested ratio of 1x evicts nothing, which the positive control correctly rejects; the
sliding window reads 6 because a window equal to the sequence still drops one position.

The sharpest single cell: requesting 1x, 2x, 3x or 4x from the transformers quantized cache
delivers the same bytes, because it has no setting meaning "off" and none between four bits
and two.

## Result 2: nominal bit width does not determine delivered bytes

Effective bits are `16 x delivered / full-KV`, the bits actually spent per value against an
FP16 baseline. Each figure below was also derived by hand from the group layout and matched.

| implementation | nominal | effective | excess | explanation |
|---|---|---|---|---|
| official KIVI | 2, 4, 8 | 3, 5, 9 | 1.0 | FP16 scale and zero per group of 32 |
| transformers quanto | 2, 4 | 2.51-4.54 | 0.55 | FP16 scale and shift per group of 64, plus an unquantized 128-token residual window |
| FlexGen `compress` | 2, 3, 4, 5, 8 | 8.5 | 6.5 | one tensor element per value regardless of width |

"KIVI 2-bit" delivers 5.33x, not 8x. Two of the three quantizers spend at least one extra
bit per value.

The FlexGen row needs care rather than drama. That repository contains two paths.
`TorchCompressedDevice.allocate` computes a genuinely packed layout. `compress`, measured
here, is labelled by its own docstring as simulating group-wise quantization and stores one
element per value. So the same project delivers different bytes by path for the same nominal
width, and a paper reporting "4-bit" does not say which one ran.

## Result 3: sliding-window eviction pins the whole prefill for one step

The retained window is a view into the prefill buffer, so immediately after prefill every
window size holds the full sequence regardless of window. The next update copies and it
drops to the window. Every adapter therefore measures in steady state, after one decode
step. Measuring at prefill reports one transient peak for every setting and makes the knob
look flat, which is how this was found.

## Result 4: the field mostly avoids the mismatch by not comparing across families

Surveyed axes, verified from the papers rather than assumed (`literature/claims.csv`):

| paper | family | axis | quantization baseline present |
|---|---|---|---|
| RDKV | eviction | token budget, five levels | no |
| PyramidKV | eviction | cache size in tokens, four levels | no |
| KVTuner | quantization | bit-width pairs, nine levels | yes, at its own native widths |
| this project's retired draft | joint | **ratio, four columns** | yes |

Within a family the axis is the native knob, and papers use it correctly. Eviction papers
index by token budget; quantization papers index by bit width. The two families largely do
not appear in the same table at all.

So the finding is not that the field miscompares routinely. It is that **the cross-family
comparison is the one nobody grounds**, and the measurements above say why: a fixed-bit-width
method has one to three delivered footprints, so a ratio axis with more columns than that
cannot be matched at every column.

The clearest instance available is our own. The retired NeurIPS draft swept four ratio
columns against KIVI, whose measured footprint is byte-identical at all four. That table's
four-column comparison rests on a single KIVI operating point.

## What this does not show

Synthetic tensors, so this measures storage and accounting and never quality. It cannot say
whether a method is good. It says whether the axis two methods were compared along was the
same axis.

It also cannot re-run anyone's experiments, so no published number is claimed to be wrong.
The checkable statement is structural: whether a table places a method on an axis with more
columns than that method has representable settings.

Grouped-query geometry is plumbed through `Shape` but the adapters that score keys against
queries refuse under it rather than measure a multi-head cache and call it grouped. Every
number here is multi-head.

## Instrument defects found and fixed along the way

Recorded because each one would have produced a confident wrong number, and the same traps
apply to anyone repeating this.

1. **Quantized tensor subclasses advertise the size they stand in for.** Counting them
   directly charges the uncompressed footprint. The accountant now opens subclasses through
   the standard flatten protocol. Before the fix, quanto's packed cache read as full FP16.
2. **Storage and logical bytes must be reported separately.** A small tensor viewing a large
   buffer pins all of it. That distinction is what surfaced Result 3.
3. **Lazy attention with no bounds check never terminates.** Legacy iteration walks
   `__getitem__` from zero and stops only on `IndexError`. A sweep ran two and a half hours
   building tensors for indices past the end before this was caught.
4. **An absent claim is not a passed claim.** The analyzer originally scored "no method
   states a footprint" as a clean result. It now reports untestable, which is what led to
   claim B' replacing claim B.
5. **Tiers must not mix.** Every cell carries a source tag so this repository's own
   baselines can never be counted as evidence about the field.

## Status

Gate is GO on the registered rule: the axis claim holds on upstream code from four projects,
and two of three quantizers show a bit-width gap of at least one bit per value. The next
step is the write-up, not more measurement.
