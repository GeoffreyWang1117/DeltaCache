# DeltaCache

**Research question (current): can a claim of "near-lossless" optimized LLM inference actually be verified?**

> ⚠️ **This project changed direction in August 2026.** It began as a KV-cache compression
> method (LayerBudget) and that method is now retired — an independent adversarial review
> returned 8/8 Reject, and the method loses to a 2024 baseline at matched memory. What remains
> useful is the *measurement infrastructure*, which is being redirected at a different question:
> whether the faithfulness claims made across this entire field are verifiable at all.
>
> **New readers start at [`docs/README.md`](docs/README.md).** Do not trust numbers in the
> git history, in `paper/`, or in older docs without checking
> [`docs/CLAIM_VERIFY_2026Q3.md`](docs/CLAIM_VERIFY_2026Q3.md) first.

---

## Where this is going

Every KV-cache compression paper reports "our distortion is small." **None reports the distortion
you get from changing nothing that should matter** — the attention backend, the batch size, the
GPU, the matmul precision. Without that denominator, "small" has no meaning.

An external audit ([arXiv 2607.11942](https://arxiv.org/abs/2607.11942)) measures that swapping
`sdpa` for `eager` attention moves RULER accuracy by **0.221 — larger than the gap between most
published methods**, with identical weights and no compression at all. If a method's distortion
sits inside that envelope, its faithfulness claim is not falsifiable as stated.

That question connects directly to verifiable inference. **ChainProve**
([VerifAI@ICLR 2026](https://arxiv.org/abs/2603.18046), ICICS 2026) states its threat model as:

> *model substitution*, where a provider silently swaps in a cheaper model, **applies aggressive
> quantization, or returns cached outputs**

KV-cache compression is precisely that adversary behaviour — and it is invisible to a weight
commitment, because the provider really is running the advertised weights. Only the cache differs.
That gap is where this project is headed.

**Plan:** [`docs/EXPERIMENT_PLAN_FAITHFULNESS.md`](docs/EXPERIMENT_PLAN_FAITHFULNESS.md)
(includes a pre-registered stop rule).
**Rationale:** [`docs/REFRAME_VERIFAI_PROPOSAL.md`](docs/REFRAME_VERIFAI_PROPOSAL.md).

---

## What the old direction was, and why it was retired

**LayerBudget** jointly allocated per-layer KV token budgets `n_l` and quantization bit-widths
`b_l` under a global memory budget, via a greedy marginal-gain solver. Three design principles:
*quantize first, evict last*; *protect early layers* via inverted importance weights; and
*mean-fill* for evicted positions.

Three independent findings retired it:

1. **The idea was independently published ~6 times during a three-month gap.** RDKV
   ([arXiv 2605.08317](https://arxiv.org/abs/2605.08317), 8 May 2026) casts KV compression as
   rate–distortion in which *"eviction and quantization are two end-points of the same bit
   allocation scheme"*, and argues explicitly that they must be solved **jointly rather than in a
   staged fashion** — which refutes "quantize first, evict last". Also HqeKV (ACL 2026 Findings,
   code public), MoE-nD, ARKV, PolyKV. The venue bar was reset to 20–40× by STAR-KV (ICML 2026
   Spotlight) and KVTC (ICLR 2026, NVIDIA), against this project's 2–6×.
   → [`docs/SCOUT_2026Q3_LANDSCAPE.md`](docs/SCOUT_2026Q3_LANDSCAPE.md)

2. **It loses to KIVI at matched memory.** KIVI's `mean_memory_bytes` is byte-identical across all
   four requested compression ratios — it ignores the CR knob entirely, so the main comparison
   table was never memory-matched. At CR=4×, the only genuinely byte-matched point, KIVI wins on
   both models tested. The CR=2× "win" costs 1.96× the memory.
   → [`docs/SELF_REVIEW_ROUND6_VERDICT.md`](docs/SELF_REVIEW_ROUND6_VERDICT.md)

3. **Several headline numbers are wrong**, including the intro's motivating memory figure (off
   ~15×), a fidelity constant that drives the solver's action ordering (off 6.4×), and an
   end-to-end vLLM claim whose own result files all report `mean_freed_pct: 0.0`.
   → [`docs/CLAIM_VERIFY_2026Q3.md`](docs/CLAIM_VERIFY_2026Q3.md)

**Two findings from that work survive and are worth keeping:** *mean-fill* (largest measured
isolated effect, orthogonal to any eviction method, unclaimed by any 2026 paper — but currently
n=2), and the observation that **the direction of per-layer eviction sensitivity is
architecture-dependent** (early-layer bottleneck on Mistral-7B, late on Llama-2-13B, flat on
Qwen2.5-14B), which falsifies the fixed-pyramid assumption several published methods rest on.

---

## Repository layout

```
docs/                       ← START HERE (docs/README.md is the index)
  EXPERIMENT_PLAN_FAITHFULNESS.md   current experiment design + stop rule
  REFRAME_VERIFAI_PROPOSAL.md       why the direction changed
  SCOUT_2026Q3_LANDSCAPE.md         competitive landscape as of Aug 2026
  CLAIM_VERIFY_2026Q3.md            which published numbers are wrong
  SELF_REVIEW_ROUND6_VERDICT.md     8-persona independent review, 8/8 Reject
  HARNESS_FEASIBILITY_2026Q3.md     which evaluation harnesses are broken
  ENV_REBUILD_DIAGNOSIS.md          how to rebuild the conda env (10-20 min)
  ENGINEERING_DESIGN.md             library internals (still current)

deltacache/                 ← library; still sound, independent of the retired claims
  core/                       allocator, profiler, KV store, quantizer
  eviction/ hf_integration/ integrations/ vllm_integration/

experiments/
  faithfulness/             ← CURRENT WORK. Forward-pass output-divergence measurement
  suite/                    ← unified runner behind the 1423-checkpoint sweep
    tasks/                    ppl, mmlu, gsm8k, math, longbench, niah, ruler, throughput
                              (⚠ longbench/gsm8k/mmlu harnesses are broken — see docs)
  baselines/                17 reimplemented baselines (⚠ ours, not upstream)
  rebuttal_round1/          May 2026 rebuttal experiments + A100 run
  results/suite/            checkpointed results by run_id
  archive/pre_suite_2026q1/ superseded runners + provenance table for paper numbers

paper/
  neurips2026/              draft that was made ready but NEVER SUBMITTED (deadline missed)
    panel_round_6/            the 8 independent reviews that retired the method
  archive/                  ICLR 2026 / SPOT (rejected) / old drafts

tests/                      unit tests for the library
```

## Known-bad artifacts

Kept for provenance, but do not cite:

| Artifact | Problem |
|---|---|
| `paper/neurips2026/main.tex` | five wrong numbers in abstract/contributions; see CLAIM_VERIFY |
| `experiments/results/suite/20260421_215801` | Llama-2-13B run, all accuracies 0 / PPL null — superseded by `20260424_080406` |
| `experiments/results/suite/20260417_140331` | Qwen2.5-14B, same failure — superseded by `20260424_080406` |
| `suite/tasks/longbench.py` | right-truncation deletes the question before the model sees it; all 16 tasks at floor even uncompressed |
| `suite/tasks/{gsm8k,mmlu,math}.py` | prompts are 60–400 tokens, too short to discriminate KV compression at all |
| all downstream task results | `fill="mean"` is applied unconditionally to *every* baseline, so those columns are not the published algorithms |

211 `(model, task, method, CR, seq)` cells conflict across runs. Prefer the newest `run_id`.

## Setup

```bash
conda activate deltacache          # bare env as of 2026-08-01; see docs/ENV_REBUILD_DIAGNOSIS.md
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.6   # 5.x breaks the suite's DynamicCache usage
pip install -e .
```

New measurements should use **fp16 weights, not 4-bit** — all prior experiments silently ran on
4-bit weight-quantized models, which is a confound for any faithfulness measurement. On a 24 GB
card this caps single-GPU runs at ~7–8B.

## Status

| | |
|---|---|
| Method paper | **Retired.** Not submitted anywhere; not submittable. |
| NeurIPS 2026 | Deadline missed May 2026; never submitted |
| Current direction | Attestation of optimized inference — design stage, no results yet |
| Next step | Rebuild env → PPL smoke cell → Experiment 0 (noise floor) |

## License

Apache-2.0
