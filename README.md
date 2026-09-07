# DeltaCache

**Research question (current): is KV-cache *eviction* detectable from a provider's outputs, and how
does the audit cost scale with context length?**

> ⚠️ **This project changed direction in August 2026.** It began as a KV-cache compression method
> (LayerBudget). That method is retired — an independent adversarial review returned 8/8 Reject, and
> it loses to a 2024 baseline at matched memory. The direction is now inference *verification*.
>
> **New readers start at [`docs/PROBLEM_FORMULATION.md`](docs/PROBLEM_FORMULATION.md)**, then
> [`docs/README.md`](docs/README.md). Do not trust numbers in the git history, in `paper/`, or in
> docs dated before August 2026 without re-deriving them from
> `experiments/results/`. The claim-by-claim audit that lists which figures are wrong
> is kept out of this repository; ask the author if you need it.

---

## ▶ Resume here

**Next action: run [DiFR](https://github.com/adamkarvonen/difr)'s open-source implementation against
one eviction method (H2O at CR=4).** This is a go/no-go gate and it comes before any code of our own.

```
1. Rebuild the conda env      — 10-20 min, docs/ENV_REBUILD_DIAGNOSIS.md
                                (env is empty, not broken; pin transformers==4.57.6)
2. PPL smoke cell             — confirms the forward path works
3. Clone + run DiFR on H2O    — github.com/adamkarvonen/difr, has a vLLM integration
```

**Read the gate before running it:** if Token-DiFR detects H2O at CR=4 within ~300 output tokens —
the budget at which it already detects 4-bit weight quantization at AUC > 0.999 — then target **Q3**
in `PROBLEM_FORMULATION.md` is falsified, there is no blind spot, and the honest move is to stop.
That outcome is a legitimate result of one day's work, not a failure.

Note the GPUs are shared: a `fisherkd` job was using ~6.5 GB on GPU 0 and ~5.1 GB on GPU 1. Use fp16
weights (not 4-bit) for anything faithfulness-related, which caps a single 24 GB card at ~7–8B.

---

## The question, stated precisely

A provider serving KV-cache compression is running the **genuine advertised weights** — so a weight
commitment passes bit-identically. Only the cache differs. **ChainProve**
([VerifAI@ICLR 2026](https://arxiv.org/abs/2603.18046), ICICS 2026) names this in its threat model:

> *model substitution*, where a provider silently swaps in a cheaper model, **applies aggressive
> quantization, or returns cached outputs**

The general verification problem is already well modelled. [DiFR](https://arxiv.org/abs/2511.20621)
(Anthropic / FAR AI, Nov 2025) formalizes the composite null over a pool of acceptable honest
configurations, ships a vLLM integration, and already benchmarks **FP8 KV-cache quantization**.

What is untested is **token eviction** (H2O, SnapKV, PyramidKV, LayerBudget), which differs
structurally from every threat those verifiers were built against:

1. **It has an exact zero region.** Every published policy protects a sink + recent window, so below
   a context threshold `S₀` nothing is evicted and the deviation is *identically zero*. No test on
   any number of tokens can detect it. The standard short-prompt audit probe is exactly wrong here.
2. **Its signal is sparse across positions**, concentrated where attention mass would have fallen on
   evicted entries — whereas quantization perturbs everywhere.

DiFR aggregates uniformly over a token batch. Against a signal present at only a fraction `π` of
positions, uniform averaging costs a factor **1/π²** in required audit budget. That is the gap, the
predicted mechanism, and the quantitative target — all with falsification conditions, in
[`docs/PROBLEM_FORMULATION.md`](docs/PROBLEM_FORMULATION.md).

**Honest sizing:** this is an *extension to DiFR*, not a new framework. Natural venue is
VerifAI@ICLR, or a direct contribution to the DiFR line — their code is open, and implementing
`κ_evict` inside their harness beats rebuilding one.

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
   (Competitive landscape notes are kept out of this repository.)

2. **It loses to KIVI at matched memory.** KIVI's `mean_memory_bytes` is byte-identical across all
   four requested compression ratios — it ignores the CR knob entirely, so the main comparison
   table was never memory-matched. At CR=4×, the only genuinely byte-matched point, KIVI wins on
   both models tested. The CR=2× "win" costs 1.96× the memory.

3. **Several headline numbers are wrong**, including the intro's motivating memory figure (off
   ~15×), a fidelity constant that drives the solver's action ordering (off 6.4×), and an
   end-to-end vLLM claim whose own result files all report `mean_freed_pct: 0.0`.

**Two findings from that work survive and are worth keeping:** *mean-fill* (largest measured
isolated effect, orthogonal to any eviction method, unclaimed by any 2026 paper — but currently
n=2), and the observation that **the direction of per-layer eviction sensitivity is
architecture-dependent** (early-layer bottleneck on Mistral-7B, late on Llama-2-13B, flat on
Qwen2.5-14B), which falsifies the fixed-pyramid assumption several published methods rest on.

---

## Repository layout

```
docs/                       ← START HERE (PROBLEM_FORMULATION.md, then README.md)
  PROBLEM_FORMULATION.md            the model, the targets, the falsification conditions
  EXPERIMENT_PLAN_FAITHFULNESS.md   measurement design + stop rule (§2 partly superseded)
  REFRAME_VERIFAI_PROPOSAL.md       why the direction changed
  SECURITY_AUDIT_2026Q3.md          supply-chain and code-path audit
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
| Current direction | Detectability of KV-cache eviction — modelled, no experiments run |
| Next step | Rebuild env → PPL smoke cell → **run DiFR against H2O** (go/no-go gate) |
| Last session | 2026-08-04. Tree clean, all analysis committed. |

## License

Apache-2.0
