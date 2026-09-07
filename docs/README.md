# Documentation index

**Read in this order if you are returning to the project or new to it.**

## 1. What is true now

| Doc | Answers |
|---|---|
| [`PROBLEM_FORMULATION.md`](PROBLEM_FORMULATION.md) | **Start here.** The mathematical model, what is being optimized, the quantitative targets, and what falsifies them. Also records that the earlier "noise floor" framing was already published (DiFR, Anthropic/FAR AI). |
| [`EXPERIMENT_PLAN_FAITHFULNESS.md`](EXPERIMENT_PLAN_FAITHFULNESS.md) | Measurement design and stop rule. ⚠ Its §2 "noise floor" claim of novelty is superseded by PROBLEM_FORMULATION §0. |
| [`REFRAME_VERIFAI_PROPOSAL.md`](REFRAME_VERIFAI_PROPOSAL.md) | Why the direction changed, and how it connects to ChainProve (VerifAI@ICLR 2026 / ICICS 2026) |
| [`ENV_REBUILD_DIAGNOSIS.md`](ENV_REBUILD_DIAGNOSIS.md) | How to get a working environment (10–20 min; the env is empty, not broken) |
| [`ENGINEERING_DESIGN.md`](ENGINEERING_DESIGN.md) | Library internals. Predates the direction change but is still accurate about the code. |

## 2. Why the old direction was retired

Read these before trusting any number in `paper/`, in the git history, or in any doc dated
before August 2026.

The assessments behind that conclusion — the adversarial self-review, the
claim-by-claim number audit, the competitive landscape survey and the evaluation-harness
feasibility study — are working documents and are kept out of this repository. What a
reader needs from them is stated here directly:

- Numbers in `paper/` and in docs dated before August 2026 have not all survived
  re-checking. Re-derive anything you intend to reuse from `experiments/results/`.
- The LongBench, GSM8K and MMLU harnesses have implementation-level defects and their
  scores should not be used.
- The measurement work that *is* published, and that stands on its own, is
  [`../experiments/byte_audit/`](../experiments/byte_audit/).

## 3. Reference

- [`examples.md`](examples.md) — library usage examples (predates the direction change)
- `../experiments/faithfulness/README.md` — the current measurement and its known limits
- `../experiments/archive/pre_suite_2026q1/README.md` — script → results → paper-claim provenance table
- `../paper/neurips2026/panel_round_6/` — the eight raw reviews

## Numbers you should not repeat

Every one of these appears in the unsubmitted draft and in the pre-August README:

| Claim as written | Reality |
|---|---|
| "118–119% of best 500-trial random search" | 115.6–120.9%, and n=2 prompts |
| "PPL ratio ≤1.03 at 4× on seven models" | Llama-2-13B is 1.11–1.14 |
| "≤1.01 at 4× up to 72B" | Qwen2.5-14B is 1.0348 |
| "Llama-2-70B at 8K needs over 40 GB" | 2.68 GB by the paper's own formula (GQA, 8 KV heads) |
| "ΔF(4→8) = 0.0035" | 0.0224 from the paper's own calibration table — and it sets the solver's action ordering |
| "43–72% GQA memory savings" | Artifact: two models with identical KV geometry are listed at 189 MB vs 95 MB. Both are 67.1 MB. Honest figure ≈43%. |
| "End-to-end vLLM integration confirms 24–72% memory savings" | Every vLLM result file reports `mean_freed_pct: 0.0`; measured peak memory is identical to baseline and 5% slower |
| mean-fill "96% KL reduction" | Arithmetically right, but it is **two single-position measurements** (`logits[:, -1, :]`, `n_texts=2`) |

## One correction the panel got wrong

Seven of eight reviewers reported that `paper/neurips2026/main.tex` **does not compile** and
flagged it as a desk-reject risk. That is a **false positive** — an artifact of the text
extraction they were given. `pdflatex -halt-on-error` exits 0 and produces 19 pages. The one
reviewer who was given the rendered PDF did not make this error; it found real rendering defects
instead (broken main table on p6, a figure contradicting its own caption). Always pass rendered
pages to a review panel.
