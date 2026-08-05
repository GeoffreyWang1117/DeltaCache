# Documentation index

**Read in this order if you are returning to the project or new to it.**

## 1. What is true now

| Doc | Answers |
|---|---|
| [`EXPERIMENT_PLAN_FAITHFULNESS.md`](EXPERIMENT_PLAN_FAITHFULNESS.md) | What we are trying to measure, how, and when to stop. **Current work.** |
| [`REFRAME_VERIFAI_PROPOSAL.md`](REFRAME_VERIFAI_PROPOSAL.md) | Why the direction changed, and how it connects to ChainProve (VerifAI@ICLR 2026 / ICICS 2026) |
| [`ENV_REBUILD_DIAGNOSIS.md`](ENV_REBUILD_DIAGNOSIS.md) | How to get a working environment (10–20 min; the env is empty, not broken) |
| [`ENGINEERING_DESIGN.md`](ENGINEERING_DESIGN.md) | Library internals. Predates the direction change but is still accurate about the code. |

## 2. Why the old direction was retired

Read these before trusting any number in `paper/`, in the git history, or in any doc dated
before August 2026.

| Doc | Finding |
|---|---|
| [`SELF_REVIEW_ROUND6_VERDICT.md`](SELF_REVIEW_ROUND6_VERDICT.md) | 8-persona independent-context panel: **8/8 Reject, mean 2.88**. KIVI beats LayerBudget at the only genuinely byte-matched compression ratio. Rounds 1–5 had scored 6.25 — those were in-session with author context and 4 personas. |
| [`SCOUT_2026Q3_LANDSCAPE.md`](SCOUT_2026Q3_LANDSCAPE.md) | The core idea was independently published ~6× during a three-month gap. RDKV subsumes and explicitly refutes the "quantize first, evict last" framing. Field bar moved to 20–40×. |
| [`CLAIM_VERIFY_2026Q3.md`](CLAIM_VERIFY_2026Q3.md) | **Which specific numbers are wrong**, with the corrected values. Consult this before reusing anything. |
| [`HARNESS_FEASIBILITY_2026Q3.md`](HARNESS_FEASIBILITY_2026Q3.md) | LongBench / GSM8K / MMLU harnesses are broken at the implementation level; two of the three should be deleted rather than repaired. |

## 3. Reference

- [`examples.md`](examples.md) — library usage examples (predates the direction change)
- `../experiments/faithfulness/README.md` — the current measurement and its known limits
- `../experiments/archive/pre_suite_2026q1/README.md` — script → results → paper-claim provenance table
- `../paper/neurips2026/panel_round_6/` — the eight raw reviews

## Numbers you should not repeat

From `CLAIM_VERIFY_2026Q3.md` and the round-6 panel. Every one of these appears in the
unsubmitted draft and in the pre-August README:

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
