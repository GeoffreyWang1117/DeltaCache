# A100 Camera-Ready Experiments

Closes the long-context / 4K-NIAH gaps for LayerBudget-KV that require
eager-attention extraction at sequences exceeding the shared-RTX-3090 ceiling.

## Hardening vs prior round

These scripts mirror the proven `experiments/scripts/a100_one_shot.sh`
patterns (which ran successfully on instance 35588280 for the 72B
experiments — 5.5h compute, no rework):

- **Per-cell checkpointing** — every (model, CR, method) cell writes a
  separate JSON. A pre-emption only loses the in-progress cell.
- **Resume on rerun** — `is_done()` skip already-completed cells.
- **Pre-flight checks** — GPU memory, disk, deltacache import,
  layer_budget_kv registration, HF auth, Llama-2 access — all fail fast.
- **Smoke mode** — `DC_SMOKE=1 bash run_all.sh smoke` runs 1 cell × 1 rep
  in ~5 min to verify the path works before committing to the full sweep.
- **Mode selection** — `bash run_all.sh niah` or `long` runs just one phase.
- **Path autodetect** — `_common.py:setup_paths()` walks up from the script
  location to find the repo root, so the same scripts work whether the
  repo is at `/root/DeltaCache` or `/workspace/DeltaCache`.
- **Phase timers** — every cell logs elapsed time in mm:ss; a stalled
  cell is detected immediately rather than being noticed at the end.
- **Aggregation step** — at end, all per-cell JSONs are bundled into a
  single timestamped file for the paper update.

## Setup on a fresh vast.ai A100 80GB

```bash
# 1. SSH in
ssh -p <port> root@ssh<n>.vast.ai

# 2. Clone the repo + checkout the paper branch
git clone https://github.com/<user>/DeltaCache.git /root/DeltaCache
cd /root/DeltaCache
git checkout paper-experiments

# 3. Install
pip install -e .
pip install bitsandbytes==0.43.1 transformers==4.57.6 datasets accelerate

# 4. HF auth (Llama-2 is gated)
huggingface-cli login

# 5. Pre-flight + smoke + full
cd experiments/rebuttal_round1/a100_camera_ready
bash preflight.sh                       # ~1 min, must pass
DC_SMOKE=1 bash run_all.sh smoke        # ~5 min, must produce non-zero accuracy
nohup bash run_all.sh > run.log 2>&1 &  # ~75 min full sweep
tail -f run.log
```

## Resume after pre-emption

If the instance is pre-empted mid-run, just restart:

```bash
nohup bash run_all.sh > run_resume.log 2>&1 &
```

Completed cells in `results/niah_4k/*.json` and `results/long_ctx_16k/*.json`
are skipped; only incomplete cells re-run.

## Expected runtime + cost

| Script | Cells | Est runtime | Est cost @ $1/hr |
|---|---|---|---|
| `niah_lb_kv_4k.py` | 12 cells (2 models × 3 CRs × 2 methods) | ~50 min | $0.85 |
| `lb_kv_long_context_16k.py` | 7 cells | ~30 min | $0.50 |
| **Total full sweep** | 19 cells | **~80 min** | **~$1.40** |

Smoke mode total: ~5 min, ~$0.10.

## Download results

After the run completes, on your local machine:

```bash
scp -r ssh<n>:/root/DeltaCache/experiments/rebuttal_round1/a100_camera_ready/results .
```

Or from the instance itself:

```bash
bash run_all.sh download <ssh-config-name>
```

## What gets fed back to the paper

After download, update `paper/neurips2026/main.tex` §7(6):

- Replace "the 4096-token NIAH ... is camera-ready" with the measured
  per-cell numbers (Mistral-7B + Llama-2-7B at CR=2, 4, 6).
- Add a new long-context row for LB-KV in §4.3's 16K table.
- Drop the page-overflow risk by also folding the "camera-ready"
  sentences into the trim plan in `PAGE_AUDIT.md`.

## Memory provisioning sanity

- 16K eager attention on Mistral-7B (8 KV heads, 32 layers):
  `8 * 16384^2 * 2B FP16 ≈ 4.3 GB per layer materialized peak`,
  but transformers eager attention computes layer-by-layer so transient
  peak is ~4 GB. Total budget at 4-bit weights + activations + KV ≈ 35 GB.
  A100 80GB has plenty of margin.
- 16K eager attention on Llama-2-7B (32 KV heads): ~17 GB per layer
  (4× more attention scores). Total ≈ 60 GB — still fits A100 80GB but
  tight; this is why we ship the 16K table at Mistral-7B only.

## Reproducibility checklist

- Random seed 42 (haystack construction, all cells).
- Mean-fill enabled for fair comparison.
- Same `layer_budget_kv.py` / `moend_baseline.py` from
  `experiments/rebuttal_round1/`. Pinned to the paper-experiments branch.
- HF model revisions: pulls latest `main` for Mistral-7B-Instruct-v0.2 and
  Llama-2-7b-chat-hf — record `git ls-remote` output if exact revision matters.
