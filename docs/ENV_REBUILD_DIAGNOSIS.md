# Environment rebuild — diagnosis
**Date:** 2026-08-04

## Headline: the environment is not broken. It is empty.

`RESEARCH_PORTFOLIO_2026H2.md` records "Python 环境全失" as the blocker for all supplementary experiments. That is not what is wrong here, and the distinction matters because it changes the fix from *days of dependency archaeology* to **one clean install**.

```
conda env: deltacache          created/wiped 2026-08-01 21:36
python:    3.11.15
size:      225 MB
installed: pip 26.1.2, setuptools 83.0.0, wheel 0.47.0, packaging 26.0   ← that is all
```

Eleven entries in `site-packages`. No torch, no transformers, no numpy. There is no broken dependency graph to untangle and no version conflict to resolve — the env was recreated three days ago and never populated.

My earlier `import transformers` failure was in **base** (`/home/coder-gw/miniconda3/bin/python`, Python 3.14.6), not in the project env. That was my error; the correct diagnosis is above.

---

## System state — all green

| Check | Result |
|---|---|
| GPUs | 2 × RTX 3090 (24 GB each), driver 595.84, CUDA 13.2, sm_86 |
| Disk | 1.8 TB free on `/`, 61 GB on `/tmp` |
| PyPI | reachable (HTTP 200); `transformers 4.57.6` present in the index |
| HuggingFace | reachable (HTTP 200) |
| Reference envs | `longspec` has a proven-working torch 2.5.1 + bitsandbytes 0.49.2 pairing on these same GPUs |

**⚠ The GPUs are in use.** A `fisherkd` job is running: PID 2986930 (6.5 GB, GPU 0) and PID 3062344 (0.6 GB GPU 0 + 5.1 GB GPU 1), ~30% utilisation on both. Free VRAM is **~16.5 GB on GPU 0, ~19.4 GB on GPU 1**. Do not kill it. A 7B model in fp16 (~14 GB) fits on GPU 1 only; plan the sweep for `cuda:1` or wait for that job.

---

## What to install, and why these pins

There is **no `requirements.txt`, no `environment.yml`, no lockfile** anywhere in the repo. The A100 preflight script installed unpinned (`pip install -q torch transformers datasets bitsandbytes accelerate`), which is exactly how the May run ended up hitting the bitsandbytes/transformers incompatibility documented in `reference_a100_run.md`. Recreate with pins this time.

### transformers — pin to 4.57.6, do not take 5.x

Latest on PyPI is 5.14.1. The suite will not survive it unmodified:

- `experiments/suite/eval_utils.py:158-190` constructs `DynamicCache()` with no arguments and calls `cache.update(k, v, layer_idx)`. Both changed in 5.x.
- `experiments/suite/eval_utils.py:48` passes `output_attentions=need_attention`, and `model_pool.py:233` forces `attn_implementation="eager"`. Eager-attention + `output_attentions` handling was reworked in 5.x.
- `deltacache/integrations/hf_cache.py` *does* carry a forward-compat shim (`hasattr(cache, "key_cache")` → falls back to `cache.layers[i].keys`), so that one file may survive — but the experiment harness does not have the same shim.

4.57.6 is the version validated in the May A100 camera-ready run. Take it.

### bitsandbytes — install it, but do not use it for the new work

`reference_a100_run.md` records that **bnb 0.43.1 is incompatible with transformers 4.57.6** ("Calling `to()` is not supported for 4-bit quantized models"); **0.49.2** is the working version, and `longspec` confirms 0.49.2 runs on these GPUs.

But note what the review panel found: *every* experiment in this project ran on **4-bit weight-quantized models**, never disclosed as a caveat on the quality claims. For the faithfulness sweep that is a confound we should drop — a KL between a compressed-cache run and a reference run is only meaningful if the weights are identical and unquantized. **Run the new sweep in fp16.** 7B fp16 ≈ 14 GB, which fits GPU 1's free 19.4 GB. Keep bnb installed only for reproducing old cells.

### Proposed install

```bash
E=/home/coder-gw/miniconda3/envs/deltacache

$E/bin/pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1

$E/bin/pip install \
    transformers==4.57.6 \
    tokenizers safetensors \
    accelerate datasets \
    sentencepiece protobuf \
    numpy scipy \
    bitsandbytes==0.49.2 \
    click tqdm matplotlib \
    pytest pytest-asyncio pytest-cov ruff mypy

$E/bin/pip install -e /home/coder-gw/Projects/DeltaCache
```

torch 2.5.1+cu124 runs fine under a CUDA 13.2 driver (drivers are backward-compatible) and is the build already proven on this machine in `longspec` alongside bnb 0.49.2. The one pin worth watching at install time is torch↔transformers-4.57.6; pip will surface it if unhappy, and the fallback is a newer cu12x torch.

### Verification after install

1. `import torch, transformers; torch.cuda.is_available()`; report `torch.__version__`, `torch.version.cuda`, device count.
2. `pytest tests/ -x -q` — the repo claims 189 unit tests; a pass here confirms the library half.
3. **The real smoke test:** run the existing PPL path on one small cell. PPL and the faithfulness measurement share the forward-pass code path, so a green PPL cell is the direct precondition for the sweep. Do **not** smoke-test via LongBench/GSM8K/MMLU — those harnesses are independently broken (evaluation-harness feasibility study, not published in this repository) and would give a false negative.
4. Freeze the result: `pip freeze > requirements.lock.txt` and commit it. The absence of this file is what caused the May incident.

**Estimated wall-clock:** 10–20 minutes, dominated by the ~2.5 GB torch wheel download.

---

## Revised critical-path estimate

The portfolio note treats the environment as a multi-day blocker. It is not. Corrected:

| Step | Cost |
|---|---|
| Env rebuild | **10–20 min** (was assumed: days) |
| Verify via PPL smoke cell | ~30 min |
| Faithfulness sweep (17 methods × 7 models × 4 CRs × N≥50 texts, forward-pass only) | GPU-bound; schedule around the `fisherkd` job |
| LongBench/GSM8K/MMLU repair | 5–7 weeks — **not on the critical path for the reframe** |

---

## What was actually installed — 2026-09-06

The rebuild above was executed. Two deviations from the plan, both worth knowing before
the faithfulness sweep is scheduled.

**torch is 2.14.0+cu130, not the pinned 2.5.1+cu124.** `optimum-quanto` requires
`torch>=2.6.0`, so installing it silently upgraded torch and pulled the cuda-toolkit 13.0.3
wheels. Nothing is broken — `torch.cuda.is_available()` is True on both 3090s — but the
2.5.1 pin existed to match the bitsandbytes 0.49.2 pairing proven in `longspec`, and that
pairing no longer holds. bitsandbytes is **not installed**, so no old 4-bit cell can be
reproduced in this env as it stands. Decide before reproducing anything: either drop
optimum-quanto and go back to 2.5.1, or keep 2.14 and re-verify bnb separately.

**transformers 4.57.6 as pinned.** It exposes both quantized-cache APIs
(`QuantoQuantizedLayer` and `QuantoQuantizedCache`), but `is_optimum_quanto_available`
lives in `transformers.utils.import_utils`, not in `cache_utils` where 4.46 kept it.

**ninja must be on PATH, not merely installed.** optimum-quanto builds a C++ extension on
first dequantize, and torch's extension loader looks for the `ninja` executable. Running
`$E/bin/python` without `$E/bin` on PATH fails with "Ninja is required to load C++
extensions" even though the package is present. Prefix commands with
`PATH="$E/bin:$PATH"` or activate the env.

Installed for the byte audit only: torch, transformers==4.57.6, tokenizers, safetensors,
numpy, accelerate, optimum-quanto, ninja, pytest, ruff, and the package in editable mode.
Not installed: bitsandbytes, datasets, sentencepiece, protobuf, scipy, matplotlib, mypy.
`requirements.lock.txt` is frozen at the repository root.

Pre-existing unrelated packages in this env (vastai, aiohttp, borb) have unmet dependencies
of their own and produce pip resolver warnings. They are not ours and were left alone.
