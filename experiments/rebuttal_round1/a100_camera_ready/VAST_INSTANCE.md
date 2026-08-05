# Vast.ai instance recommendation

Mirroring the previous successful instance (35588280, 5.5h Qwen2.5-72B run).

## Search filter

```
GPU: A100 SXM4 80GB
GPU count: 1
RAM: ≥64 GB
Disk: ≥80 GB (we use ~30 GB but headroom matters)
CUDA: ≥12.1
Internet: ≥500 Mbps (model downloads ~20 GB)
Reliability: ≥99%
DLPerf: any
Price: ≤$1.50/hr (target $0.80-1.20)
```

## Recommended image

```
pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

Same image as the prior Qwen2.5-72B run; deltacache + bitsandbytes install
without additional CUDA toolkit work.

## Typical cost

| Run | Wall time | Cost @ $1/hr |
|---|---|---|
| Smoke (`DC_SMOKE=1`) | ~5 min | $0.10 |
| Full sweep | ~80 min | $1.40 |
| Buffer for download + retries | +20 min | +$0.30 |
| **Total budget** | ~2 hr | **~$2** |

If selecting a $0.80/hr A100 the total is ~$1.50.

## Anti-rework checklist

Before destroying the instance:

- [ ] `results/niah_4k/` has 12 cell JSONs (or 6 if Llama-2 access blocked)
- [ ] `results/long_ctx_16k/` has 7 cell JSONs (or 6 without full_kv)
- [ ] Aggregated bundle JSONs exist in `results/`
- [ ] All cells show `_saved_at` timestamp
- [ ] No `*_FAILED.json` files (or known/expected failures only)
- [ ] `scp -r` of the entire `results/` dir to local
- [ ] Cross-check at least one cell's `ppl_ratio_mean` or `overall_accuracy`
      against the smoke run
- [ ] Updated `paper/neurips2026/main.tex` §7(6) with the measured numbers
- [ ] Saved a memory entry (`reference_vastai_camera_ready.md`) with the new
      instance ID + key results

## How this differs from the prior run

| Aspect | Prior (Qwen2.5-72B) | Current (camera-ready) |
|---|---|---|
| Models | 1 (72B) | 2 (7B each) |
| GPU | A100 80GB | Same |
| Wall time | 5.5h | ~80 min |
| Tasks | PPL, NIAH, RULER, MMLU | PPL @ 16K, NIAH @ 4K |
| Method coverage | Suite-driven | Targeted: LB vs LB-KV |
| Resume | suite checkpoint manager | per-cell JSON checkpoint |
| Mode selector | `7b/13b/all/quick` in `a100_one_shot.sh` | `niah/long/all/smoke` |

## Common pitfalls + mitigations

| Pitfall | Mitigation |
|---|---|
| HF auth missing → hours wasted on Llama-2 download fail | `preflight.sh` checks before any model load |
| Wrong `transformers` version → cache API mismatch | `pip install transformers==4.57.6` (pin) |
| Eager attention OOM at 16K Llama-2 (32 KV heads) | Long-context script is Mistral-7B only |
| Pre-emption mid-run | Per-cell checkpoint + rerun resumes |
| Forgetting to download results | `bash run_all.sh download <host>` shortcut |
| Wrong git revision (no `layer_budget_kv`) | `preflight.sh` imports it explicitly |
| Random differs across runs | seed=42 hard-coded in haystack builder |
