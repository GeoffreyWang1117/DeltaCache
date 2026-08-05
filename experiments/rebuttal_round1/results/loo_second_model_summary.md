# LOO Second-Model Findings

Goal: confirm or reject the 'early layers are the eviction bottleneck' principle on architectures other than Mistral-7B.

| Model | Arch | L | Early-LOO mean | Late-LOO mean | Verdict |
|---|---|---|---|---|---|
| llama2_13b | MHA | 40 | +0.0304 | +0.0930 | **late-bottleneck** |
| qwen2.5_14b | — | — | FAILED | — | CUDA out of memory. Tried to allocate 136.00 MiB. GPU 0 has  |

**Reading:** higher LOO improvement at early-bin layers ⇒ restoring an early layer recovers more PPL ⇒ early layers are the eviction bottleneck (matches Mistral-7B finding).
If both models show 'early-bottleneck', the inverted-importance principle generalizes; if one or both show 'late-bottleneck' or 'indeterminate', the principle is Mistral-specific and the paper should scope the claim down.