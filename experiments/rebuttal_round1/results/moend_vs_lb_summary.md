# MoE-nD vs LayerBudget — head-to-head

WikiText-2 PPL ratio (compressed/full), seq_len=1024, prefix=60%, n=6 chunks

## mistral_7b

| CR | full_kv | kivi_uniform | moend_perlayer | layer_budget |
|---|---|---|---|---|
| 2× | 1.0 | 1.0064 | 1.028 | 0.9995 |
| 3× | 1.0 | 1.0064 | 1.0295 | 1.0056 |
| 4× | 1.0 | 1.0064 | 1.0292 | 1.0204 |
| 6× | 1.0 | 1.0064 | 1.0378 | 1.2499 |

## llama2_7b: FAILED — CUDA out of memory. Tried to allocate 52.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 31.12 MiB is free. Process 1270231 has 11.65 GiB memory in use. Including non-PyTorch memory, this process has 11.85 GiB memory in use. Of the allocated memory 11.32 GiB is allocated by PyTorch, and 230.41 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
## Interpretation
- **moend_perlayer** is our faithful re-implementation of MoE-nD's per-layer (n_l, b_K_l, b_V_l) joint allocation framing (Sun et al. 2026).
- **layer_budget** uses shared b_l with inverted importance + closed-form coverage law.
- A close result on Mistral-7B suggests both joint framings are essentially equivalent at moderate CR; a divergence indicates the importance signal or the K/V split matters at the scale tested.