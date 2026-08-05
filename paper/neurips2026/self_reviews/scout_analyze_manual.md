# /scout analyze — manual LLM relevance grading

**Date:** 2026-05-03
**Note:** Automated scout analyze failed (Groq 403 rate-limit, Gemini 503, Claude/ChatGPT key-loading bug). Grading produced manually from scan abstracts + TLDRs.

## Grading rubric

- **Must Cite (5/5)** — direct competitor or foundational work; not citing would be a reviewer-noticeable gap.
- **Consider (4/5)** — closely related; citing strengthens positioning but not strictly required.
- **Consider (3/5)** — same subfield, less directly aligned; a one-sentence mention suffices.
- **Optional (2/5)** — adjacent work, only if space permits.
- **False Positive (1/5)** — wrong field; ignore.

## Must Cite (5/5)

### 1. MoE-nD: Per-Layer Mixture-of-Experts Routing for Multi-Axis KV Cache Compression (Sun et al. 2026)
**Role:** Direct competitor (concurrent work).
**Already added** to references.bib + §2 differentiation paragraph.

### 2. No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization (Yang et al. 2024)
**Role:** Closest *prior* art for joint quant+evict framing.
**Already added** to references.bib + §2.

### 3. EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving (Feng et al. 2025)
**Role:** Same joint-eviction-quant principle, request-level vs our per-layer.
**Already added** to references.bib + §2.

### 4. xKV: Cross-Layer SVD for KV-Cache Compression (Chang et al. 2025)
**Role:** Orthogonal axis (cross-layer); composable with our work.
**Already added** to references.bib + §2.

### 5. KV Pareto: Systems-Level Optimization of KV Cache and Model Compression (Patwari et al. EACL 2026)
**Role:** Long-context joint optimization; positioning-relevant.
**Already added** to references.bib (no §2 mention yet — recommend adding).

## Consider (4/5)

### 6. HCAttention: Extreme KV Cache Compression via Heterogeneous Attention Computing (Yang et al. 2025)
**Role:** Claims 4M-token context on Llama-3-8B 80GB. Different angle (heterogeneous attn vs per-layer alloc).
**Already added** to references.bib (consider §2 mention).

### 7. PoD: Compressing KV Cache for Long-Context LLM Inference with Inter-Layer Attention Similarity (Ma et al. 2024)
**Role:** Inter-layer redundancy — directly relevant to our cross-layer-coherence limitation.
**Already added** to references.bib (consider §2 mention).

### 8. KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization (Li et al. AAAI 2026)
**Role:** Layer-importance-driven mixed precision — already cited.

### 9. SimLayerKV: A Simple Framework for Layer-Level KV Cache Reduction (Zhang et al. 2024)
**Role:** Layer-level reduction baseline.
**Action:** Add to bib + brief §2 mention.

### 10. SVDq: 1.25-bit and 410× Key Cache Compression (Hong et al. 2025)
**Role:** Aggressive quantization ceiling.
**Action:** Add to bib if space; cite as comparison reference for quant-only methods.

## Consider (3/5)

### 11. Expected Attention: KV Cache Compression by Estimating Attention from Future Queries (Devoto et al. 2025)
**Role:** Future-query-distribution importance.
**Action:** Optional bib add.

### 12. ClusterKV: Manipulating LLM KV Cache in Semantic Space (Sun et al. 2025)
**Role:** Semantic-space compression — adjacent.

### 13. SmallKV: Small Model Assisted Compensation of KV Cache Compression (Zhao et al. 2025)
**Role:** Cross-model KV compensation — adjacent.

### 14. AttentionPredictor: Temporal Patterns Matter for KV Cache Compression (2025)
**Role:** Temporal-pattern-driven importance.

### 15. FAEDKV: Infinite-Window Fourier Transform for Unbiased KV Cache Compression (Li et al. 2025)
**Role:** Fourier-based axis; non-overlapping with our work.

### 16. Token-Aware Sensitivity-Guided Quantization (Zhang et al. 2026)
**Role:** Static mixed-precision; differs from our online approach.

### 17. LORC: Low-Rank Compression for LLMs KV Cache (Zhang et al. NeurIPS 2024 Workshop)
**Role:** Low-rank axis.

## False Positive (1/5)

These appeared in the scan due to arXiv category overlap and should be ignored:

- Geometric densities and compression radii of knot types (math.GT)
- Energy-Aware Quantum-Enhanced Computing Continuum (cs.ET)
- Scale-freeness under node removal (physics.soc-ph)
- HERMES++ (autonomous-driving world model)
- ResiHMR (3D human mesh recovery)
- TAFA-GSGC (point cloud geometry compression)
- Action Motifs (human body movement representation)
- MoCapAnything V2 (motion capture)
- Diffusion-OAMP (image transmission)
- Beyond Gaussian Bottlenecks (vision-transformer encoding)
- Auto-FlexSwitch (model merging)
- Latent-GRPO (latent reasoning)
- DPN-LE (LLM neuron editing)
- KV-RAPTOR (retrieval QA, distinct domain)

## Summary

- **5 papers added to references.bib + §2 in this session**: MoE-nD, xKV, EVICPRESS, KV Pareto, No Token Left Behind, PoD, HCAttention (7 total).
- **2-3 more should be added if space**: SimLayerKV, SVDq, Expected Attention.
- **Most critical remaining action**: head-to-head experimental comparison vs MoE-nD on a shared task (LongBench or AIME). Without this, the §2 "complementary work" framing reads as deflection rather than positioning.
