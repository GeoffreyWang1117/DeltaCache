# Prior-art context supplied to reviewers (state of the field as of 2026-08-04)

This is the concurrent/prior work a well-informed reviewer of this paper would
know about. Use it for probe P1 (existing alternatives) and P13 (venue fit).
Nothing here is a finding about the paper — judge the paper yourself.

## Directly overlapping: per-layer joint eviction + quantization allocation

- **RDKV — "Rate-Distortion Bit Allocation for Joint Eviction and Quantization of
  the KV Cache"** (ETH Zürich + Tsinghua, arXiv 2605.08317, 8 May 2026).
  Casts KV cache compression as a rate–distortion problem in which *"eviction and
  quantization are two end-points of the same bit allocation scheme."* Derives each
  token's/channel's weight from the distortion compression induces on the attention
  computation, then assigns bit-widths from full precision down to zero bits by
  reverse water-filling with Lagrangian relaxation, applied once after prefill.
  Training-free. Explicitly argues the two actions must be *"explored jointly rather
  than in a staged fashion."* Instantiates the allocation as a discrete knapsack over
  hardware-supported bit-widths, and realizes the mixed-bit cache with **TriZone**, a
  packed-decode layout fused into the attention kernel.
  Results: outperforms the best evaluated baseline by 9.1% on average on LongBench,
  RULER and InfiniteBench; recovers **97.81% of full-cache accuracy at 2.48% cache
  retention** on LongBench; **4.5× decode speedup and 1.9× peak memory reduction at
  128K context** vs full-cache FlashAttention-2.

- **HqeKV — "Towards Hybrid Quantization and Eviction for KV Cache in Long-Context
  LLM Inference"** (ACL 2026 Findings; code public). Hybrid framework over both
  quantization and eviction with an integrated optimizer that selects the compression
  action per cached element, plus a joint K–V importance metric. Reports output
  quality 40.53 → 49.98 under the same memory constraint.

- **MoE-nD — "Per-Layer Mixture-of-Experts Routing for Multi-Axis KV Cache
  Compression"** (arXiv 2604.17695, Apr 2026). Routes each layer to its own
  (eviction-ratio, K-bits, V-bits) tuple under a global memory budget via an
  offline-calibrated greedy solver. Matches an uncompressed 1.9 GB baseline at **14×
  compression** on a LongBench-v1 subset; +6 to +27 pts over the strongest per-layer
  quantization baseline on AIME.

- **ARKV.** Estimates per-layer original/quantization ratios from prefill-time
  attention statistics with scoring thresholds searched offline.

- **PolyKV — "Heterogeneous Retention and Allocation for KV Cache Compression"**
  (KAUST, arXiv 2606.15157, Jun 2026). Reformulates KV eviction as a *layer-wise
  design space* coupling two choices — which eviction pattern each layer uses and how
  much cache capacity it receives — under a shared memory constraint; offline
  calibration converts layer signals into a fixed heterogeneous strategy. On
  LongBench recovers 54.5% / 25.7% of the FullKV gap on LLaMA-3.1-8B / Qwen3-8B at a
  512-token average budget.

- **LKV** (arXiv 2605.06676, May 2026). End-to-end *learned* head-wise budgets and
  token selection.

## Top-venue 2026 results that set the current bar

- **STAR-KV** (ICML 2026 **Spotlight**). Adaptive low-rank compression with
  differentiable soft-thresholding rank control at head and block level, hybrid K/V
  decomposition, and low-rank-aware **mixed-precision quantization**. **20× full KV
  cache compression**, 6.9× faster attention, 3.1× generation throughput, custom GPU
  kernels.
- **KVTC — "KV Cache Transform Coding"** (ICLR 2026, NVIDIA). PCA decorrelation +
  adaptive quantization + entropy coding; bit widths assigned under a global bit
  budget to minimize reconstruction error. **20× (40×+ in some settings)** with
  reasoning and long-context accuracy retained, on AIME25, GSM8K, LiveCodeBench,
  LongBench, MATH-500, MMLU, Qasper, RULER.
- **DefensiveKV / Layer-DefensiveKV** (ICLR 2026). Defensive score aggregation; the
  Layer- variant adds AdaKV-style layer-wise budget allocation. RULER at 20% cache:
  SnapKV 39.0 → DefensiveKV 85.3 → Layer-DefensiveKV **91.4**.
- **TurboQuant** (ICLR 2026, Google). Near-optimal online vector quantization, 3-bit
  keys / 2-bit values, Triton kernels, vLLM integration.
- **"Adaptive KV-Cache Compression without Manually Setting Budget"** (ICLR 2026).
  Formalizes compression as scoring → allocation → selection with adaptive per-layer
  budgets; evaluates on GSM8K, RULER, **LongBench-v2**.
- **KVCompose** (2026). Composite tokens with a global allocation mechanism that
  adapts retention budgets across layers.

## Long-output / reasoning-model regime (now a major sub-field)

- **ThinKV** — thought-adaptive **hybrid quantization + eviction**: per-thought-type
  precision (8/4/2-bit, ~3.4 average bits) plus progressive eviction. Near-lossless
  at **<5% of the original KV cache**, up to 5.8× throughput, on DeepSeek-R1-Distill,
  GPT-OSS, QwQ-32B, AceReason, across AIME / MATH-500 / GSM8K / LiveCodeBench.
- **MixKVQ** — query-aware mixed-precision (BF16 / UINT4 / UINT2), 2.3–2.7 effective
  bits, on AIME'24–'25, MATH-500, GPQA-Diamond, LiveCodeBench.
- **InfoKV**, **VaSE**, **Adaptive Mass-Segmented KV**, **Reasoning Path
  Compression** — same regime.

## Evaluation-methodology results the community now expects papers to respect

- **"How Query Visibility Changes KV-Cache Compression Rankings: A Matched-Budget
  Audit"** (arXiv 2607.11942). 144,300 paired RULER records, 3 models, bootstrap
  B=50,000. Under **query-aware** compression (the literature default) four published
  methods beat trivial baselines; under **query-agnostic** compression (compress the
  context, *then* append the question — the deployment order) only KeyDiff wins and
  **SnapKV falls below a "keep start + recent window" trivial baseline (−0.066)**.
  Method gaps track query leakage into the scoring signal (SnapKV Δ=+0.198 → KeyDiff
  Δ=+0.011). Swapping the attention backend **sdpa → eager shifts RULER accuracy by
  −0.221, larger than most method gaps.** RULER's nominal "8192" overflows Gemma-2's
  positional budget by 30%, silently zeroing 7 of 13 subtasks with no compression.
- **"Ablation, Statistical Inference, and Validation for KV-Cache Compression"**
  (arXiv 2607.09683). Demands synthetic diagnostic regimes, separation of algorithmic
  from implementation variance, and multi-dimensional error geometry over
  accuracy-only reporting.
- **NVIDIA KVPress** is the de-facto harness: 30+ methods behind one API, a public
  HuggingFace leaderboard, a RULER / InfiniteBench / Loogle CLI, and built-in
  `PerLayerCompressionPress` and `QuantizedCache` support.

## Standard benchmarks in this area as of 2026

LongBench-v2 (v1 now considered dated), RULER (4K→128K, 13 subtasks), InfiniteBench,
SCBench (the KV-cache-centric benchmark: cache generation / compression / retrieval /
loading, multi-turn shared context), HELMET, NIAH, and for reasoning models AIME
2024/2025, MATH-500, GPQA-Diamond, LiveCodeBench. Effective bits/token is a commonly
reported common currency alongside compression ratio.
