# H2O: Heavy-Hitter Oracle — Reading Notes (Experiments Complete)

## Paper Info
- **Title:** H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
- **Venue:** NeurIPS 2023
- **Authors:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, ... Beidi Chen
- **Links:** arXiv:2306.14048 | github.com/FMInference/H2O

---

## Core Observation (Fig 1-2) — 已验证
- [x] Attention score分布高度不均匀：少数token获得大部分attention
- [x] 这些token称为"Heavy Hitters"，在不同layer中保持一定稳定性 (cross-layer stability ~61%)
- [x] 实验发现：仅保留~20% KV cache，捕获84% attention mass (7B), 79% (1.1B)

## 算法 (Section 3)
- [x] Eviction policy: 维护一个固定大小的KV cache
- [x] 每个step：计算当前所有token的cumulative attention score
- [x] 保留策略：Top-k heavy hitter tokens + 最近的recent_window个token
- [x] Greedy eviction：每次新token进来，evict score最低的旧token
- [x] 理论：formulated as dynamic submodular problem，有approximation guarantee

## 关键公式
- Cumulative attention: A_j = Σ_i attn(q_i, k_j) for all query i attending to key j
- Eviction decision: evict argmin_{j ∉ recent} A_j
- Budget分配: budget = heavy_ratio * seq_len + recent_window

## 实验结果 (已复现)

### Llama-2-7b-chat (32 layers, 87 tokens input)
- [x] Top 5% captures 75.8% attention (average across layers)
- [x] Top 10% captures 79.0%
- [x] Top 20% captures 84.3%
- [x] Top 50% captures 93.5%
- [x] Layer 0: uniform (18.5% at 5%), Layer 24: most sparse (89.4% at 5%)

### TinyLlama-1.1B (22 layers, 259 tokens)
- [x] 20% budget retains 73.4% attention mass
- [x] Cross-layer stability: ~60-69%
- [x] Layer 0 entropy ratio 0.94 (uniform), Layer 4 Gini 0.925 (very sparse)
- [x] Even 5% budget retains 64.7% attention

### H2O vs DeltaCache Layer Analysis
- [x] H2O effectiveness peaks at early layers (layer 4: Gini=0.925, capture=93.5%)
- [x] DeltaCache weight peaks at late layers (layer 20: weight=0.955)
- [x] **Correlation = 0.194 (weak) → complementary signals**

## 与DeltaCache对比 (面试核心, 有实验数据支撑)

| 维度 | H2O | DeltaCache |
|------|-----|------------|
| 优化目标 | 减少KV cache总量 | 复用共享prefix的KV |
| 粒度 | Token-level | Prefix-level |
| 决策依据 | Attention score (content-aware) | Layer position (structure-aware) |
| 最佳layer | 早期layer (layer 4: 93.5% capture) | 晚期layer (layer 20: weight 0.955) |
| 适用场景 | 所有long-context inference | Shared-prefix场景 (RAG, system prompt) |
| 互补性 | 可以在prefix cache内部做H2O eviction | 可以在H2O基础上加layer优先级 |

## 组合方案 (面试加分点, 有实验依据)
- H2O peaks at early layers, DeltaCache peaks at late layers → 自然分工
- Early layers: aggressive H2O eviction (Gini高, 少量token就够)
- Late layers: preserve more KV (semantically important, DeltaCache insight)
- Combined: per-layer adaptive budget, 不是uniform 20%
- 实现：`budget[layer] = base * (1 - α * gini[layer])`, 高Gini层给更少budget

## 局限性 (诚实评估)
- [x] H2O需要在inference时维护attention score累积，有计算overhead
- [x] Heavy Hitter pattern可能随context变化 (cross-layer stability只有61%)
- [x] 对于short sequence, eviction overhead > 节省的memory
- [x] Monkey-patching approach有兼容性问题 (我们实验中遇到device mismatch)

## Key Code Paths (已读 ✓)
- [x] `h2o_hf/run_text_generation.py` — main entry, H2O注入pipeline
- [x] `h2o_hf/utils_lm_eval/modify_llama.py` — LlamaAttention_heavy_hitter
- [x] 核心: `convert_kvcache_llama_heavy_hitter()` — monkey-patch attention
