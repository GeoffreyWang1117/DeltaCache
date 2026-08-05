# 可行研究方向分析：论文结合 × 本机可跑

> 硬件约束：5950X / 128GB DDR4 / 2×RTX 3090 (24GB each)
> 可用模型：Llama-2-7B, TinyLlama-1.1B, Llama-3.1-8B, Mistral-7B, Qwen2.5
> 已有环境：h2o, sequoia, deltacache

---

## 硬件能力边界（先搞清楚能做什么）

| 能力 | 上限 |
|------|------|
| 单卡模型 | 7B (FP16) / 13B (INT4) |
| 双卡模型 | 13B (FP16, tensor parallel) |
| KV cache上限 | 7B@32K ≈ 8GB，可以跑；7B@128K ≈ 32GB，要offload到CPU |
| Batch size | 7B: batch 8-16 (短context)；batch 2-4 (长context) |
| CPU内存 | 84GB可用，足够做GPU-CPU offloading实验 |
| 不能做的 | 70B模型、multi-node、A100/H100特有的功能(如FP8) |

---

## 方向排序（按 创新性×可行性 综合评分）

### ⭐⭐⭐ 方向1：H2O + DeltaCache → 层自适应KV预算（Bridge 1加强版）

**创新点：** H2O对所有层用相同eviction budget，但层间sparsity差异巨大（Gini从0.705到0.925）。用per-layer Gini自动分配budget，高sparsity层少给、低sparsity层多给，总内存不变。

**为什么你能做：**
- h2o环境已有，H2O已复现，数据在手
- 只改budget分配代码，核心改动<100行
- 在7B和1.1B上跑，不需要大模型
- benchmark用LongBench/RULER的短-中长度任务即可

**实验计划：**
1. H2O原版 uniform budget (baseline)
2. Gini-based adaptive budget: `budget[l] = base × (1 - α × gini[l])`
3. 扫α从0到1，看perplexity和accuracy的pareto curve
4. 对比PyramidKV（已有论文，不同的per-layer信号）

**硬件需求：** 单卡3090，7B，context 2K-8K → 完全够用

**风险：** PyramidKV已经做了per-layer budget，需要说清差异（你用的是attention Gini，他们用的是不同信号）

**时间：1-2周出初步结果**

---

### ⭐⭐⭐ 方向2：KIVI + DeltaCache → 层自适应量化精度

**创新点：** KIVI对所有层统一用2-bit量化。但DeltaCache的实验表明层重要性差异5倍。可以对不同层用不同量化精度：不重要的层2-bit，重要的层4-bit或FP16，总内存和uniform 2-bit持平但质量更好。

**为什么你能做：**
- KIVI是tuning-free的，不需要训练
- 量化在CPU上做，3090的24GB够用
- DeltaCache的layer importance weights已经profiled好了
- 实验就是改量化bit数的layer分配

**实验计划：**
1. KIVI uniform 2-bit (baseline)
2. 层自适应：重要层4-bit，不重要层2-bit，总budget相同
3. 用DeltaCache的layer weight做分配 vs 用H2O的Gini做分配 vs random
4. 评估：perplexity, downstream task accuracy

**硬件需求：** 单卡3090即可，量化反而降低了内存需求

**和方向1的协同：** 可以把 per-layer eviction budget + per-layer quantization precision 统一到一个框架里 → "dual-adaptive KV management"

**时间：2-3周**

---

### ⭐⭐☆ 方向3：StreamingLLM + DeltaCache → Prefix-Aware Streaming

**创新点：** StreamingLLM = attention sinks + sliding window，但完全丢掉了中间context。DeltaCache可以把共享prefix的KV cache持久化。组合：sink tokens + cached prefix KV + sliding window for new tokens → 在streaming场景下保留prefix语义不丢失。

**为什么你能做：**
- StreamingLLM代码简单，核心就是保留开头几个token
- DeltaCache的trie index已有
- 实验场景：RAG streaming（prefix是document，user queries流式到来）
- 7B + 4K-8K context够了

**实验计划：**
1. StreamingLLM (baseline, loses prefix)
2. Full KV (baseline, memory explodes)
3. Prefix-aware streaming: keep prefix KV + sinks + sliding window
4. 评估：long-conversation quality (多轮对话perplexity不崩) + memory footprint

**硬件需求：** 单卡3090

**创新性评估：** 中等。idea比较直接，但实用价值高（RAG+streaming是真实场景）

**时间：2周**

---

### ⭐⭐☆ 方向4：Q-Hitter思路 + DeltaCache → 三信号KV管理

**创新点：** Q-Hitter用两个信号（attention score + quantization friendliness）决定evict谁。加入DeltaCache的第三个信号（layer importance），变成三维决策：
- Token维度：attention score（H2O）
- 量化维度：quantization friendliness（Q-Hitter）
- 层维度：layer importance weight（DeltaCache）

**为什么你能做：**
- H2O已复现，attention score现成
- quantization friendliness = 看token的outlier程度，实现不复杂
- layer importance weight已有
- 决策函数就是三个信号的加权组合

**实验计划：**
1. H2O单信号 (baseline)
2. Q-Hitter双信号 (baseline)
3. 三信号：score(token, layer) = w1×attn + w2×quant_friendly + w3×layer_importance
4. 扫w1,w2,w3的pareto surface

**硬件需求：** 单卡3090，7B

**风险：** 三个信号可能不是独立的（需要验证）；实验设计要干净

**时间：3-4周**

---

### ⭐⭐☆ 方向5：MagicDec fixed-window + 本机dual-3090 → 小batch投机解码

**创新点：** MagicDec证明了fixed 257-token draft KV在大batch下有效。但没人测过小batch（4-8）在consumer GPU上的效果。用双3090做tensor parallel，测MagicDec在7B/8B上的small-batch性能 → 回答一个问题：consumer GPU上投机解码值不值得？

**为什么你能做：**
- 双3090做tensor parallel跑7B
- MagicDec的SnapKV selection可以复现
- sequoia环境已有，tree verification code可以复用
- batch 4-8是你的硬件甜点

**实验计划：**
1. Autoregressive baseline (batch 1-8)
2. MagicDec-style fixed-window spec (batch 1-8)
3. 测throughput和latency的crossover point：batch多大时speculation开始赢？
4. 和Sequoia的on-chip结果对比

**硬件需求：** 双卡3090 tensor parallel

**价值：** 填补consumer GPU上的spec decoding数据空白，MagicDec论文只测了A100

**时间：3-4周**

---

### ⭐☆☆ 方向6：LESS + H2O + DeltaCache → 层自适应eviction + recurrence recovery

**创新点：** 方向1的extension。激进eviction后用LESS的small recurrence cache恢复部分丢失的信息。per-layer不同策略：高Gini层 → 激进eviction + recurrence，低Gini层 → 保守eviction + 无recurrence。

**硬件需求：** 单卡3090

**风险：** LESS的recurrence cache实现复杂度不确定；组合三个方法的engineering成本高

**时间：4-6周**

---

### ⭐☆☆ 方向7：Sequoia DP + network cost → 网络感知树优化（Bridge 2）

**创新点：** 在DP目标函数中加入网络传输项。

**硬件：** 可以simulation，不需要真实edge设备

**问题：** 商业价值不确定，场景太窄。建议先不做，除非组里对edge-cloud有兴趣

---

## 推荐组合策略

### 短期（1-2周）→ 面试后立即可开始
**方向1：层自适应KV预算**
- 改H2O代码，跑Gini-based budget allocation
- 出一组perplexity对比数据
- 写成技术备忘录给组里看

### 中期（3-4周）→ 第一个完整实验
**方向1 + 方向2 合并 → "Dual-Adaptive KV Management"**
- Per-layer adaptive eviction budget（方向1）
- Per-layer adaptive quantization precision（方向2）
- 统一框架：每层的(eviction ratio, quant bits)联合优化
- 和H2O、KIVI、Q-Hitter、PyramidKV做full comparison
- 这是一篇完整的paper idea

### 长期（如果join了）
- 方向4（三信号KV管理）深化
- 方向5（consumer GPU spec decoding benchmark）
- A100到手后做Bridge 3（TriForce prefix retrieval）

---

## 面试时怎么讲这些

> "我分析了组里的论文和我自己的工作，找到了几个可以在我现有硬件上立即开展的方向。最有前途的是把H2O的per-token attention sparsity和DeltaCache的per-layer importance结合起来做layer-adaptive KV budget。这个我已经有数据基础了，改H2O的代码大概一两周就能出初步结果。如果进一步结合KIVI的量化，可以做一个dual-adaptive框架——每层同时优化eviction ratio和quantization precision。这个方向不需要A100，我的3090就能跑完整实验。"

---

## 关键数据速查

| 信号 | 来源 | Peak层 | 值 | 在哪一端更强 |
|------|------|--------|-----|------------|
| Attention Gini | H2O复现 | Layer 4 | 0.925 | 早期层 |
| Layer importance | DeltaCache profiling | Layer 20 | 0.955 | 晚期层 |
| Correlation | 我的实验 | — | 0.194 | 弱相关→互补 |
| Eviction tolerance | H2O: 50%→20% budget | — | 只丢12% attn | 全局 |
| Quant sensitivity | KIVI论文 | Late layers | Higher | 晚期层 |

→ 早期层：高sparsity，可以激进evict + 低bit量化
→ 晚期层：低sparsity，需要保守evict + 高bit量化
→ 两个信号方向相反 → 联合优化空间大
