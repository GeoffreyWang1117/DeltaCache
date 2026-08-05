# DeltaCache vs 2025 SOTA 对比分析

**分析日期**: 2026-01-01
**目标会议**: ICML 2026

---

## 一、2025年 KV Cache 优化 SOTA 概览

### 1.1 主要系统对比

| 系统 | 机构 | 发表 | 核心方法 | 报告加速 |
|------|------|------|----------|----------|
| **vLLM APC** | UC Berkeley | 2023+ | PagedAttention + Automatic Prefix Caching | 基线系统 |
| **SGLang RadixAttention** | LMSYS | ICLR 2024 | Radix Tree + LRU eviction | 5-6.4x throughput |
| **ChunkAttention** | Microsoft | ACL 2024 | Prefix Tree + Two-Phase Partition | 3.2-4.8x kernel speedup |
| **LMCache** | CMU/MIT | EuroSys 2025 | KV Cache offloading + 分布式共享 | 3-15x throughput |
| **SGLang HiCache** | LMSYS | 2025 | Hierarchical Radix Tree (GPU/CPU/Disk) | 84% TTFT reduction |
| **IMPRESS** | - | USENIX FAST 2025 | Important KV selection + I/O优化 | 2.8x TTFT |
| **BatchLLM** | - | 2024 | Global prefix preprocessing | - |
| **DeltaCache** | 本文 | ICML 2026 (投稿) | Prefix Tree + Incremental Delta | 9.7x prefill, 19.5x TTFT |

### 1.2 技术路线分类

1. **Serving系统级别** (vLLM, SGLang, LMCache)
   - 完整的推理引擎，包含调度、batching
   - 需要部署整套系统
   - 功能丰富但集成成本高

2. **Kernel/算子级别** (ChunkAttention, FlashInfer)
   - 优化attention计算kernel
   - 需要替换底层实现

3. **中间件/库级别** (DeltaCache, LMCache)
   - 可与现有系统集成
   - 更灵活的部署方式

---

## 二、DeltaCache vs SOTA 详细对比

### 2.1 vs SGLang RadixAttention

| 维度 | SGLang | DeltaCache | 对比结论 |
|------|--------|------------|----------|
| **核心数据结构** | Radix Tree | Prefix Tree (Trie) | 类似 |
| **Eviction策略** | LRU only | LRU/LFU/Composite/Tiered/Adaptive | **DeltaCache更丰富** |
| **集成方式** | 完整serving系统 | 独立Python库 | **DeltaCache更轻量** |
| **报告加速** | 5-6.4x throughput | 9.7x prefill speedup | 指标不同，难以直接对比 |
| **TTFT改善** | 10% over vLLM | 19.5x over baseline | **DeltaCache baseline不同** |

**关键差异**:
- SGLang是完整serving系统，DeltaCache是轻量级库
- SGLang的5-6.4x是与vLLM对比的throughput，DeltaCache的9.7x是prefill speedup
- **实际上SGLang已成为工业界标准，DeltaCache在架构层面没有本质创新**

### 2.2 vs ChunkAttention

| 维度 | ChunkAttention | DeltaCache | 对比结论 |
|------|----------------|------------|----------|
| **数据结构** | Prefix Tree | Prefix Tree | **相同** |
| **核心创新** | Two-Phase Partition kernel | Multiple eviction policies | 不同关注点 |
| **加速效果** | 3.2-4.8x kernel speedup | 9.7x prefill speedup | 场景不同 |
| **发表时间** | ACL 2024 | ICML 2026投稿 | ChunkAttention更早 |

**关键问题**:
- **ChunkAttention在ACL 2024已经提出了prefix tree管理KV cache的方法**
- DeltaCache的prefix tree方法与ChunkAttention高度相似
- ChunkAttention专注于kernel优化，DeltaCache专注于eviction策略

### 2.3 vs LMCache

| 维度 | LMCache | DeltaCache | 对比结论 |
|------|---------|------------|----------|
| **核心功能** | KV cache offloading + 分布式共享 | Prefix caching + Eviction | 互补 |
| **报告加速** | 3-15x throughput | 9.7x prefill | LMCache更强 |
| **生态集成** | 集成vLLM/SGLang生产环境 | 独立库 | **LMCache更成熟** |
| **分布式支持** | ✓ 跨节点共享 | ✗ 单机 | **LMCache更强** |

### 2.4 vs vLLM Automatic Prefix Caching

| 维度 | vLLM APC | DeltaCache | 对比结论 |
|------|----------|------------|----------|
| **实现层级** | Serving引擎 | Python库 | 不同层级 |
| **页式管理** | PagedAttention | Prefix Tree | 不同方法 |
| **v1引擎优化** | Zero overhead prefix caching | - | vLLM持续优化 |
| **生产就绪** | ✓ 广泛使用 | ✗ 研究原型 | **vLLM更成熟** |

---

## 三、DeltaCache的真实定位

### 3.1 ❌ 不足以作为SOTA贡献

1. **Prefix Tree方法已有先例**
   - ChunkAttention (ACL 2024) 已提出prefix tree管理KV cache
   - SGLang RadixAttention (ICLR 2024) 使用Radix Tree

2. **性能不够突出**
   - LMCache报告15x throughput，DeltaCache仅9.7x prefill speedup
   - 没有与SGLang/ChunkAttention的直接公平对比

3. **工程成熟度差距大**
   - SGLang/vLLM/LMCache都是生产级系统
   - DeltaCache仍是研究原型

4. **真实场景收益有限**
   - ShareGPT多轮对话: 0.90x (无收益)
   - 长文档QA: 1.02x (几乎无收益)

### 3.2 ✓ 可能的贡献点

1. **多种Eviction策略对比** (Ablation study价值)
   - LRU vs LFU vs Composite vs Tiered vs Adaptive
   - 但这更适合作为工程实践而非学术创新

2. **轻量级库实现**
   - 不依赖完整serving系统
   - 可嵌入任何HuggingFace模型
   - 但这是工程贡献，非学术贡献

3. **适用场景分析**
   - 明确了长prefix vs 短context的收益边界
   - 但这是实验观察，非新方法

---

## 四、与2025 SOTA的创新性差距

### 4.1 技术创新对比

| 创新点 | 2025 SOTA | DeltaCache | 差距 |
|--------|-----------|------------|------|
| **分布式KV共享** | LMCache, llm-d | 无 | ❌ 缺失 |
| **层次化存储** | SGLang HiCache (GPU/CPU/Disk) | GPU only | ❌ 缺失 |
| **重要token选择** | IMPRESS | 无 | ❌ 缺失 |
| **Kernel优化** | ChunkAttention Two-Phase | 无 | ❌ 缺失 |
| **位置无关融合** | CacheBlend | 无 | ❌ 缺失 |

### 4.2 实验对比差距

| 实验 | 2025 SOTA论文 | DeltaCache | 差距 |
|------|---------------|------------|------|
| **模型规模** | LLaMA-70B, Mixtral-8x7B | TinyLlama-1.1B, Mistral-7B | ❌ 模型太小 |
| **硬件环境** | 8xH100, 多节点 | 2xRTX3090 | ❌ 硬件受限 |
| **Baseline对比** | vLLM, SGLang, TRT-LLM | HuggingFace only | ❌ Baseline弱 |
| **真实数据集** | ShareGPT, LMSYS-Chat | 合成数据为主 | ⚠️ 需加强 |

---

## 五、结论与建议

### 5.1 论文定位评估

| 评估维度 | 结论 |
|----------|------|
| **技术新颖性** | ⚠️ 中等 - Prefix Tree已有先例 (ChunkAttention, SGLang) |
| **实验充分性** | ⚠️ 中等 - 模型小，缺少SOTA对比 |
| **工程完成度** | ✓ 较好 - 可用的Python库实现 |
| **适合顶会** | ❌ 风险较高 - 可能被认为增量创新 |

### 5.2 建议方向

**Option A: 放弃投稿，重新定位**
- 当前工作与ChunkAttention/SGLang重叠度高
- 建议转向工程博客/技术报告

**Option B: 补充创新点后投稿**
可能的补充方向：
1. **层次化存储**: 添加CPU offloading (类似HiCache)
2. **分布式共享**: 支持跨进程/跨节点KV共享
3. **重要token选择**: 结合attention score选择性cache
4. **Kernel融合**: 优化attention计算

**Option C: 调整投稿目标**
- 系统类会议 (OSDI, SOSP) - 需要更完整系统
- 工程track (MLSys Industry Track)
- Workshop (NeurIPS Workshop)

### 5.3 核心问题

**DeltaCache是一个"原创点子"吗？**

❌ **不完全是**。核心思想 (prefix tree + KV cache复用) 已经在2024年的ChunkAttention和SGLang中被提出并广泛使用。

**DeltaCache相对2025 SOTA有优势吗？**

❌ **没有明显优势**。SGLang/vLLM/LMCache都是更成熟、功能更完整的系统，且有更好的性能报告。

---

## 六、参考资料

- [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
- [ChunkAttention (ACL 2024)](https://aclanthology.org/2024.acl-long.623/)
- [LMCache](https://arxiv.org/abs/2510.09665)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [SGLang HiCache](https://lmsys.org/blog/2025-09-10-sglang-hicache/)
- [IMPRESS (FAST 2025)](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)

---

*本报告基于2026年1月1日的公开资料分析*
