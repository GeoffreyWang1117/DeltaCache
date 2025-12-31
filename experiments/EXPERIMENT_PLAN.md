# DeltaCache 论文实验计划

## 硬件环境
- GPU: 2x NVIDIA RTX 3090 (24GB VRAM each, 48GB total)
- 单卡可运行: 7B模型 (FP16 ~14GB)
- 双卡可运行: 13B模型 (需tensor parallel)

## 实验模型选择

| 模型 | 参数量 | 显存占用 | 用途 |
|------|--------|---------|------|
| TinyLlama-1.1B | 1.1B | ~2GB | 快速迭代、正确性验证 |
| Qwen2-1.5B | 1.5B | ~3GB | 小模型验证 |
| Mistral-7B-v0.1 | 7B | ~14GB | **主要基准模型** |
| Llama-2-7B | 7B | ~14GB | 重要对比基线 |
| Llama-2-13B (INT8) | 13B | ~13GB | 可选：大模型验证 |

## 实验设计

### 实验1: 正确性验证
**目标**: 证明缓存KV产生与非缓存相同的输出

**方法**:
- 对比: `output_with_cache` vs `output_without_cache`
- 指标: 最大绝对误差、相对误差、输出token一致性
- 模型: TinyLlama (快速), Mistral-7B (主要)

### 实验2: 系统提示缓存
**目标**: 测量共享系统提示的加速效果

**设置**:
- 系统提示长度: 100, 200, 500, 1000 tokens
- 用户查询数量: 100
- 测量: TTFT, 吞吐量, 内存占用, 缓存命中率

### 实验3: RAG文档缓存
**目标**: 测量文档问答场景的缓存效果

**设置**:
- 文档长度: 256, 512, 1024, 2048 tokens
- 每文档查询数: 5-10
- 数据集: 真实文档(Wikipedia摘要)

### 实验4: Few-shot学习
**目标**: 测量共享示例的缓存效果

**设置**:
- Few-shot示例数: 1, 3, 5, 10
- 分类任务: SST-2情感分类
- 测量: 准确率 + 延迟

### 实验5: 淘汰策略对比
**目标**: 对比5种淘汰策略

**设置**:
- 固定内存限制 (如4GB缓存)
- 相同workload
- 策略: LRU, LFU, Composite, Tiered, Adaptive

### 实验6: 可扩展性分析
**目标**: 测量系统可扩展性

**设置**:
- 缓存大小: 1GB, 2GB, 4GB, 8GB, 16GB
- 并发请求: 1, 2, 4, 8
- 测量: 查询延迟、内存开销

### 实验7: 基线对比 (如果vLLM可用)
**目标**: 与vLLM PagedAttention对比

**设置**:
- 相同模型、相同workload
- 测量: 延迟、吞吐量、内存效率

## 指标定义

| 指标 | 定义 |
|------|------|
| TTFT | Time to First Token (首token延迟) |
| TPOT | Time Per Output Token (每token生成时间) |
| 缓存命中率 | cache_hits / total_lookups |
| Token复用率 | cached_tokens / total_tokens |
| 加速比 | baseline_time / deltacache_time |

## 输出物

1. `experiments/real_model_benchmark.py` - 真实模型实验
2. `experiments/correctness_validation.py` - 正确性验证
3. `experiments/results/paper/` - 论文数据和图表
4. LaTeX表格和matplotlib图表

## 时间规划

- Day 1-2: 实现真实模型适配器 + 正确性验证
- Day 3-4: 系统提示/RAG/Few-shot实验
- Day 5-6: 淘汰策略对比 + 可扩展性
- Day 7: 整理结果、生成图表
