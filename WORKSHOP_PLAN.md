# DeltaCache ICLR 2026 Workshop 投稿计划

**日期**: 2026-01-01
**策略**: Option B - 补充创新点后投稿Workshop

---

## 一、Workshop 选择分析

### 1.1 候选Workshop评估

| Workshop | 全称 | 匹配度 | 理由 |
|----------|------|--------|------|
| **SPOT** | Scaling Post-Training for LLMs | ⭐⭐⭐⭐⭐ | 涵盖系统优化、基础设施、算法设计 |
| **ICBINB** | I Can't Believe It's Not Better | ⭐⭐⭐⭐ | 可展示"真实场景收益有限"的发现 |
| **LLM Reasoning** | Reasoning and Planning for LLMs | ⭐⭐⭐ | 涵盖efficient inference，但重点在reasoning |
| **ES-Reasoning** | (待确认) | ⭐⭐⭐ | 可能涉及efficient scalable |
| DeLTa | Deep Generative Models | ⭐ | 关于生成模型，不匹配 |
| AIMS | (待确认) | ⭐⭐ | 信息不足 |

### 1.2 推荐: SPOT Workshop

**Workshop on Scaling Post-Training for LLMs**

**为什么选择SPOT?**

1. **主题高度匹配**:
   - 系统和基础设施优化 ✓
   - 算法设计 ✓
   - 数据中心方法 ✓

2. **组织者背景强**:
   - Ion Stoica (UC Berkeley, Databricks创始人)
   - Inderjit Dhillon (Google VP)
   - 系统研究背景，对KV cache优化友好

3. **截止日期合适**:
   - 论文提交: 2026年1月30日
   - 通知: 2026年2月27日
   - 距今约4周，有时间补充创新

### 1.3 备选: ICBINB Workshop

**I Can't Believe It's Not Better: Where LLMs Need to Improve**

如果选择展示"负面结果"角度：
- 真实场景(ShareGPT)下DeltaCache无加速(0.9x)
- 多轮对话场景收益有限
- 适用场景边界的发现

这个角度也有学术价值，但需要重新定位论文。

---

## 二、Option B: 创新点补充方案

### 2.1 可行的创新方向 (按难度排序)

#### 🟢 低难度 (1-2周可完成)

**创新1: Attention-Aware Eviction Policy**
```
核心思想: 根据attention score决定哪些KV cache更重要
- 高attention位置的KV更可能被后续token使用
- 结合IMPRESS的思想，但更轻量
```
**工作量**: 约100行代码
**预期收益**: 可能提升cache hit率10-20%

**创新2: 层级感知缓存 (Layer-Aware Caching)**
```
核心思想: 不同层的KV重要性不同
- 底层(early layers)捕获局部特征，变化大
- 高层(later layers)捕获语义，更稳定
- 对高层KV给予更高缓存优先级
```
**工作量**: 约50行代码
**预期收益**: 减少不必要的缓存，提升效率

#### 🟡 中等难度 (2-3周)

**创新3: CPU Offloading (类似HiCache)**
```
核心思想: GPU满时offload到CPU，而非直接evict
- 已有内存池设计支持
- 需要添加GPU↔CPU数据传输
- 参考SGLang HiCache的设计
```
**工作量**: 约300行代码
**预期收益**:
- 更大的有效cache容量
- 对长文档场景收益显著
- 与HiCache形成对比亮点

**创新4: Speculative Prefix Matching**
```
核心思想: 推测性地预加载可能匹配的前缀
- 分析历史访问模式
- 预测下一个请求可能的前缀
- 提前加载KV cache
```
**工作量**: 约200行代码
**预期收益**: 减少cold start延迟

#### 🔴 高难度 (需要更多时间)

**创新5: 分布式KV Cache共享**
```
核心思想: 跨进程/跨节点共享KV cache
- 使用共享内存或Redis
- 需要处理并发访问
- 参考LMCache设计
```
**工作量**: 500+行代码
**不建议**: 时间不足，且LMCache已做得很好

---

## 三、推荐实施方案

### 3.1 最小可行方案 (MVP)

选择 **创新1 + 创新2**，总工作量约1周：

```
DeltaCache + Attention-Aware Eviction + Layer-Aware Caching
```

**新卖点**:
- 不只是简单的LRU/LFU
- 结合模型内部信息(attention, layer)做智能缓存决策
- 与IMPRESS有相似思想但更轻量

### 3.2 推荐方案 (如果时间允许)

选择 **创新1 + 创新2 + 创新3**：

```
DeltaCache + Attention-Aware + Layer-Aware + CPU Offloading
```

**新卖点**:
- 层次化存储 (GPU → CPU)
- 智能eviction策略
- 与SGLang HiCache对比

### 3.3 论文重新定位

**原标题**:
DeltaCache: Incremental Computation-Aware KV Cache Management

**新标题建议**:
DeltaCache: Attention-Aware Hierarchical KV Cache for Efficient LLM Inference

**新摘要要点**:
1. 层次化存储 (GPU/CPU)
2. Attention-aware eviction
3. Layer-aware caching priority
4. 9.7x prefill speedup (长prefix场景)
5. 适用场景边界分析

---

## 四、实施时间线

### Week 1 (1/1 - 1/7)
- [ ] 实现 Attention-Aware Eviction
- [ ] 实现 Layer-Aware Caching
- [ ] 基础实验验证

### Week 2 (1/8 - 1/14)
- [ ] 实现 CPU Offloading
- [ ] 对比实验 (vs baseline, vs HiCache思想)
- [ ] 更新论文

### Week 3 (1/15 - 1/21)
- [ ] 完善实验
- [ ] 论文写作
- [ ] 内部review

### Week 4 (1/22 - 1/30)
- [ ] 论文polish
- [ ] 提交到SPOT Workshop (DDL: 1/30)

---

## 五、创新点技术设计

### 5.1 Attention-Aware Eviction

```python
class AttentionAwareEvictionPolicy:
    """根据历史attention score决定eviction优先级"""

    def compute_importance(self, node):
        # 获取该prefix在历史中的平均attention score
        avg_attention = node.metadata.get('avg_attention', 0.0)
        recency = time.time() - node.last_access
        frequency = node.access_count

        # 综合评分: attention高 + 最近访问 + 频繁访问 = 高优先级
        importance = (
            0.4 * avg_attention +      # attention权重
            0.3 * (1.0 / (recency + 1)) +  # 时间衰减
            0.3 * log(frequency + 1)   # 频率
        )
        return importance

    def select_victim(self, cache):
        # 选择importance最低的节点evict
        return min(cache.nodes, key=self.compute_importance)
```

### 5.2 Layer-Aware Caching

```python
class LayerAwareCacheManager:
    """对不同层采用不同的缓存策略"""

    def __init__(self, num_layers):
        self.num_layers = num_layers
        # 高层(语义层)给予更高权重
        self.layer_weights = [
            0.5 + 0.5 * (i / num_layers)
            for i in range(num_layers)
        ]

    def should_cache_layer(self, layer_idx, memory_pressure):
        """在内存压力下，优先保留高层KV"""
        weight = self.layer_weights[layer_idx]
        threshold = memory_pressure  # 0.0-1.0
        return weight > threshold

    def selective_cache(self, kv_cache, memory_pressure):
        """选择性缓存：压力大时只缓存高层"""
        cached = {}
        for layer_idx, kv in enumerate(kv_cache):
            if self.should_cache_layer(layer_idx, memory_pressure):
                cached[layer_idx] = kv
        return cached
```

### 5.3 CPU Offloading

```python
class HierarchicalMemoryPool:
    """层次化内存池: GPU -> CPU -> Evict"""

    def __init__(self, gpu_limit, cpu_limit):
        self.gpu_pool = GPUMemoryPool(gpu_limit)
        self.cpu_pool = CPUMemoryPool(cpu_limit)

    def allocate(self, size):
        # 优先GPU
        if self.gpu_pool.available >= size:
            return self.gpu_pool.allocate(size), 'gpu'

        # GPU满，尝试offload到CPU腾出空间
        if self.gpu_pool.used > 0:
            victim = self.select_victim(self.gpu_pool)
            self.offload_to_cpu(victim)
            return self.gpu_pool.allocate(size), 'gpu'

        # CPU分配
        if self.cpu_pool.available >= size:
            return self.cpu_pool.allocate(size), 'cpu'

        # 都满了，evict CPU中最不重要的
        self.evict_from_cpu()
        return self.cpu_pool.allocate(size), 'cpu'

    def offload_to_cpu(self, gpu_block):
        """GPU -> CPU"""
        cpu_block = self.cpu_pool.allocate(gpu_block.size)
        cpu_block.data = gpu_block.data.cpu()  # 异步传输
        gpu_block.cpu_backup = cpu_block
        self.gpu_pool.free(gpu_block)

    def prefetch_to_gpu(self, cpu_block):
        """CPU -> GPU (预取)"""
        gpu_block = self.gpu_pool.allocate(cpu_block.size)
        gpu_block.data = cpu_block.data.cuda(non_blocking=True)
        return gpu_block
```

---

## 六、预期结果

### 6.1 新实验设计

| 实验 | 目的 | 预期结果 |
|------|------|----------|
| Attention-Aware vs LRU | 验证智能eviction效果 | +10-20% cache hit |
| Layer-Aware效果 | 验证选择性缓存 | 减少30%内存占用 |
| CPU Offloading | 验证层次化存储 | 2x有效cache容量 |
| 长文档场景 | 展示最佳场景 | 10x+ speedup |

### 6.2 论文贡献更新

**原贡献**:
1. Prefix tree管理 (与ChunkAttention重叠)
2. Multiple eviction policies (工程贡献)

**新贡献**:
1. **Attention-aware eviction**: 首次将attention信息用于cache eviction决策
2. **Layer-aware selective caching**: 根据层级重要性选择性缓存
3. **Hierarchical storage**: GPU/CPU层次化存储 (轻量级实现)
4. **适用场景分析**: 明确边界条件

---

## 七、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 创新点实现失败 | 低 | 高 | 先实现MVP(创新1+2) |
| 实验效果不显著 | 中 | 中 | 调整参数，选择最佳场景展示 |
| Workshop拒稿 | 中 | 中 | 备选ICBINB(负面结果角度) |
| 时间不足 | 中 | 高 | 优先MVP，放弃CPU offloading |

---

## 八、决策建议

### 推荐方案

1. **目标Workshop**: SPOT (Scaling Post-Training for LLMs)
2. **创新补充**: Attention-Aware + Layer-Aware + CPU Offloading
3. **时间规划**: 3周开发 + 1周写作
4. **备选方案**: 如果SPOT拒稿，改投ICBINB(负面结果角度)

### 立即行动

1. 开始实现Attention-Aware Eviction
2. 同步修改论文框架
3. 确认SPOT Workshop的CFP细节

---

*计划创建于 2026-01-01*
