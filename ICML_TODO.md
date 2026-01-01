# ICML 2026 投递工作计划

**截止日期**: 2026年1月28日 (约4周)
**当前状态**: 论文已更新，实验数据部分完成
**上次更新**: 2025-12-31

---

## 已完成工作

### 实验
- [x] 创建 `experiments/icml_experiments.py` - 专注 prefix length scaling
- [x] TinyLlama-1.1B 实验完成
  - Prefix Scaling: 2.2x-9.7x speedup (50-1500 tokens)
  - TTFT: 17-18x speedup
  - RAG: 1.75x speedup, 87% token reuse
- [x] Mistral-7B (8-bit) 实验完成
  - Prefix Scaling: 3.8x-9.7x speedup
  - TTFT: 19.5x speedup
  - RAG: 2.5x speedup, 87% token reuse

### 论文更新
- [x] 摘要更新 (9.7x speedup, 19.5x TTFT)
- [x] 添加理论分析 section (复杂度 O(n²) → O((n-m)²))
- [x] 更新 Performance 表格 (prefix length scaling)
- [x] 更新 TTFT 表格 (17-19x speedup)
- [x] 更新 RAG 表格 (1.75x-2.5x speedup)
- [x] 更新 Conclusion
- [x] 添加量化模型正确性警告

---

## 待办工作 (按优先级排序)

### P0 - 必须完成

#### 1. 修复 vLLM 对比实验
- **问题**: NCCL 分布式通信超时
- **方案A**: 设置 `NCCL_P2P_DISABLE=1` 或 `NCCL_IB_DISABLE=1`
- **方案B**: 使用 vLLM 离线模式，分别测量后对比
- **文件**: `experiments/vllm_comparison_fixed.py`

#### 2. 全精度模型正确性验证
- **问题**: 8-bit Mistral 只有 10% 正确性 (max diff 0.6)
- **方案**: 在 fp16 Mistral-7B 上验证 100% 正确性
- **所需显存**: 约14GB (需要单独使用一张 RTX 3090)

#### 3. Multi-turn 实验优化
- **问题**: 当前实验显示 0.92x-1.05x (几乎无加速)
- **原因**: 每个 turn 的 context 较短，开销 > 收益
- **方案**: 使用更长的 system prompt + 更多 turn 数

### P1 - 重要但可 rebuttal 补充

#### 4. SGLang RadixAttention 对比
```bash
pip install sglang
# 参考 SGLang benchmark 脚本
```

#### 5. 真实数据集验证
- ShareGPT: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- LMSys-Chat-1M: https://huggingface.co/datasets/lmsys/lmsys-chat-1m

### P2 - 锦上添花

#### 6. 专业图表
- Speedup vs Prefix Length 曲线图 (matplotlib)
- 可选: 系统架构图更新

#### 7. ICML 格式调整
- 检查是否符合 ICML 2026 模板
- 页数限制检查 (当前12页，ICML 通常限制8页正文)

---

## 实验结果汇总

### Prefix Length Scaling (核心结果)
| Model | Prefix | Speedup | TTFT Speedup |
|-------|--------|---------|--------------|
| TinyLlama-1.1B | 50 | 2.2x | - |
| TinyLlama-1.1B | 500 | 5.6x | 18.7x |
| TinyLlama-1.1B | 1500 | 9.7x | - |
| Mistral-7B (8-bit) | 50 | 3.8x | - |
| Mistral-7B (8-bit) | 500 | 6.7x | 19.5x |
| Mistral-7B (8-bit) | 1500 | 9.7x | - |

### 正确性
| Model | Precision | Accuracy | Max Diff |
|-------|-----------|----------|----------|
| TinyLlama-1.1B | fp16 | 80% | 1.56e-02 |
| Mistral-7B | 8-bit | 10% | 0.6 |

---

## 文件位置

- 论文: `paper/main.tex`
- 实验脚本: `experiments/icml_experiments.py`
- 实验结果: `experiments/results/paper/`
  - `icml_tinyllama_1.1b_chat_v1.0.json`
  - `icml_mistral_7b_v0.1_8bit.json`

---

## 下次工作建议

1. 先运行 vLLM 对比实验 (设置环境变量解决 NCCL 问题)
2. 在有足够显存时运行 fp16 Mistral-7B 正确性验证
3. 调整论文格式以符合 ICML 要求
