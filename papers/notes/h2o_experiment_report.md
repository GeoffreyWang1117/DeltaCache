# H2O Reproduction Experiment Report

## Hardware
- GPU: NVIDIA RTX 3090 (24GB)
- Models tested: Llama-2-7b-chat-hf (13.5GB fp16), TinyLlama-1.1B-Chat (2.2GB fp16)

---

## Experiment 1: Heavy Hitter Observation (Core Hypothesis Validation)

### Llama-2-7b-chat-hf (32 layers, 87 tokens input)

| Layer | Top 5% captures | Top 10% captures | Top 20% captures | Top 50% captures |
|-------|:-:|:-:|:-:|:-:|
| 0 (first) | 18.5% | 29.8% | 49.8% | 81.9% |
| 4 | 87.0% | 88.6% | 91.3% | 96.3% |
| 8 | 81.9% | 84.4% | 88.5% | 95.3% |
| 12 | 80.2% | 82.7% | 86.6% | 94.3% |
| 16 | 78.5% | 81.1% | 85.4% | 93.3% |
| 20 | 84.2% | 85.9% | 88.9% | 94.8% |
| 24 | 89.4% | 90.8% | 92.8% | 96.6% |
| 28 | 86.8% | 88.6% | 91.0% | 95.7% |
| **Average** | **75.8%** | **79.0%** | **84.3%** | **93.5%** |

### TinyLlama-1.1B-Chat (22 layers, 259 tokens input)

| Layer | Top 5% captures | Top 10% captures | Top 20% captures | Entropy ratio |
|-------|:-:|:-:|:-:|:-:|
| 0 (first) | 16.8% | 28.8% | 46.8% | 0.94 (uniform) |
| 4 | 90.6% | 91.7% | 93.5% | 0.17 (very sparse) |
| 8 | 85.9% | 87.6% | 90.2% | 0.24 |
| 12 | 59.6% | 65.0% | 73.2% | 0.58 |
| 16 | 65.3% | 69.8% | 76.7% | 0.51 |
| 20 | 74.0% | 77.1% | 82.0% | 0.39 |
| 21 (last) | 67.0% | 71.0% | 76.6% | 0.49 |
| **Average** | **67.5%** | **72.2%** | **78.6%** | — |

### Key Findings
1. **H2O's core hypothesis is validated**: Top 20% tokens capture ~79-84% of attention
2. **Layer 0 is unique**: Nearly uniform attention (entropy ratio ~0.94), H2O has limited benefit
3. **Early-middle layers (4-8) most sparse**: Highest Heavy Hitter concentration (90%+ in top 20%)
4. **Pattern is consistent across model sizes**: Both 1.1B and 7B show same phenomenon
5. **Larger model is more sparse**: 7B captures 84.3% vs 1.1B captures 78.6% at 20% budget

---

## Experiment 2: H2O Eviction Quality (TinyLlama-1.1B)

| Budget | Tokens Kept | Attention Retained | Cross-layer Stability |
|--------|:-:|:-:|:-:|
| 100% (full) | 259 | 100% | — |
| 50% | 129 | **85.1%** | 69.0% |
| 20% | 51 | **73.4%** | 60.8% |
| 10% | 25 | **68.3%** | 60.0% |
| 5% | 12 | **64.7%** | 66.7% |

### Key Findings
1. **Diminishing returns**: 50%→20% budget loses only 12% attention; 20%→5% loses only 9%
2. **Cross-layer stability ~60%**: Heavy Hitters are moderately stable across layers but not identical
3. **Even 5% budget retains 65% attention**: Extreme compression still captures majority

---

## Experiment 3: H2O vs DeltaCache Layer Analysis (TinyLlama-1.1B)

| Layer | H2O Gini (sparsity) | H2O Top20% Capture | DeltaCache Weight |
|-------|:-:|:-:|:-:|
| 0 | 0.431 (low) | 46.8% | 0.182 (low) |
| 4 | **0.925** (very high) | **93.5%** | 0.356 |
| 8 | 0.887 | 90.2% | 0.579 |
| 12 | 0.705 | 73.2% | 0.773 |
| 16 | 0.738 | 76.7% | 0.894 |
| 20 | 0.794 | 82.0% | **0.955** (high) |

**Correlation between H2O effectiveness and DeltaCache layer weight: 0.194 (weak)**

### Key Insight (Interview-Critical)
- **H2O effectiveness peaks in early-middle layers** (layer 4: Gini=0.925, capture=93.5%)
- **DeltaCache weight peaks in late layers** (layer 20: weight=0.955)
- **Weak correlation (0.19) = they capture DIFFERENT signals**
  - H2O says: "Layer 4's KV is most compressible (few tokens dominate)"
  - DeltaCache says: "Layer 20's KV is most valuable (semantically important)"
  - **Combined approach adds value**: Use H2O to compress early layers aggressively, preserve late layers more carefully

---

## Summary: Interview-Ready Numbers

```
1. "Top 20% tokens capture ~80-84% of attention" (validated on 7B and 1.1B)

2. "H2O at 20% budget retains 73% attention mass with 61% cross-layer stability"

3. "H2O and DeltaCache are complementary (correlation=0.19):
    - H2O identifies WHICH tokens matter (content-aware, token-level)
    - DeltaCache identifies WHICH layers matter (structure-aware, layer-level)
    - Natural combination: aggressive H2O in early layers + preserve late layers"

4. "Larger models are more sparse → H2O is MORE effective on 7B than 1.1B"
```
