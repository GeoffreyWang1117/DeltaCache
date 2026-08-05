# Mean-Fill KL Generalization

Mistral-7B paper claim: zero-fill logits-KL = 0.249, mean-fill = 0.009 (96% reduction).
This script checks whether the reduction generalizes.

| Model | Zero-fill KL | Mean-fill KL | Reduction |
|---|---|---|---|
| llama2_7b | 0.6696 | 0.0636 | **90.5%** |
| llama3.1_8b | 0.4092 | 0.0175 | **95.7%** |

**Outcome interpretation:** if reduction ≥80% on both models, the mean-fill claim is generalizable — abstract can stay at '96% on Mistral, ≥80% across other tested models'. If reduction <50% on either, scope down.