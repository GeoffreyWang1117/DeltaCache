# Paired-bootstrap 95% CI on PPL-ratio differences

`n_boot=10000`, `seed=42`. CI is for `mean(ppl_ratio[A] - ppl_ratio[B])`
across paired chunks. Negative = A better than B.

## mistral_7b

| CR | A | B | mean(A) | mean(B) | Δ (A−B) | 95% CI | Sig A better? |
|---|---|---|---|---|---|---|---|
| 2× | layer_budget | kivi_uniform | 0.9995 | 1.0064 | -0.0069 | [-0.0101, -0.0030] | Yes |
| 2× | layer_budget | moend_perlayer | 0.9995 | 1.0280 | -0.0284 | [-0.0853, +0.0007] | No (overlap) |
| 2× | layer_budget | full_kv | 0.9995 | 1.0000 | -0.0005 | [-0.0017, +0.0008] | No (overlap) |
| 2× | moend_perlayer | kivi_uniform | 1.0280 | 1.0064 | +0.0215 | [-0.0085, +0.0763] | No (overlap) |
| 3× | layer_budget | kivi_uniform | 1.0055 | 1.0064 | -0.0008 | [-0.0031, +0.0020] | No (overlap) |
| 3× | layer_budget | moend_perlayer | 1.0055 | 1.0295 | -0.0238 | [-0.0812, +0.0067] | No (overlap) |
| 3× | moend_perlayer | kivi_uniform | 1.0295 | 1.0064 | +0.0230 | [-0.0081, +0.0797] | No (overlap) |
| 4× | layer_budget | kivi_uniform | 1.0204 | 1.0064 | +0.0141 | [+0.0026, +0.0302] | No (A worse) |
| 4× | layer_budget | moend_perlayer | 1.0204 | 1.0292 | -0.0086 | [-0.0728, +0.0339] | No (overlap) |
| 4× | moend_perlayer | kivi_uniform | 1.0292 | 1.0064 | +0.0227 | [-0.0078, +0.0775] | No (overlap) |
| 6× | layer_budget | kivi_uniform | 1.2499 | 1.0064 | +0.2438 | [+0.1327, +0.3401] | No (A worse) |
| 6× | layer_budget | moend_perlayer | 1.2499 | 1.0377 | +0.2126 | [+0.0765, +0.3373] | No (A worse) |
| 6× | moend_perlayer | kivi_uniform | 1.0377 | 1.0064 | +0.0312 | [-0.0026, +0.0879] | No (overlap) |
| 2× | layer_budget | kivi_uniform | 0.9995 | 1.0064 | -0.0069 | [-0.0101, -0.0030] | Yes |
| 2× | layer_budget | full_kv | 0.9995 | 1.0000 | -0.0005 | [-0.0017, +0.0008] | No (overlap) |
| 2× | layer_budget | xkv_single | 0.9995 | 1.0779 | -0.0786 | [-0.0978, -0.0586] | Yes |
| 2× | layer_budget | xkv_g2 | 0.9995 | 1.0771 | -0.0777 | [-0.0874, -0.0700] | Yes |
| 2× | xkv_single | kivi_uniform | 1.0779 | 1.0064 | +0.0717 | [+0.0483, +0.0924] | No (A worse) |
| 2× | xkv_g2 | xkv_single | 1.0771 | 1.0779 | -0.0010 | [-0.0157, +0.0135] | No (overlap) |
| 3× | layer_budget | kivi_uniform | 1.0055 | 1.0064 | -0.0008 | [-0.0031, +0.0020] | No (overlap) |
| 3× | layer_budget | xkv_single | 1.0055 | 1.1119 | -0.1065 | [-0.1276, -0.0884] | Yes |
| 3× | layer_budget | xkv_g2 | 1.0055 | 1.1291 | -0.1237 | [-0.1490, -0.1026] | Yes |
| 3× | xkv_single | kivi_uniform | 1.1119 | 1.0064 | +0.1056 | [+0.0874, +0.1260] | No (A worse) |
| 3× | xkv_g2 | xkv_single | 1.1291 | 1.1119 | +0.0172 | [+0.0029, +0.0362] | No (A worse) |
| 4× | layer_budget | kivi_uniform | 1.0204 | 1.0064 | +0.0141 | [+0.0026, +0.0302] | No (A worse) |
| 4× | layer_budget | xkv_single | 1.0204 | 1.1553 | -0.1351 | [-0.1491, -0.1180] | Yes |
| 4× | layer_budget | xkv_g2 | 1.0204 | 1.2094 | -0.1893 | [-0.2577, -0.1354] | Yes |
| 4× | xkv_single | kivi_uniform | 1.1553 | 1.0064 | +0.1492 | [+0.1275, +0.1744] | No (A worse) |
| 4× | xkv_g2 | xkv_single | 1.2094 | 1.1553 | +0.0542 | [+0.0124, +0.1138] | No (A worse) |
| 6× | layer_budget | kivi_uniform | 1.2499 | 1.0064 | +0.2438 | [+0.1327, +0.3401] | No (A worse) |
| 6× | layer_budget | xkv_single | 1.2499 | 1.3549 | -0.1048 | [-0.4142, +0.1222] | No (overlap) |
| 6× | layer_budget | xkv_g2 | 1.2499 | 1.3650 | -0.1145 | [-0.5011, +0.1407] | No (overlap) |
| 6× | xkv_single | kivi_uniform | 1.3549 | 1.0064 | +0.3487 | [+0.1909, +0.6000] | No (A worse) |
| 6× | xkv_g2 | xkv_single | 1.3650 | 1.3549 | +0.0097 | [-0.0474, +0.0900] | No (overlap) |

## llama2_7b

| CR | A | B | mean(A) | mean(B) | Δ (A−B) | 95% CI | Sig A better? |
|---|---|---|---|---|---|---|---|
| 2× | layer_budget | kivi_uniform | 0.9956 | 1.0059 | -0.0102 | [-0.0170, -0.0039] | Yes |
| 2× | layer_budget | moend_perlayer | 0.9956 | 1.0270 | -0.0319 | [-0.0806, -0.0034] | Yes |
| 2× | layer_budget | full_kv | 0.9956 | 1.0000 | -0.0044 | [-0.0076, -0.0009] | Yes |
| 2× | moend_perlayer | kivi_uniform | 1.0270 | 1.0059 | +0.0217 | [-0.0088, +0.0749] | No (overlap) |
| 3× | layer_budget | kivi_uniform | 1.0039 | 1.0059 | -0.0019 | [-0.0089, +0.0062] | No (overlap) |
| 3× | layer_budget | moend_perlayer | 1.0039 | 1.0463 | -0.0428 | [-0.0856, -0.0020] | Yes |
| 3× | moend_perlayer | kivi_uniform | 1.0463 | 1.0059 | +0.0409 | [-0.0032, +0.0854] | No (overlap) |
| 4× | layer_budget | kivi_uniform | 1.0209 | 1.0059 | +0.0149 | [+0.0101, +0.0221] | No (A worse) |
| 4× | layer_budget | moend_perlayer | 1.0209 | 1.0882 | -0.0677 | [-0.1815, +0.0147] | No (overlap) |
| 4× | moend_perlayer | kivi_uniform | 1.0882 | 1.0059 | +0.0826 | [+0.0043, +0.1951] | No (A worse) |
| 6× | layer_budget | kivi_uniform | 1.0990 | 1.0059 | +0.0927 | [+0.0600, +0.1277] | No (A worse) |
| 6× | layer_budget | moend_perlayer | 1.0990 | 1.1364 | -0.0381 | [-0.1488, +0.0520] | No (overlap) |
| 6× | moend_perlayer | kivi_uniform | 1.1364 | 1.0059 | +0.1308 | [+0.0436, +0.2664] | No (A worse) |
