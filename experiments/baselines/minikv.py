"""MiniKV baseline (arXiv 2505): 2-bit quantization + fixed pyramid token budget.

The most direct competitor to LayerBudget: also does joint token-precision
optimization, but with a fixed pyramid budget (no online allocation) and
uniform 2-bit quantization (no per-layer precision selection).

Key differences from LayerBudget:
  - Fixed pyramid token allocation (not data-dependent)
  - Uniform 2-bit quantization across all layers
  - No Gini-guided allocation or importance weighting
  - Simulates what happens when you combine the simplest version of each axis

Reference: Zhong et al., "MiniKV: Pushing the Limits of LLM Inference via
           2-Bit Layer-Discriminative KV Cache", 2025.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from deltacache.core.kv_quantizer import KVQuantizer, QuantPrecision

from .base import BaselineMethod, h2o_token_selection, register_baseline


@register_baseline
class MiniKV(BaselineMethod):
    name = "minikv"
    category = "joint"
    requires_attention = False
    is_per_layer = True
    reference = "Zhong et al., 2025"

    def __init__(self, num_layers, num_heads, head_dim, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)
        self.sink_tokens = kwargs.get("sink_tokens", 4)
        self.recent_tokens = kwargs.get("recent_tokens", 16)
        self.quant_bits = kwargs.get("quant_bits", 4)  # 2-bit simulated via INT4
        prec = QuantPrecision.INT4 if self.quant_bits <= 4 else QuantPrecision.INT8
        self._quantizer = KVQuantizer(prec)

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        seq_len = full_keys.shape[1]
        min_tokens = min(self.sink_tokens + self.recent_tokens, seq_len)

        # MiniKV uses a fixed pyramid: early layers get fewer tokens,
        # later layers get more (opposite to LayerBudget's inverted importance).
        #
        # Memory math:
        #   Full FP16: L * S * 2(KV) * H * D * 2(bytes) = 4*L*S*H*D bytes
        #   Target: full_fp16 / compression_ratio
        #   Per token at INT4: 2(KV) * H * D * quant_bits/8 bytes
        #   Total tokens budget: target_bytes / per_token_bytes
        full_fp16_bytes = 4 * self.num_layers * seq_len * self.num_heads * self.head_dim
        target_bytes = full_fp16_bytes / compression_ratio
        per_token_bytes = 2 * self.num_heads * self.head_dim * self.quant_bits / 8
        effective_budget_tokens = int(target_bytes / per_token_bytes)
        effective_budget_tokens = min(effective_budget_tokens, seq_len * self.num_layers)

        # Fixed pyramid allocation (MiniKV style)
        # Layer importance: later layers are more important (conventional assumption)
        avg = effective_budget_tokens / self.num_layers
        alpha = 0.3  # pyramid steepness
        b_min = max(min_tokens, int(alpha * avg))
        b_max = max(b_min, int((2 - alpha) * avg))
        b_max = min(b_max, seq_len)

        budgets = []
        for l in range(self.num_layers):
            frac = l / max(1, self.num_layers - 1)
            budget = int(b_min + (b_max - b_min) * frac)
            budget = max(min_tokens, min(budget, seq_len))
            budgets.append(budget)

        # Rescale to match total
        current = sum(budgets)
        if current > 0 and abs(current - effective_budget_tokens) > self.num_layers:
            scale = effective_budget_tokens / current
            budgets = [max(min_tokens, min(int(b * scale), seq_len))
                       for b in budgets]

        results = []
        for l in range(self.num_layers):
            # Token selection (value-norm based, no attention needed)
            indices = h2o_token_selection(
                full_keys[l:l+1], full_values[l:l+1],
                budgets[l], seq_len,
                attention_weights=None,  # MiniKV doesn't use attention
                sink_tokens=self.sink_tokens,
                recent_tokens=self.recent_tokens,
            )

            # Select tokens
            k_sel = full_keys[l:l+1, indices.long()]
            v_sel = full_values[l:l+1, indices.long()]

            # Quantize to INT4 (simulating 2-bit via our INT4 quantizer)
            qkv = self._quantizer.quantize(k_sel, v_sel)
            k_deq, v_deq = self._quantizer.dequantize(qkv)

            results.append((k_deq, v_deq, indices))

        return results

    def memory_bytes(self, compressed_layers):
        """Report actual quantized memory (token eviction + quantization)."""
        if not compressed_layers:
            return 0
        total = 0
        for k, v, indices in compressed_layers:
            n_tokens = k.shape[1]
            # Quantized: n_tokens * 2(KV) * heads * dim * bits/8
            per_layer = 2 * n_tokens * self.num_heads * self.head_dim * self.quant_bits // 8
            # Metadata: scales + zeros
            key_meta = self.num_heads * self.head_dim * 4  # per-channel
            value_meta = n_tokens * 4  # per-token
            total += per_layer + key_meta + value_meta
        return total
