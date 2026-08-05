"""xKV-style baseline: per-layer (and grouped) SVD truncation of KV.

Faithful reimplementation of the core xKV mechanism (Chang et al. 2025):
  - Single SVD per layer: SVD truncate to rank r, multiply back, eval downstream
  - Cross-layer SVD (xKV proper): concatenate consecutive G layers along seq dim,
    SVD jointly to rank r, project back to each layer

Byte-CR accounting follows xKV's paper: low-rank stores U (sl, r) and V (r, n_h*d).
Memory CR = (sl * n_h * d * 2 bytes) / (r * (sl + n_h*d) * 2 bytes)
         = sl * n_h * d / (r * (sl + n_h*d))

For Mistral-7B: sl=1024, n_kv*d = 8*128 = 1024
  CR=2 → rank = sl * n_kv*d / (2 * (sl + n_kv*d)) = 1048576 / 4096 = 256
  CR=3 → rank ≈ 170
  CR=4 → rank ≈ 128
  CR=6 → rank ≈ 85

Registers methods "xkv_single" (group_size=1) and "xkv_g2" (group_size=2).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

DELTACACHE_ROOT = Path("/home/coder-gw/Projects/DeltaCache")
sys.path.insert(0, str(DELTACACHE_ROOT))
sys.path.insert(0, str(DELTACACHE_ROOT / "experiments"))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from baselines.base import BaselineMethod, register_baseline  # noqa: E402


def fake_svd_truncate(tensor: Tensor, rank: int) -> Tensor:
    """SVD truncate (sl, nh*d) at rank r, multiply back. tensor shape: (1, sl, nh, d)."""
    bs, sl, nh, d = tensor.shape
    flat = tensor.reshape(bs, sl, nh * d)  # (bs, sl, nh*d)
    flat32 = flat.float()
    U, S, Vh = torch.linalg.svd(flat32, full_matrices=False)
    rank = max(1, min(rank, U.shape[-1]))
    U = U[:, :, :rank]
    S = S[:, :rank]
    Vh = Vh[:, :rank, :]
    sqrtS = torch.sqrt(S)
    U_scaled = U * sqrtS.unsqueeze(1)
    Vh_scaled = sqrtS.unsqueeze(-1) * Vh
    approx = torch.matmul(U_scaled, Vh_scaled).to(tensor.dtype)
    return approx.reshape(bs, sl, nh, d)


def cross_layer_svd_truncate(tensors: List[Tensor], rank: int) -> List[Tensor]:
    """Concatenate G layers along seq dim, SVD truncate jointly, split back.
    Each tensor shape: (1, sl, nh, d). Returns list of approximated tensors.
    """
    bs, sl, nh, d = tensors[0].shape
    G = len(tensors)
    stacked = torch.cat([t.reshape(bs, sl, nh * d) for t in tensors], dim=1)  # (1, G*sl, nh*d)
    flat32 = stacked.float()
    U, S, Vh = torch.linalg.svd(flat32, full_matrices=False)
    rank = max(1, min(rank, U.shape[-1]))
    U = U[:, :, :rank]
    S = S[:, :rank]
    Vh = Vh[:, :rank, :]
    sqrtS = torch.sqrt(S)
    U_scaled = U * sqrtS.unsqueeze(1)
    Vh_scaled = sqrtS.unsqueeze(-1) * Vh
    approx = torch.matmul(U_scaled, Vh_scaled).to(tensors[0].dtype)  # (1, G*sl, nh*d)
    approx = approx.reshape(bs, G * sl, nh, d)
    out = []
    for g in range(G):
        out.append(approx[:, g * sl:(g + 1) * sl, :, :].contiguous())
    return out


def rank_from_cr(cr: float, sl: int, nh: int, d: int, group_size: int = 1) -> int:
    """Rank that yields target CR per the xKV byte accounting.

    For group_size=1: CR = sl * nh * d / (r * (sl + nh*d))
    For group_size=G: CR = G*sl * nh*d / (r * (G*sl + nh*d))
      (single SVD over G*sl × nh*d, stored as U(G*sl, r) + V(r, nh*d))
    """
    L_eff = group_size * sl
    full = L_eff * nh * d  # numerator (elements of original)
    # We want full / (r * (L_eff + nh*d)) = cr  ⇒  r = full / (cr * (L_eff + nh*d))
    r = int(full / (cr * (L_eff + nh * d)))
    return max(1, r)


@register_baseline
class XKVSingleSVD(BaselineMethod):
    name = "xkv_single"
    category = "low_rank"
    requires_attention = False
    is_per_layer = True
    reference = "Chang et al., xKV 2025 (Single SVD, layer_group_size=1)"

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        # full_keys: (L, sl, nh, d)
        L, sl, nh, d = full_keys.shape
        rank = rank_from_cr(compression_ratio, sl, nh, d, group_size=1)
        out = []
        for l in range(L):
            k = full_keys[l:l + 1].permute(0, 2, 1, 3).contiguous()  # (1, sl, nh, d) — already
            v = full_values[l:l + 1].permute(0, 2, 1, 3).contiguous()
            # Wait: full_keys[l] shape is (sl, nh, d). Let me reshape correctly:
            kk = full_keys[l].unsqueeze(0)  # (1, sl, nh, d)
            vv = full_values[l].unsqueeze(0)
            kk_approx = fake_svd_truncate(kk, rank)
            vv_approx = fake_svd_truncate(vv, rank)
            indices = torch.arange(sl, device=full_keys.device)
            out.append((kk_approx, vv_approx, indices))
        return out


@register_baseline
class XKVGroup2(BaselineMethod):
    """xKV with consecutive layer groups of size 2 (cross-layer SVD)."""
    name = "xkv_g2"
    category = "low_rank"
    requires_attention = False
    is_per_layer = True
    reference = "Chang et al., xKV 2025 (Cross-layer, layer_group_size=2)"

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        L, sl, nh, d = full_keys.shape
        rank = rank_from_cr(compression_ratio, sl, nh, d, group_size=2)
        out: List[Tuple[Tensor, Tensor, Tensor]] = [None] * L  # type: ignore
        for g_start in range(0, L, 2):
            g_end = min(g_start + 2, L)
            G = g_end - g_start
            ks = [full_keys[l].unsqueeze(0) for l in range(g_start, g_end)]
            vs = [full_values[l].unsqueeze(0) for l in range(g_start, g_end)]
            if G == 1:
                k_apx = [fake_svd_truncate(ks[0], rank)]
                v_apx = [fake_svd_truncate(vs[0], rank)]
            else:
                k_apx = cross_layer_svd_truncate(ks, rank)
                v_apx = cross_layer_svd_truncate(vs, rank)
            for li, l in enumerate(range(g_start, g_end)):
                indices = torch.arange(sl, device=full_keys.device)
                out[l] = (k_apx[li], v_apx[li], indices)
        return out
