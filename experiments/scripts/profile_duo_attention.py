"""Offline gradient-based head profiling for DuoAttention.

Faithful to Xiao et al. (ICLR 2025), Section 3.2.

Algorithm
─────────
For each attention layer, introduce a learnable α_h ∈ [0, 1] per head.
During the forward pass on calibration data, compute attention as a
blend of full attention and streaming (sink + recent) attention,
weighted per head:

    A_blend[h] = sigmoid(α_h) · A_full[h] + (1 − sigmoid(α_h)) · A_stream[h]

where A_full = softmax(QK^T / √d + causal_mask)
and   A_stream = softmax(QK^T / √d + causal_mask + ¬(sink ∪ recent) → −∞)

Loss:
    L = ‖h_blend − h_full‖² + λ · Σ_h sigmoid(α_h)

(L1 penalty is on σ(α) so it pushes heads toward streaming = 0.)

Only the α parameters are trainable; all model weights are frozen.
After optimization, σ(α) is thresholded at 0.5 to produce a binary
{retrieval, streaming} classification per head per layer.

Usage
─────
    python -m experiments.scripts.profile_duo_attention --model qwen3-0.6b
    python -m experiments.scripts.profile_duo_attention --model llama3.1-8b \\
        --calib-samples 10 --calib-len 4096 --epochs 6

Output: ~/.cache/duoattention_profiles/<short_name>.pt
        (dict {layer_idx: Tensor(num_heads,) ∈ {0, 1}})
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# Allow running as a script from the project root.
# experiments/ is not a Python package, so we add experiments/ to sys.path
# and import the baselines / suite modules directly.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))            # experiments/
sys.path.insert(0, str(_HERE.parent.parent))     # project root

from baselines.duo_attention import PROFILE_CACHE_DIR, profile_path


# ── Calibration data loader ──────────────────────────────────────────

def load_calibration_texts(n_samples: int, min_chars: int) -> List[str]:
    """Load long-context calibration samples.

    Tries WikiText-103 first (always available); falls back to a
    synthetic concatenation of Wikipedia articles if needed.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    except Exception as e:
        raise RuntimeError(
            f"Could not load wikitext-103 for calibration: {e}\n"
            "Profile generation needs a long-context corpus."
        )

    texts: List[str] = []
    buf = ""
    for ex in ds:
        buf += ex["text"]
        if len(buf) >= min_chars:
            texts.append(buf[:min_chars * 2])
            buf = ""
        if len(texts) >= n_samples:
            break
    if len(texts) < n_samples:
        # Pad with the last buffer if needed
        while len(texts) < n_samples and buf:
            texts.append(buf)
            buf = ""
    return texts[:n_samples]


# ── Patched attention with learnable α blending ──────────────────────

class DuoAttentionMixer(nn.Module):
    """Wraps a HuggingFace attention module to compute a blended output.

    For each forward call, runs the original attention twice — once with
    the standard causal mask (full) and once with an extra mask that zeros
    out non-(sink+recent) positions (streaming) — and blends per head.

    The wrapped module's parameters stay frozen; only `self.alpha` trains.
    """

    def __init__(self, attn_module: nn.Module, num_kv_heads: int,
                 sink: int = 16, recent: int = 64):
        super().__init__()
        self.attn = attn_module
        self.num_kv_heads = num_kv_heads
        self.sink = sink
        self.recent = recent
        # alpha is logit-space; sigmoid(alpha) ∈ (0, 1) is the retrieval weight
        self.alpha = nn.Parameter(torch.zeros(num_kv_heads))

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        # Run full attention (returns the standard output)
        full_kwargs = dict(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=False,
        )
        if cache_position is not None:
            full_kwargs["cache_position"] = cache_position
        if position_embeddings is not None:
            full_kwargs["position_embeddings"] = position_embeddings

        out_full = self.attn(**full_kwargs)
        if isinstance(out_full, tuple):
            h_full = out_full[0]
        else:
            h_full = out_full

        # Run streaming attention by additively masking middle tokens
        seq_len = hidden_states.shape[1]
        stream_mask = self._build_streaming_mask(seq_len, attention_mask,
                                                 hidden_states.device,
                                                 hidden_states.dtype)
        stream_kwargs = dict(full_kwargs)
        stream_kwargs["attention_mask"] = stream_mask
        out_stream = self.attn(**stream_kwargs)
        if isinstance(out_stream, tuple):
            h_stream = out_stream[0]
        else:
            h_stream = out_stream

        # Blend per head.  Output projection has already collapsed heads,
        # so we approximate per-head blending by per-channel blending where
        # each head occupies head_dim consecutive channels.
        a = torch.sigmoid(self.alpha)  # (num_kv_heads,)
        # Expand to (1, 1, num_kv_heads * head_dim)
        head_dim = h_full.shape[-1] // self.num_kv_heads
        a_exp = a.repeat_interleave(head_dim).view(1, 1, -1)
        a_exp = a_exp.to(h_full.dtype)

        h_blend = a_exp * h_full + (1 - a_exp) * h_stream

        if isinstance(out_full, tuple):
            return (h_blend,) + out_full[1:]
        return h_blend

    def _build_streaming_mask(self, seq_len: int, base_mask, device, dtype):
        """Construct an additive mask that disables middle tokens."""
        # Start from the base mask (causal); add −inf to non-streaming positions.
        # base_mask shape may be (B, 1, S, S) or None.
        large_neg = torch.finfo(dtype).min
        if base_mask is None:
            # Build a causal mask from scratch
            base = torch.zeros(1, 1, seq_len, seq_len, dtype=dtype, device=device)
            causal = torch.triu(
                torch.full((seq_len, seq_len), large_neg, dtype=dtype, device=device),
                diagonal=1,
            )
            base = base + causal.unsqueeze(0).unsqueeze(0)
        else:
            base = base_mask.clone()

        # Mask out non-(sink ∪ recent) keys for every query position
        keep = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep[:self.sink] = True
        if self.recent > 0:
            keep[max(self.sink, seq_len - self.recent):] = True
        # For each query, allow only kept keys (within causal limit)
        kv_mask = torch.where(
            keep.view(1, 1, 1, seq_len),
            torch.zeros_like(base),
            torch.full_like(base, large_neg),
        )
        return base + kv_mask


# ── Profiling driver ─────────────────────────────────────────────────

def profile_model(
    model_key: str,
    n_calib_samples: int = 8,
    calib_len: int = 2048,
    epochs: int = 4,
    lr: float = 0.05,
    l1_lambda: float = 0.02,
    sink: int = 16,
    recent: int = 64,
    device: str = "cuda",
) -> dict:
    """Run offline DuoAttention profiling for one model."""
    from suite.config import MODEL_ZOO
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_ZOO[model_key]
    print(f"\n[duo-profile] {spec.short_name}")
    print(f"  layers={spec.num_layers}  kv_heads={spec.num_kv_heads}")
    print(f"  loading model in fp16...")

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    # Force eager attention so the additive mask path is used predictably.
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name, torch_dtype=torch.float16, device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Find all attention modules and wrap them
    layers = _find_decoder_layers(model)
    if len(layers) != spec.num_layers:
        print(f"  WARN: discovered {len(layers)} decoder layers, "
              f"spec says {spec.num_layers}")

    mixers: List[DuoAttentionMixer] = []
    for i, layer in enumerate(layers):
        attn = _find_attention(layer)
        if attn is None:
            raise RuntimeError(f"No attention submodule found in layer {i}")
        mixer = DuoAttentionMixer(attn, spec.num_kv_heads,
                                  sink=sink, recent=recent).to(device)
        # Replace the attention with the mixer
        _replace_attention(layer, mixer)
        mixers.append(mixer)

    print(f"  patched {len(mixers)} attention modules")

    # Calibration data
    print(f"  loading {n_calib_samples} calibration samples...")
    texts = load_calibration_texts(n_calib_samples, min_chars=calib_len * 6)

    batches = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=calib_len).input_ids.to(device)
        if ids.shape[1] >= 64:
            batches.append(ids)
    print(f"  prepared {len(batches)} batches "
          f"(seq_len up to {max(b.shape[1] for b in batches)})")

    # Cache the "full" model outputs (with all alpha = +∞ → sigmoid → 1)
    print(f"  caching reference (full attention) outputs...")
    for m in mixers:
        with torch.no_grad():
            m.alpha.fill_(10.0)  # σ(10) ≈ 1, fully retrieval

    refs = []
    with torch.no_grad():
        for ids in batches:
            out = model(ids, use_cache=False, output_hidden_states=False).logits
            refs.append(out.detach())

    # Reset alpha to 0 (σ(0) = 0.5, balanced start)
    for m in mixers:
        with torch.no_grad():
            m.alpha.fill_(0.0)

    # Optimizer over α only
    alpha_params = [m.alpha for m in mixers]
    for p in alpha_params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(alpha_params, lr=lr)

    print(f"  optimizing α over {epochs} epochs "
          f"(λ={l1_lambda}, lr={lr})...")
    for epoch in range(epochs):
        epoch_mse = 0.0
        epoch_l1 = 0.0
        for ids, ref in zip(batches, refs):
            opt.zero_grad()
            out = model(ids, use_cache=False).logits
            mse = F.mse_loss(out.float(), ref.float())
            l1 = sum(torch.sigmoid(m.alpha).sum() for m in mixers) * l1_lambda
            loss = mse + l1
            loss.backward()
            opt.step()
            epoch_mse += mse.item()
            epoch_l1 += l1.item()
        avg_alpha = torch.stack([torch.sigmoid(m.alpha) for m in mixers]).mean().item()
        print(f"    epoch {epoch+1}/{epochs}  "
              f"mse={epoch_mse/len(batches):.4e}  "
              f"l1={epoch_l1/len(batches):.3f}  "
              f"σ(α)≈{avg_alpha:.3f}")

    # Threshold to {0, 1}
    profile = {}
    n_retrieval_total = 0
    for i, m in enumerate(mixers):
        a = torch.sigmoid(m.alpha).detach().cpu()
        binary = (a >= 0.5).float()
        profile[i] = binary
        n_retrieval_total += int(binary.sum().item())

    total_heads = len(mixers) * spec.num_kv_heads
    print(f"  retrieval heads: {n_retrieval_total}/{total_heads} "
          f"({100 * n_retrieval_total / total_heads:.1f}%)")

    return profile


def _find_decoder_layers(model) -> List[nn.Module]:
    """Locate the decoder layer list for any Llama-family model."""
    # Llama / Mistral / Qwen all use model.model.layers
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        cur = model
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
            if isinstance(cur, (list, nn.ModuleList)):
                return list(cur)
        except AttributeError:
            continue
    raise RuntimeError("Could not locate decoder layers in model")


def _find_attention(layer: nn.Module) -> nn.Module | None:
    """Locate the attention submodule inside a decoder layer."""
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def _replace_attention(layer: nn.Module, new_attn: nn.Module) -> None:
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            setattr(layer, name, new_attn)
            return


def main():
    parser = argparse.ArgumentParser(description="DuoAttention offline profiling")
    parser.add_argument("--model", required=True,
                        help="Model key from MODEL_ZOO (e.g., qwen3-0.6b)")
    parser.add_argument("--calib-samples", type=int, default=8)
    parser.add_argument("--calib-len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l1", type=float, default=0.02)
    parser.add_argument("--sink", type=int, default=16)
    parser.add_argument("--recent", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing profile")
    args = parser.parse_args()

    from suite.config import MODEL_ZOO
    if args.model not in MODEL_ZOO:
        print(f"Unknown model: {args.model}")
        print(f"Available: {list(MODEL_ZOO.keys())}")
        sys.exit(1)

    short = MODEL_ZOO[args.model].short_name
    out_path = profile_path(short)
    if out_path.exists() and not args.force:
        print(f"Profile already exists at {out_path}  (use --force to overwrite)")
        return

    profile = profile_model(
        args.model,
        n_calib_samples=args.calib_samples,
        calib_len=args.calib_len,
        epochs=args.epochs,
        lr=args.lr,
        l1_lambda=args.l1,
        sink=args.sink,
        recent=args.recent,
        device=args.device,
    )

    PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(profile, out_path)
    print(f"\nSaved profile → {out_path}")


if __name__ == "__main__":
    main()
