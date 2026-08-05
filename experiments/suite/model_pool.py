"""Smart model loading with memory management.

Key optimizations:
- Load model once, run ALL tasks before switching to next model
- Explicit GPU memory cleanup between models
- Auto-detect device map for multi-GPU setups
- Profiler integration: profile on first load, cache to disk
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deltacache.core.layer_profiler import LayerAttentionProfiler
from deltacache.core.layer_budget_allocator import LayerBudgetAllocator

from .config import ModelSpec, MODEL_ZOO
from .checkpoint import CheckpointManager


def clear_gpu():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


class ModelPool:
    """Manages model lifecycle: load, profile, hold, unload.

    Usage:
        pool = ModelPool(ckpt_mgr)
        model, tokenizer = pool.get("mistral-7b")  # loads + profiles
        # ... run experiments ...
        pool.release()  # frees GPU memory
        model, tokenizer = pool.get("llama2-13b")   # loads next
    """

    def __init__(self, ckpt: CheckpointManager):
        self.ckpt = ckpt
        self._current_key: Optional[str] = None
        self._model = None
        self._tokenizer = None
        self._spec: Optional[ModelSpec] = None
        self._gini: Optional[Dict[int, float]] = None
        self._importance: Optional[Dict[int, float]] = None

    @property
    def spec(self) -> ModelSpec:
        assert self._spec is not None, "No model loaded"
        return self._spec

    @property
    def model(self):
        assert self._model is not None, "No model loaded"
        return self._model

    @property
    def tokenizer(self):
        assert self._tokenizer is not None, "No model loaded"
        return self._tokenizer

    @property
    def gini_scores(self) -> Dict[int, float]:
        assert self._gini is not None, "No profile loaded"
        return self._gini

    @property
    def importance_weights(self) -> Dict[int, float]:
        assert self._importance is not None, "No profile loaded"
        return self._importance

    def get(self, model_key: str) -> Tuple[Any, Any]:
        """Get model+tokenizer, loading if needed."""
        if self._current_key == model_key:
            return self._model, self._tokenizer

        # Release previous model
        if self._current_key is not None:
            self.release()

        spec = MODEL_ZOO[model_key]
        self._spec = spec
        self._current_key = model_key

        # Load model
        t0 = time.perf_counter()
        self._model, self._tokenizer = _load_model(spec)
        load_time = time.perf_counter() - t0
        mem = gpu_memory_mb()
        print(f"  Model loaded in {load_time:.1f}s, GPU mem: {mem:.0f} MB")

        # Verify model is actually on GPU (not silently on CPU)
        try:
            device = str(next(self._model.parameters()).device)
            if 'cpu' in device:
                print(f"  *** WARNING: Model loaded on CPU ({device})! ***")
                print(f"  *** This will be extremely slow. ***")
        except StopIteration:
            pass

        # Load or compute Gini profile
        self._load_profile(model_key)

        return self._model, self._tokenizer

    def _load_profile(self, model_key: str):
        """Load cached profile or compute fresh one."""
        cached = self.ckpt.load_profile(model_key)
        if cached is not None:
            self._gini = {int(k): v for k, v in cached["gini"].items()}
            self._importance = {int(k): v for k, v in cached["importance"].items()}
            print(f"  Profile loaded from cache ({len(self._gini)} layers)")
            return

        print("  Profiling attention sparsity...")
        spec = self._spec

        # Profile with 2 calibration samples at 512 tokens (fast, stable)
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = " ".join(x["text"] for x in ds if x["text"].strip())
        tokens = self._tokenizer.encode(text, add_special_tokens=False)

        gini_accum = {}
        n_samples = 2
        profile_len = min(512, len(tokens) // n_samples)

        profiler = LayerAttentionProfiler()
        for i in range(n_samples):
            chunk = tokens[i * profile_len: (i + 1) * profile_len]
            input_ids = torch.tensor([chunk], device=self._model.device)
            result = profiler.profile(input_ids, self._model,
                                      device=str(self._model.device))
            for layer_idx, g in result.gini_scores().items():
                gini_accum.setdefault(layer_idx, []).append(g)
            clear_gpu()

        self._gini = {l: sum(gs) / len(gs) for l, gs in gini_accum.items()}

        # Inverted importance (early layers → higher weight)
        nl = spec.num_layers
        sigmoid = LayerBudgetAllocator.compute_importance_weights(nl)
        self._importance = {l: sigmoid[nl - 1 - l] for l in range(nl)}

        # Cache to disk
        self.ckpt.save_profile(model_key, {
            "model": spec.hf_name,
            "num_layers": nl,
            "gini": self._gini,
            "importance": self._importance,
        })
        print(f"  Profile computed and cached ({nl} layers)")

    def release(self):
        """Unload model and free GPU memory.

        BitsAndBytes 4-bit models hold GPU tensors that survive simple
        `del model`.  We move all parameters to CPU first, then delete,
        then aggressively clear CUDA caches to avoid fragmentation that
        would force the next model onto CPU.
        """
        if self._model is not None:
            # Move all parameters to CPU before deleting
            try:
                self._model.cpu()
            except Exception:
                pass
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._current_key = None
        self._spec = None
        self._gini = None
        self._importance = None
        # Aggressive multi-pass cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        mem = gpu_memory_mb()
        print(f"  Model released, GPU: {mem:.0f}MB")


def _load_model(spec: ModelSpec):
    """Load a model with optimal quantization settings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n{'='*60}")
    print(f"Loading {spec.short_name} ({spec.hf_name})")
    print(f"  Weights: {spec.load_in}-bit, KV heads: {spec.num_kv_heads}, "
          f"Layers: {spec.num_layers}, Arch: {spec.arch}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if spec.load_in == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif spec.load_in == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name,
        quantization_config=quant_config,
        device_map=spec.device_map,
        attn_implementation="eager",  # needed for hook-based profiling
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer
