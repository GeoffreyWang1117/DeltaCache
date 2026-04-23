"""Experiment configuration: models, tasks, hardware profiles, cost estimation.

All experiment parameters defined here so nothing is hardcoded in task modules.
Edit MODEL_ZOO / EXPERIMENT_MATRIX / HARDWARE_PROFILES to customize runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Model Definitions ───────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Everything needed to load and profile a model."""
    hf_name: str
    short_name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    arch: str  # "MHA" or "GQA"
    max_context: int  # model's native max context
    vram_fp16_gb: float  # approximate FP16 weight size
    load_in: int = 4  # quantization bits for weights (4 or 8)
    device_map: str = "auto"  # "auto" for multi-GPU, "cuda:0" for single

    @property
    def kv_bytes_per_token(self) -> int:
        """FP16 KV memory per token across all layers."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2

    @property
    def kv_mb_at(self) -> callable:
        """Return a function that computes KV cache MB for a given seq_len."""
        bpt = self.kv_bytes_per_token
        return lambda seq_len: bpt * seq_len / 1024 / 1024


MODEL_ZOO = {
    # --- Small tier: Qwen3 series (2025 SOTA, standard GQA full-attention) ---
    # NOTE: Qwen3.5 and Gemma 4 have hybrid linear/sliding attention which
    # breaks standard KV cache compression assumptions — not used here.
    "qwen3-0.6b": ModelSpec(
        hf_name="Qwen/Qwen3-0.6B",
        short_name="Qwen3-0.6B", num_layers=28, num_kv_heads=8,
        head_dim=128, arch="GQA", max_context=40960, vram_fp16_gb=1.2,
    ),
    "qwen3-1.7b": ModelSpec(
        hf_name="Qwen/Qwen3-1.7B",
        short_name="Qwen3-1.7B", num_layers=28, num_kv_heads=8,
        head_dim=128, arch="GQA", max_context=40960, vram_fp16_gb=3.4,
    ),
    # --- Legacy small (kept for back-compat with prior validation runs) ---
    "qwen2-0.5b": ModelSpec(
        hf_name="Qwen/Qwen2-0.5B",
        short_name="Qwen2-0.5B", num_layers=24, num_kv_heads=2,
        head_dim=64, arch="GQA", max_context=32768, vram_fp16_gb=1.0,
    ),
    "tinyllama": ModelSpec(
        hf_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        short_name="TinyLlama-1.1B", num_layers=22, num_kv_heads=4,
        head_dim=64, arch="GQA", max_context=2048, vram_fp16_gb=2.2,
    ),
    "llama2-7b": ModelSpec(
        hf_name="meta-llama/Llama-2-7b-chat-hf",
        short_name="Llama-2-7B", num_layers=32, num_kv_heads=32,
        head_dim=128, arch="MHA", max_context=4096, vram_fp16_gb=13.5,
    ),
    "mistral-7b": ModelSpec(
        hf_name="mistralai/Mistral-7B-Instruct-v0.2",
        short_name="Mistral-7B", num_layers=32, num_kv_heads=8,
        head_dim=128, arch="GQA", max_context=32768, vram_fp16_gb=14.5,
    ),
    "llama3.1-8b": ModelSpec(
        hf_name="meta-llama/Llama-3.1-8B-Instruct",
        short_name="Llama-3.1-8B", num_layers=32, num_kv_heads=8,
        head_dim=128, arch="GQA", max_context=131072, vram_fp16_gb=16.0,
    ),
    "qwen3-8b": ModelSpec(
        hf_name="Qwen/Qwen3-8B",
        short_name="Qwen3-8B", num_layers=36, num_kv_heads=8,
        head_dim=128, arch="GQA", max_context=40960, vram_fp16_gb=17.0,
    ),
    # --- New: for NeurIPS completeness ---
    "llama2-13b": ModelSpec(
        hf_name="meta-llama/Llama-2-13b-chat-hf",
        short_name="Llama-2-13B", num_layers=40, num_kv_heads=40,
        head_dim=128, arch="MHA", max_context=4096, vram_fp16_gb=26.0,
        load_in=4, device_map="auto",  # needs 2 GPUs or large VRAM
    ),
    "llama3-70b": ModelSpec(
        hf_name="meta-llama/Llama-3.1-70B-Instruct",
        short_name="Llama-3.1-70B", num_layers=80, num_kv_heads=8,
        head_dim=128, arch="GQA", max_context=131072, vram_fp16_gb=140.0,
        load_in=4, device_map="auto",
    ),
    "qwen2.5-14b": ModelSpec(
        hf_name="Qwen/Qwen2.5-14B-Instruct",
        short_name="Qwen2.5-14B", num_layers=48, num_kv_heads=4,
        head_dim=128, arch="GQA", max_context=131072, vram_fp16_gb=28.0,
        load_in=4, device_map="auto",
    ),
}


# ── Hardware Profiles ───────────────────────────────────────────────

@dataclass
class HardwareProfile:
    name: str
    gpu_name: str
    gpu_count: int
    vram_per_gpu_gb: int
    cost_per_hour_usd: float  # rental cost

    @property
    def total_vram_gb(self) -> int:
        return self.gpu_count * self.vram_per_gpu_gb

    def can_fit_model(self, spec: ModelSpec) -> bool:
        """Conservative check: weight + ~30% overhead for KV/activations."""
        weight_gb = spec.vram_fp16_gb / (16 / spec.load_in)
        return weight_gb * 1.3 < self.total_vram_gb


HARDWARE_PROFILES = {
    # Local workstation with two RTX 3090s, only ~9 GB free per GPU
    # (rest occupied by other processes).  Use this for phased local testing.
    "local_3090x2_9gb": HardwareProfile(
        "local_3090x2_9gb", "RTX 3090", 2, 9, 0.0,
    ),
    "local_3090x2": HardwareProfile(
        "local_3090x2", "RTX 3090", 2, 24, 0.0,
    ),
    "a100_40g": HardwareProfile(
        "a100_40g", "A100 40GB", 1, 40, 1.10,
    ),
    "a100_80g": HardwareProfile(
        "a100_80g", "A100 80GB", 1, 80, 1.60,
    ),
    "a100_80gx2": HardwareProfile(
        "a100_80gx2", "A100 80GB", 2, 80, 3.20,
    ),
    "h100_80g": HardwareProfile(
        "h100_80g", "H100 80GB", 1, 80, 2.50,
    ),
    "h100_80gx2": HardwareProfile(
        "h100_80gx2", "H100 80GB", 2, 80, 5.00,
    ),
}


# ── Method Configurations ───────────────────────────────────────────

# Methods to compare in main PPL table (paper Table 1)
# Core methods for main PPL table — 8 methods covering all categories.
# Full 18-method comparison available from Llama-2-7B run + local Qwen3 runs.
MAIN_TABLE_METHODS = [
    "full_kv",
    "layer_budget",       # ours (joint)
    "h2o_uniform",        # top eviction (NeurIPS 2023)
    "kivi_uniform",       # top quantization (ICML 2024)
    "snapkv",             # top attention-based eviction
    "streaming_llm",      # lower bound (ICLR 2024)
    "duo_attention",      # per-head policy (ICLR 2025)
    "minikv",             # joint competitor
]

# Extended set for appendix (run on selected models with more budget)
EXTENDED_METHODS = MAIN_TABLE_METHODS + [
    "pyramidkv", "d2o", "squeeze_attention", "cake",
    "dynamickv", "adakv", "lava", "evolkv", "kvtuner", "xquant",
]

# Lighter set for expensive tasks (MMLU, LongBench, GSM8K)
DOWNSTREAM_METHODS = [
    "full_kv",
    "layer_budget",
    "h2o_uniform",
    "streaming_llm",
    "duo_attention",
    "kivi_uniform",
    "snapkv",
    "minikv",
]

# Even lighter set for generation-heavy tasks (MATH, GSM8K, NIAH, RULER).
# Autoregressive decoding is ~100× costlier per token than a forward pass,
# so we only compare the most important methods here.
GENERATION_METHODS = [
    "full_kv",
    "layer_budget",
    "kivi_uniform",
    "streaming_llm",
    "h2o_uniform",
]

# For generation tasks, only the most paper-relevant CRs
GENERATION_CRS = [2.0, 4.0]

COMPRESSION_RATIOS = [2.0, 3.0, 4.0, 6.0]

# ── Model Size Tiers (for phased execution) ──────────────────────────
#
# Phase 1 (local, free): validate all task/baseline code, get preliminary numbers
# Phase 2 (local 7B):    short-context 7B on local hardware (fits with 4-bit @ ≤1024tok)
# Phase 3 (rented A100): 7B long-context + 13B all-context
# Phase 4 (rented H100): 70B (only for final full-tier run)
#
MODEL_SIZE_TIERS = {
    "small":  ["qwen3-0.6b", "qwen3-1.7b"],
    "7b":     ["llama2-7b", "mistral-7b", "llama3.1-8b", "qwen3-8b"],
    "13b":    ["llama2-13b", "qwen2.5-14b"],
    "70b":    ["llama3-70b"],
}

# Seq-lens per phase (local phases stay short to avoid OOM on shared GPU)
SEQ_LENS_BY_PHASE = {
    "small":  [512, 1024, 2048, 4096],  # small models support all lengths locally
    "7b":     [512, 1024],              # local; extend to 4096/8192 on rented GPU
    "7b_long":[512, 1024, 2048, 4096, 8192],  # rented server only
    "13b":    [512, 1024, 2048, 4096],
    "70b":    [512, 1024, 2048],
}


# ── Experiment Matrix ───────────────────────────────────────────────

@dataclass
class ExperimentUnit:
    """Single atomic experiment: one (model, task, method, config)."""
    model_key: str
    task_name: str
    method_name: str
    compression_ratio: float
    seq_len: int = 0  # task-specific; 0 = use task default
    extra: Dict = field(default_factory=dict)
    priority: int = 0  # lower = run first

    @property
    def checkpoint_key(self) -> str:
        """Unique key for checkpoint/resume."""
        parts = [self.model_key, self.task_name, self.method_name,
                 f"cr{self.compression_ratio}", f"seq{self.seq_len}"]
        if self.extra:
            parts.extend(f"{k}={v}" for k, v in sorted(self.extra.items()))
        return "__".join(parts)


def build_experiment_matrix(
    tier: str = "full",
    hardware: str = "a100_80g",
    model_size: Optional[str] = None,
) -> List[ExperimentUnit]:
    """Build the full experiment matrix.

    Args:
        tier: Scope of experiments.
          "quick"    — smoke test (1 model, 1 CR, 1 seq_len per task)
          "paper"    — NeurIPS minimum
          "full"     — all models, all CRs, all seq_lens
        hardware: Key into HARDWARE_PROFILES. Used for VRAM feasibility check
          and cost estimation.  Use "local_3090x2_9gb" for phased local runs.
        model_size: Optional filter by size tier for phased execution:
          "small"    — Qwen2-0.5B, TinyLlama (local, validates all code)
          "7b"       — Llama-2-7B, Mistral-7B, Llama-3.1-8B
          "13b"      — Llama-2-13B, Qwen2.5-14B
          "70b"      — Llama-3.1-70B
          None       — use all models allowed by tier + hardware
    """
    hw = HARDWARE_PROFILES[hardware]
    units = []

    # When model_size is specified, it fully determines the model list
    # and seq_len range — tier only controls CRs and baseline method set.
    if model_size is not None:
        models = list(MODEL_SIZE_TIERS.get(model_size, []))
        # For 7B models: use longer seqs when running on a server GPU (≥40 GB VRAM)
        # so the same --model-size flag works for both local validation and full runs.
        if model_size == "small":
            ppl_seqs = SEQ_LENS_BY_PHASE["small"]
            downstream_seqs = [s for s in SEQ_LENS_BY_PHASE["small"] if s >= 1024]
        elif model_size == "7b":
            if hw.total_vram_gb >= 40:
                # Cap at 4096 to avoid 8192tok OOM cascading that fragments
                # GPU memory and kills subsequent shorter-context runs.
                # 8192tok results can be obtained in a dedicated run.
                ppl_seqs = [512, 1024, 2048, 4096]
                downstream_seqs = [1024, 2048, 4096]
            else:
                ppl_seqs = SEQ_LENS_BY_PHASE["7b"]
                downstream_seqs = SEQ_LENS_BY_PHASE["7b"]
        elif model_size == "13b":
            ppl_seqs = SEQ_LENS_BY_PHASE["13b"]
            downstream_seqs = SEQ_LENS_BY_PHASE["13b"]
        else:  # 70b
            ppl_seqs = SEQ_LENS_BY_PHASE["70b"]
            downstream_seqs = SEQ_LENS_BY_PHASE["70b"]

        if tier == "quick":
            models = models[:1]
            crs = [4.0]
            ppl_seqs = ppl_seqs[:1]
            downstream_seqs = ppl_seqs[:1]
        else:
            crs = COMPRESSION_RATIOS

    elif tier == "quick":
        models = ["mistral-7b"]
        crs = [4.0]
        ppl_seqs = [512]
        downstream_seqs = [512]
    elif tier == "paper":
        models = ["llama2-7b", "mistral-7b", "llama3.1-8b", "llama2-13b"]
        crs = COMPRESSION_RATIOS
        ppl_seqs = [512, 1024, 2048, 4096]
        downstream_seqs = [1024, 2048, 4096]
    else:  # full
        models = ["llama2-7b", "mistral-7b", "llama3.1-8b", "llama2-13b",
                  "qwen2.5-14b", "llama3-70b"]
        crs = COMPRESSION_RATIOS
        ppl_seqs = [512, 1024, 2048, 4096, 8192]
        downstream_seqs = [1024, 2048, 4096, 8192]

    # Filter to models that fit on hardware (VRAM check)
    models = [m for m in models if hw.can_fit_model(MODEL_ZOO[m])]

    # P0: PPL evaluation (main table) — full baseline set
    for model in models:
        spec = MODEL_ZOO[model]
        for seq in ppl_seqs:
            if seq > spec.max_context:
                continue
            for method in MAIN_TABLE_METHODS:
                if method == "full_kv":
                    # full_kv produces identical results at every CR;
                    # only create one unit (CR=2.0) and skip duplicates.
                    units.append(ExperimentUnit(
                        model, "ppl", method, 2.0, seq, priority=0,
                    ))
                else:
                    for cr in crs:
                        units.append(ExperimentUnit(
                            model, "ppl", method, cr, seq, priority=0,
                        ))

    # P1: Downstream tasks
    # MMLU: forward-pass only (cheap) → DOWNSTREAM_METHODS × all CRs
    # LongBench, GSM8K, MATH: generation-heavy → GENERATION_METHODS × GENERATION_CRS
    for model in models:
        spec = MODEL_ZOO[model]
        # MMLU: cheap, full method set
        for method in DOWNSTREAM_METHODS:
            method_crs = [2.0] if method == "full_kv" else crs
            for cr in method_crs:
                units.append(ExperimentUnit(
                    model, "mmlu", method, cr, priority=1,
                ))

        # LongBench / GSM8K / MATH: generation, lean set
        for method in GENERATION_METHODS:
            method_crs = [2.0] if method == "full_kv" else GENERATION_CRS
            for cr in method_crs:
                for seq in downstream_seqs:
                    if seq > spec.max_context:
                        continue
                    units.append(ExperimentUnit(
                        model, "longbench", method, cr, seq, priority=1,
                    ))
                units.append(ExperimentUnit(
                    model, "gsm8k", method, cr, priority=1,
                ))
                units.append(ExperimentUnit(
                    model, "math", method, cr, priority=1,
                ))

    # P2: RULER + NIAH (synthetic, generation-heavy)
    for model in models:
        spec = MODEL_ZOO[model]
        for method in GENERATION_METHODS:
            method_crs = [2.0] if method == "full_kv" else GENERATION_CRS
            for cr in method_crs:
                for seq in [2048, 4096, 8192]:
                    if seq > spec.max_context:
                        continue
                    units.append(ExperimentUnit(
                        model, "ruler", method, cr, seq, priority=2,
                    ))
                    units.append(ExperimentUnit(
                        model, "niah", method, cr, seq, priority=2,
                    ))

    # P3: Throughput (system benchmark)
    for model in models:
        spec = MODEL_ZOO[model]
        for seq in ppl_seqs:
            if seq > spec.max_context:
                continue
            for cr in crs:
                units.append(ExperimentUnit(
                    model, "throughput", "layer_budget", cr, seq, priority=3,
                ))

    # Sort: priority → model (minimize loads) → task → seq_len → full_kv first
    def _sort_key(u):
        # full_kv = 0 so it runs first (populates PPL cache)
        method_order = 0 if u.method_name == "full_kv" else 1
        return (u.priority, u.model_key, u.task_name, u.seq_len, method_order)
    units.sort(key=_sort_key)
    return units


# ── Cost Estimation ─────────────────────────────────────────────────

# Empirical seconds-per-unit estimates (from RTX 3090 runs, scaled)
TASK_TIME_ESTIMATES = {
    # (task, model_size_bucket) -> seconds per (method, cr) unit
    "ppl": {"small": 5, "7b": 15, "13b": 30, "70b": 120},
    "mmlu": {"small": 30, "7b": 120, "13b": 240, "70b": 600},
    "longbench": {"small": 20, "7b": 60, "13b": 120, "70b": 300},
    "gsm8k": {"small": 20, "7b": 90, "13b": 180, "70b": 450},
    "math": {"small": 30, "7b": 150, "13b": 300, "70b": 750},
    "ruler": {"small": 10, "7b": 30, "13b": 60, "70b": 180},
    "niah": {"small": 5, "7b": 20, "13b": 40, "70b": 120},
    "throughput": {"small": 15, "7b": 45, "13b": 90, "70b": 300},
}

# Model loading time (seconds)
MODEL_LOAD_ESTIMATES = {
    "small": 10, "7b": 30, "13b": 60, "70b": 180,
}


def _model_bucket(model_key: str) -> str:
    spec = MODEL_ZOO[model_key]
    if spec.vram_fp16_gb < 5:
        return "small"
    elif spec.vram_fp16_gb < 20:
        return "7b"
    elif spec.vram_fp16_gb < 50:
        return "13b"
    else:
        return "70b"


def estimate_cost(
    units: List[ExperimentUnit],
    hardware: str = "a100_80g",
    gpu_speedup_vs_3090: float = 2.0,
) -> Dict:
    """Estimate total GPU hours and cost for the experiment matrix.

    Args:
        units: Experiment units to estimate.
        hardware: Hardware profile key.
        gpu_speedup_vs_3090: Speed multiplier vs RTX 3090 baseline.

    Returns:
        Dict with per-task and total estimates.
    """
    hw = HARDWARE_PROFILES[hardware]

    # Group by model to count model loads
    models_needed = set(u.model_key for u in units)
    load_time = sum(
        MODEL_LOAD_ESTIMATES[_model_bucket(m)] for m in models_needed
    )

    per_task = {}
    total_seconds = load_time
    for u in units:
        bucket = _model_bucket(u.model_key)
        est = TASK_TIME_ESTIMATES.get(u.task_name, {}).get(bucket, 30)
        est /= gpu_speedup_vs_3090
        total_seconds += est
        per_task.setdefault(u.task_name, 0)
        per_task[u.task_name] += est

    total_hours = total_seconds / 3600
    total_cost = total_hours * hw.cost_per_hour_usd

    return {
        "hardware": hw.name,
        "gpu": hw.gpu_name,
        "cost_per_hour": hw.cost_per_hour_usd,
        "n_units": len(units),
        "n_models": len(models_needed),
        "model_load_seconds": load_time,
        "per_task_seconds": per_task,
        "total_seconds": total_seconds,
        "total_hours": round(total_hours, 2),
        "estimated_cost_usd": round(total_cost, 2),
    }
