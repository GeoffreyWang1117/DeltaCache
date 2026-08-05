# Archived Experiment Scripts

These scripts are archived because they use simulated data, are incomplete, or have been superseded by newer versions.

## Mock / Simulated (no real model inference)
- `cache_experiment_mock.py` — Basic cache behavior with synthetic data
- `memory_pressure_simulated.py` — Simulated memory pressure (no real GPU)
- `large_model_pressure_simulated.py` — Simulated large model scenarios
- `layer_ablation_simulated.py` — Layer ablation with synthetic importance scores
- `tail_latency_simulated.py` — Tail latency simulation
- `advanced_policies_simulated.py` — Policy comparison with synthetic workloads
- `ijcnn_memory_pressure_simulated.py` — IJCNN memory pressure (simulated)
- `ijcnn_layer_sensitivity_simulated.py` — IJCNN layer sensitivity (simulated)
- `ijcnn_eviction_mock.py` — IJCNN eviction experiment (mock data)
- `tiered_cache_synthetic.py` — Tiered cache with synthetic transfers

## Superseded
- `vllm_comparison_v1.py` — First vLLM comparison attempt (broken)
- `vllm_comparison_v2.py` — Fixed vLLM comparison (superseded by `bench_vllm_comparison.py`)
- `sglang_comparison_incomplete.py` — Incomplete SGLang integration
- `baseline_comparison_utility.py` — Utility script for baseline comparisons
- `visualize_results_old.py` — Old visualization script

## Active scripts are in `experiments/bench_*.py`
