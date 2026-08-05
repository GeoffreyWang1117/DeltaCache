"""Unified experiment suite for LayerBudget paper.

Cost-optimized for rented GPU servers:
- Profile once per model, cache to disk
- Load model once, run all tasks before switching
- Checkpoint/resume at (model, task, method, config) granularity
- OOM recovery with automatic batch size reduction
- Dry-run cost estimation before execution
"""
