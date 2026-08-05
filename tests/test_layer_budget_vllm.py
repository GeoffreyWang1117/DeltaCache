"""Tests for LayerBudgetBlockManager (vLLM block-level compression).

Tests use synthetic data — no vLLM or GPU required.
"""

import pytest
import torch

from deltacache.vllm_integration.layer_budget_block_manager import (
    LayerBudgetBlockManager,
    LayerBlockAllocation,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def manager():
    """Standard manager: 4 layers, 4 KV heads, 64 head_dim, block_size=16."""
    return LayerBudgetBlockManager(
        num_layers=4, num_kv_heads=4, head_dim=64,
        block_size=16, sink_blocks=1, recent_blocks=1,
    )


@pytest.fixture
def large_manager():
    """7B-like manager: 32 layers, 8 KV heads, 128 head_dim."""
    return LayerBudgetBlockManager(
        num_layers=32, num_kv_heads=8, head_dim=128,
        block_size=16, sink_blocks=1, recent_blocks=2,
    )


def make_attention_weights(num_layers, num_heads, seq_len, sparse=False):
    """Create synthetic attention weights."""
    attn = []
    for l in range(num_layers):
        w = torch.rand(1, num_heads, seq_len, seq_len)
        if sparse:
            # Make later layers more sparse (concentrate on sink + recent)
            mask = torch.zeros(seq_len)
            mask[:4] = 1.0  # sink
            mask[-8:] = 1.0  # recent
            if l > num_layers // 2:
                w = w * mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # Normalize
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        attn.append(w)
    return attn


def make_vllm_cache(num_layers, num_blocks, block_size, num_kv_heads, head_dim):
    """Create synthetic vLLM-format GPU cache."""
    cache = []
    for _ in range(num_layers):
        t = torch.randn(2, num_blocks, block_size, num_kv_heads, head_dim)
        cache.append(t)
    return cache


# ── Tests ─────────────────────────────────────────────────────────────────

class TestBlockManagerInit:
    def test_basic_init(self, manager):
        assert manager.num_layers == 4
        assert manager.block_size == 16
        assert manager.sink_blocks == 1
        assert manager.recent_blocks == 1

    def test_block_memory_cost(self, manager):
        # FP16: 2 * 16 * 4 * 64 * 2 = 16384 bytes per block per layer
        cost = manager.block_memory_cost(16)
        assert cost == 2 * 16 * 4 * 64 * 16 // 8
        # INT4: 2 * 16 * 4 * 64 * 4 / 8 = 4096
        cost4 = manager.block_memory_cost(4)
        assert cost4 == cost // 4


class TestProfileAndAllocate:
    def test_uniform_allocation_no_attention(self, manager):
        """Without attention weights, uses uniform sparsity."""
        seq_len = 256  # 16 blocks
        full_mem = manager.full_memory(16)
        budget = full_mem // 2  # 50% budget

        alloc = manager.profile_and_allocate(None, seq_len, budget)

        assert isinstance(alloc, LayerBlockAllocation)
        assert len(alloc.allocations) == 4  # all layers allocated
        assert alloc.total_memory_bytes <= budget

    def test_with_attention_weights(self, manager):
        seq_len = 128  # 8 blocks
        attn = make_attention_weights(4, 4, seq_len, sparse=True)
        full_mem = manager.full_memory(8)
        budget = full_mem // 3

        alloc = manager.profile_and_allocate(attn, seq_len, budget)

        assert alloc.total_memory_bytes <= budget
        for layer_idx in range(4):
            retained, bits = alloc.allocations[layer_idx]
            assert len(retained) > 0  # at least some blocks retained

    def test_full_budget_retains_all(self, manager):
        """With generous budget, all blocks should be retained."""
        seq_len = 128  # 8 blocks
        full_mem = manager.full_memory(8)
        budget = full_mem * 2  # way more than needed

        alloc = manager.profile_and_allocate(None, seq_len, budget)

        for layer_idx in range(4):
            retained, bits = alloc.allocations[layer_idx]
            assert len(retained) == 8  # all blocks
            assert bits == 16  # full precision

    def test_tiny_budget(self, manager):
        """Very small budget should still retain sink + recent blocks."""
        seq_len = 256  # 16 blocks
        budget = manager.block_memory_cost(4) * 2 * 4  # minimal

        alloc = manager.profile_and_allocate(None, seq_len, budget)

        for layer_idx in range(4):
            retained, bits = alloc.allocations[layer_idx]
            # Should retain at least sink (block 0) and recent (block 15)
            assert 0 in retained, f"Layer {layer_idx}: sink block 0 not retained"
            assert 15 in retained, f"Layer {layer_idx}: recent block 15 not retained"

    def test_budget_respected(self, large_manager):
        """Budget constraint must be strictly respected."""
        seq_len = 1024  # 64 blocks
        full_mem = large_manager.full_memory(64)

        for ratio in [0.1, 0.25, 0.5, 0.75]:
            budget = int(full_mem * ratio)
            alloc = large_manager.profile_and_allocate(None, seq_len, budget)
            assert alloc.total_memory_bytes <= budget, (
                f"ratio={ratio}: used {alloc.total_memory_bytes} > budget {budget}"
            )


class TestBlockSelection:
    def test_sink_always_retained(self, manager):
        seq_len = 256  # 16 blocks
        budget = manager.block_memory_cost(4) * 4 * 4  # very tight

        alloc = manager.profile_and_allocate(None, seq_len, budget)

        for layer_idx in range(4):
            retained, _ = alloc.allocations[layer_idx]
            assert 0 in retained, f"Layer {layer_idx}: block 0 (sink) evicted"

    def test_recent_always_retained(self, manager):
        seq_len = 256  # 16 blocks
        budget = manager.block_memory_cost(4) * 4 * 4

        alloc = manager.profile_and_allocate(None, seq_len, budget)

        for layer_idx in range(4):
            retained, _ = alloc.allocations[layer_idx]
            assert 15 in retained, f"Layer {layer_idx}: block 15 (recent) evicted"

    def test_freed_blocks_are_complement(self, manager):
        """Freed blocks should be the complement of retained blocks."""
        seq_len = 128  # 8 blocks
        full_mem = manager.full_memory(8)
        budget = full_mem // 2

        alloc = manager.profile_and_allocate(None, seq_len, budget)

        for layer_idx in range(4):
            retained, _ = alloc.allocations[layer_idx]
            freed = alloc.freed_block_indices[layer_idx]
            all_blocks = set(range(8))
            assert set(retained) | set(freed) == all_blocks
            assert set(retained) & set(freed) == set()


class TestBlockImportance:
    def test_importance_shape(self, manager):
        attn = torch.rand(1, 4, 128, 128)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        scores = manager.compute_block_importance(0, attn, 16, 8)

        assert scores.shape == (8,)
        assert (scores >= 0).all()

    def test_importance_sums_to_attention_mass(self, manager):
        """Block importance should sum to ~total attention mass in last row."""
        attn = torch.rand(1, 4, 128, 128)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        scores = manager.compute_block_importance(0, attn, 16, 8)

        # Last-token attention row sums to ~1.0 (normalized)
        last_row = attn[0, :, -1, :].mean(dim=0)
        expected_total = last_row.sum().item()
        actual_total = scores.sum().item()
        assert abs(actual_total - expected_total) < 0.01


class TestApplyCompression:
    def test_evicted_blocks_are_zeroed(self, manager):
        num_blocks = 8
        gpu_cache = make_vllm_cache(4, num_blocks, 16, 4, 64)
        seq_len = num_blocks * 16

        full_mem = manager.full_memory(num_blocks)
        alloc = manager.profile_and_allocate(None, seq_len, full_mem // 3)

        manager.apply_compression(gpu_cache, alloc)

        for layer_idx in range(4):
            retained, bits = alloc.allocations[layer_idx]
            retained_set = set(retained)
            for b in range(num_blocks):
                if b not in retained_set:
                    assert (gpu_cache[layer_idx][:, b] == 0).all(), (
                        f"Layer {layer_idx} block {b}: not zeroed after eviction"
                    )

    def test_retained_blocks_nonzero(self, manager):
        num_blocks = 8
        gpu_cache = make_vllm_cache(4, num_blocks, 16, 4, 64)
        seq_len = num_blocks * 16

        full_mem = manager.full_memory(num_blocks)
        alloc = manager.profile_and_allocate(None, seq_len, full_mem // 2)

        manager.apply_compression(gpu_cache, alloc)

        for layer_idx in range(4):
            retained, bits = alloc.allocations[layer_idx]
            for b in retained:
                if b < num_blocks:
                    assert not (gpu_cache[layer_idx][:, b] == 0).all(), (
                        f"Layer {layer_idx} block {b}: should be retained but is zero"
                    )

    def test_qdq_changes_values(self, manager):
        """INT4/INT8 QDQ should modify values (quantization error)."""
        num_blocks = 4
        gpu_cache = make_vllm_cache(4, num_blocks, 16, 4, 64)
        originals = [c.clone() for c in gpu_cache]

        # Force all retained at INT4
        alloc = LayerBlockAllocation(
            allocations={l: (list(range(num_blocks)), 4) for l in range(4)},
            total_memory_bytes=0,
            budget_bytes=0,
            freed_block_indices={l: [] for l in range(4)},
        )

        manager.apply_compression(gpu_cache, alloc)

        for layer_idx in range(4):
            # Values should be different due to QDQ
            diff = (gpu_cache[layer_idx] - originals[layer_idx]).abs().mean()
            assert diff > 0, f"Layer {layer_idx}: QDQ had no effect"

    def test_fp16_no_qdq(self, manager):
        """FP16 blocks should not be modified."""
        num_blocks = 4
        gpu_cache = make_vllm_cache(4, num_blocks, 16, 4, 64)
        originals = [c.clone() for c in gpu_cache]

        alloc = LayerBlockAllocation(
            allocations={l: (list(range(num_blocks)), 16) for l in range(4)},
            total_memory_bytes=0,
            budget_bytes=0,
            freed_block_indices={l: [] for l in range(4)},
        )

        manager.apply_compression(gpu_cache, alloc)

        for layer_idx in range(4):
            assert torch.equal(gpu_cache[layer_idx], originals[layer_idx])
