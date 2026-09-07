"""Tests for LayerAttentionProfiler."""

import torch

from deltacache.core.layer_profiler import (
    LayerAttentionProfiler,
    LayerProfile,
    ProfileResult,
    compute_entropy,
    compute_gini,
)


class TestGiniCoefficient:
    """Tests for Gini coefficient computation."""

    def test_uniform_distribution(self):
        """Uniform distribution should have Gini ≈ 0."""
        weights = torch.ones(100) / 100
        gini = compute_gini(weights)
        assert gini < 0.05, f"Uniform Gini should be ~0, got {gini}"

    def test_concentrated_distribution(self):
        """Concentrated distribution should have high Gini."""
        weights = torch.zeros(100)
        weights[0] = 1.0
        gini = compute_gini(weights)
        assert gini > 0.9, f"Concentrated Gini should be ~1, got {gini}"

    def test_moderate_sparsity(self):
        """Half-and-half should have moderate Gini."""
        weights = torch.zeros(100)
        weights[:10] = 0.1  # 10% of tokens get all mass
        gini = compute_gini(weights)
        assert 0.3 < gini < 0.95, f"Moderate Gini out of range: {gini}"

    def test_single_element(self):
        """Single-element should return 0."""
        gini = compute_gini(torch.tensor([1.0]))
        assert gini == 0.0

    def test_empty_weights(self):
        """Zero weights should return 0."""
        gini = compute_gini(torch.zeros(10))
        assert gini == 0.0

    def test_range(self):
        """Gini should always be in [0, 1]."""
        for _ in range(10):
            weights = torch.softmax(torch.randn(50), dim=0)
            gini = compute_gini(weights)
            assert 0.0 <= gini <= 1.0, f"Gini out of range: {gini}"


class TestEntropy:
    """Tests for normalized entropy computation."""

    def test_uniform_distribution(self):
        """Uniform should have entropy ≈ 1."""
        weights = torch.ones(100) / 100
        h = compute_entropy(weights)
        assert h > 0.95, f"Uniform entropy should be ~1, got {h}"

    def test_concentrated_distribution(self):
        """Delta should have entropy ≈ 0."""
        weights = torch.zeros(100)
        weights[0] = 1.0
        h = compute_entropy(weights)
        assert h < 0.1, f"Concentrated entropy should be ~0, got {h}"

    def test_single_element(self):
        h = compute_entropy(torch.tensor([1.0]))
        assert h == 0.0


class TestProfileResult:
    """Tests for ProfileResult data structure."""

    def test_gini_scores(self):
        profiles = [
            LayerProfile(
                0, gini=0.3, entropy=0.7, top10_mass=0.5, top20_mass=0.7, max_attention=0.1
            ),
            LayerProfile(
                1, gini=0.8, entropy=0.3, top10_mass=0.9, top20_mass=0.95, max_attention=0.4
            ),
        ]
        result = ProfileResult(profiles, num_layers=2, seq_len=64, profiling_time_ms=5.0)

        scores = result.gini_scores()
        assert scores == {0: 0.3, 1: 0.8}

    def test_sparsity_ranking(self):
        profiles = [
            LayerProfile(
                0, gini=0.3, entropy=0.7, top10_mass=0.5, top20_mass=0.7, max_attention=0.1
            ),
            LayerProfile(
                1, gini=0.8, entropy=0.3, top10_mass=0.9, top20_mass=0.95, max_attention=0.4
            ),
            LayerProfile(
                2, gini=0.5, entropy=0.5, top10_mass=0.7, top20_mass=0.85, max_attention=0.2
            ),
        ]
        result = ProfileResult(profiles, num_layers=3, seq_len=64, profiling_time_ms=5.0)

        ranking = result.sparsity_ranking()
        assert ranking == [1, 2, 0]  # Most sparse first


class TestLayerAttentionProfiler:
    """Tests for the profiler using synthetic attention weights."""

    def test_profile_from_attention_weights(self):
        """Test profiling from pre-extracted attention tensors."""
        profiler = LayerAttentionProfiler()

        # Create synthetic attention weights for 4 layers
        # Shape: (batch=1, heads=4, seq=32, seq=32)
        attn_weights = []
        for layer in range(4):
            # Create attention pattern: later layers more concentrated
            attn = torch.randn(1, 4, 32, 32)
            if layer >= 2:
                # Make later layers more peaked
                attn = attn * 3
            attn = torch.softmax(attn, dim=-1)
            attn_weights.append(attn)

        result = profiler.profile_from_attention_weights(attn_weights)

        assert result.num_layers == 4
        assert result.seq_len == 32
        assert len(result.layer_profiles) == 4

        # All profiles should have valid metrics
        for p in result.layer_profiles:
            assert 0.0 <= p.gini <= 1.0
            assert 0.0 <= p.entropy <= 1.0
            assert 0.0 <= p.top10_mass <= 1.0
            assert 0.0 <= p.max_attention <= 1.0

    def test_profile_sparse_vs_uniform(self):
        """Sparse attention should have higher Gini than uniform."""
        profiler = LayerAttentionProfiler()

        # Layer 0: near-uniform attention
        uniform_attn = torch.ones(1, 4, 32, 32) / 32
        # Ensure valid softmax (small perturbation)
        uniform_attn = torch.softmax(
            torch.zeros(1, 4, 32, 32) + 0.01 * torch.randn(1, 4, 32, 32),
            dim=-1,
        )

        # Layer 1: very sparse attention (all mass on first token)
        sparse_logits = torch.full((1, 4, 32, 32), -100.0)
        sparse_logits[:, :, :, 0] = 10.0  # Strong attention to position 0
        sparse_attn = torch.softmax(sparse_logits, dim=-1)

        result = profiler.profile_from_attention_weights([uniform_attn, sparse_attn])

        assert result.layer_profiles[1].gini > result.layer_profiles[0].gini
        assert result.layer_profiles[1].entropy < result.layer_profiles[0].entropy

    def test_find_attention_modules_returns_empty_for_non_transformer(self):
        """Non-transformer model should return empty module list."""
        profiler = LayerAttentionProfiler()

        class DummyModel:
            pass

        modules = profiler._find_attention_modules(DummyModel())
        assert len(modules) == 0

    def test_hooks_cleanup(self):
        """Hooks should be properly cleaned up."""
        profiler = LayerAttentionProfiler()
        assert len(profiler._hooks) == 0

        # Register some dummy hooks
        profiler._hooks.append("fake_hook")
        profiler._hooks.clear()
        assert len(profiler._hooks) == 0
