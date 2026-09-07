"""Tests for the KV byte audit instrument.

The instrument's job is to tell three things apart: a method that shrinks the
cache, a method that only reports shrinking it, and a call that did nothing at
all. Each of those is tested directly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from experiments.byte_audit.accountant import (  # noqa: E402
    Footprint,
    float_byte_fraction,
    is_simulated,
    measure_structural,
)
from experiments.byte_audit.analyze import classify_knob, summarize  # noqa: E402
from experiments.byte_audit.gate import GEOMETRIES, Shape, make_inputs  # noqa: E402


def test_views_are_charged_once() -> None:
    base = torch.zeros(4, 8, dtype=torch.float16)
    view = base[:2]
    fp = measure_structural({"k": base, "v": view})
    assert fp.n_tensors == 2
    assert fp.n_storages == 1
    assert fp.total_bytes == 4 * 8 * 2


def test_separate_buffers_are_both_charged() -> None:
    fp = measure_structural([torch.zeros(4, dtype=torch.float16), torch.zeros(4, dtype=torch.int8)])
    assert fp.total_bytes == 8 + 4
    assert fp.by_dtype == {"float16": 8, "int8": 4}


def test_tensors_are_found_through_nested_containers_and_objects() -> None:
    class Holder:
        def __init__(self) -> None:
            self.payload = {"layers": [(torch.zeros(6, dtype=torch.float32),)]}

    assert measure_structural(Holder()).total_bytes == 24


def test_dequantized_cache_reads_as_simulated_at_realistic_size() -> None:
    """A method that returns FP16 plus an index vector has packed nothing.

    The index tensor is integer, so a flag over dtype names would call this
    packed. At any realistic cache size the indices are negligible and the
    fraction gives the right answer.
    """
    seq, heads, dim = 512, 8, 64
    cache = [
        torch.zeros(seq, heads, dim, dtype=torch.float16),
        torch.zeros(seq, heads, dim, dtype=torch.float16),
        torch.arange(seq),
    ]
    fp = measure_structural(cache)
    assert is_simulated(fp)
    assert float_byte_fraction(fp) > 0.99


def test_genuinely_packed_cache_of_the_same_shape_is_not_simulated() -> None:
    seq, heads, dim = 512, 8, 64
    cache = [
        torch.zeros(seq, heads, dim, dtype=torch.int8),
        torch.zeros(seq, heads, dim, dtype=torch.int8),
        torch.zeros(seq, heads, dtype=torch.float16),  # scales
    ]
    assert not is_simulated(measure_structural(cache))


def test_packed_storage_is_not_simulated() -> None:
    fp = measure_structural([torch.zeros(1024, dtype=torch.int8), torch.zeros(8)])
    assert is_simulated(fp) is False


def test_empty_footprint_has_zero_fraction() -> None:
    assert float_byte_fraction(Footprint()) == 0.0


@pytest.mark.parametrize(
    ("distinct", "points", "expected"),
    [(1, 6, "FLAT"), (3, 6, "DISCRETE"), (6, 6, "CONTINUOUS"), (0, 6, "FLAT")],
)
def test_knob_classification(distinct: int, points: int, expected: str) -> None:
    assert classify_knob(distinct, points) == expected


def _cell(method: str, cr: float, **kw) -> dict:
    base = {
        "method": method,
        "category": "eviction",
        "compression_ratio": cr,
        "ok": True,
        "did_something": True,
        "reported_bytes": 100,
        "structural_bytes": 100,
    }
    base.update(kw)
    return base


def test_inert_method_is_unverified_not_flat() -> None:
    """A method that never acted must not be reported as having a flat knob."""
    cells = [_cell("noop", cr, did_something=False) for cr in (2.0, 4.0, 8.0)]
    row = summarize(cells)[0]
    assert row["status"] == "UNVERIFIED"
    assert "3 no-op" in row["note"]


def test_errored_cells_do_not_reach_the_verdict() -> None:
    cells = [_cell("broken", cr, ok=False, error="boom") for cr in (2.0, 4.0)]
    assert summarize(cells)[0]["status"] == "UNVERIFIED"


def test_reported_and_delivered_are_classified_independently() -> None:
    """The sharpest cell shape: a smoothly varying claim over a fixed footprint."""
    cells = [
        _cell("claims_a_knob", cr, reported_bytes=int(1000 / cr), structural_bytes=1000)
        for cr in (2.0, 4.0, 8.0)
    ]
    row = summarize(cells)[0]
    assert row["delivered_knob"] == "FLAT"
    assert row["reported_knob"] == "CONTINUOUS"
    assert row["reported_over_delivered"] < 0.5


def test_requested_ratios_snap_onto_the_backend_native_widths() -> None:
    """The axis claim in one assertion: most requested ratios are unrepresentable.

    A quantized cache exposes bit widths, not a ratio. Against an FP16 baseline
    only 4x and 8x land exactly on quanto's two widths; everything else snaps,
    and four of the six requested points collapse onto the same setting.
    """
    from experiments.byte_audit.adapters.hf_quantized_cache import nbits_for_cr

    grid = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
    chosen = [nbits_for_cr(cr, "quanto") for cr in grid]
    exact = [cr for cr, (_, snapped) in zip(grid, chosen) if not snapped]
    assert exact == [4.0, 8.0]
    assert len({nbits for nbits, _ in chosen}) == 2
    assert sum(1 for nbits, _ in chosen if nbits == 4) == 4


def test_hqq_represents_more_points_than_quanto() -> None:
    """More native widths means fewer snaps, so the effect is backend-specific."""
    from experiments.byte_audit.adapters.hf_quantized_cache import nbits_for_cr

    grid = (2.0, 4.0, 8.0, 16.0)
    quanto_exact = sum(1 for cr in grid if not nbits_for_cr(cr, "quanto")[1])
    hqq_exact = sum(1 for cr in grid if not nbits_for_cr(cr, "hqq")[1])
    assert hqq_exact > quanto_exact


def test_unknown_backend_is_rejected() -> None:
    from experiments.byte_audit.adapters.hf_quantized_cache import nbits_for_cr

    with pytest.raises(KeyError):
        nbits_for_cr(4.0, "not-a-backend")


def test_gqa_geometry_shrinks_the_cache_not_the_attention() -> None:
    """Key/value heads size the cache; query heads size the attention."""
    shape = Shape(num_layers=2, num_heads=8, num_kv_heads=2, head_dim=16, seq_len=32)
    assert shape.is_gqa
    data = make_inputs(shape, torch.device("cpu"))
    assert data["keys"].shape == (2, 32, 2, 16)
    assert data["attention_weights"][0].shape == (1, 8, 32, 32)


def test_named_geometries_are_gqa_except_the_older_shapes() -> None:
    assert GEOMETRIES["qwen2.5-7b"]["num_kv_heads"] < GEOMETRIES["qwen2.5-7b"]["num_heads"]
    assert GEOMETRIES["llama-2-7b"]["num_kv_heads"] == GEOMETRIES["llama-2-7b"]["num_heads"]


def test_attention_is_not_built_until_indexed() -> None:
    """A sweep reaches lengths where materialising every layer would not fit."""
    shape = Shape(num_layers=4, num_heads=4, head_dim=8, seq_len=64)
    lazy = make_inputs(shape, torch.device("cpu"))["attention_weights"]
    assert len(lazy) == 4
    assert lazy._cache == {}
    lazy[2]
    assert list(lazy._cache) == [2]


def test_oversized_attention_refuses_rather_than_shrinking() -> None:
    shape = Shape(num_layers=1, num_heads=64, head_dim=8, seq_len=1 << 17)
    lazy = make_inputs(shape, torch.device("cpu"))["attention_weights"]
    with pytest.raises(MemoryError, match="skipped rather than silently shrunk"):
        lazy[0]


def _quant_cell(method: str, cr: float, nominal: int, delivered: int, full: int) -> dict:
    return {
        "method": method,
        "category": "quantization",
        "compression_ratio": cr,
        "ok": True,
        "did_something": True,
        "reported_bytes": None,
        "structural_bytes": delivered,
        "fullkv_bytes": full,
        "nominal_bits": nominal,
    }


def test_effective_bits_exposes_a_width_that_was_never_delivered() -> None:
    """Eight bits of storage per value, whatever width was requested."""
    cells = [_quant_cell("sim", cr, n, 512, 1024) for cr, n in ((4.0, 4), (8.0, 2))]
    row = summarize(cells)[0]
    assert row["nominal_bits"] == [2, 4]
    assert row["effective_bits"] == [8.0]
    assert row["worst_bit_gap"] == 6.0


def test_honest_quantizer_shows_only_metadata_overhead() -> None:
    cells = [_quant_cell("real", 4.0, 4, 320, 1024)]
    row = summarize(cells)[0]
    assert row["effective_bits"] == [5.0]
    assert row["worst_bit_gap"] == 1.0


def test_eviction_rows_carry_no_bit_width() -> None:
    """Only a method that states a width can be held to one."""
    row = summarize([_cell("evict", cr) for cr in (2.0, 4.0)])[0]
    assert row["nominal_bits"] is None
    assert row["worst_bit_gap"] is None


def test_lazy_attention_terminates_iteration() -> None:
    """The legacy iteration protocol stops only on IndexError.

    Without a bounds check, `for a in attention_weights` builds a fresh tensor
    for every index without end. That is not hypothetical: it hung a full sweep.
    """
    shape = Shape(num_layers=3, num_heads=2, head_dim=8, seq_len=16)
    lazy = make_inputs(shape, torch.device("cpu"))["attention_weights"]
    assert len(list(lazy)) == 3
    with pytest.raises(IndexError):
        lazy[3]


def test_lazy_attention_supports_negative_indices() -> None:
    shape = Shape(num_layers=3, num_heads=2, head_dim=8, seq_len=16)
    lazy = make_inputs(shape, torch.device("cpu"))["attention_weights"]
    assert torch.equal(lazy[-1], lazy[2])
