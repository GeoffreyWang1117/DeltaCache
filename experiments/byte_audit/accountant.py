"""Generic measurement of how many bytes a KV cache object actually occupies.

Nothing here knows what a baseline is. Given any Python object, it finds every
torch tensor reachable from it and reports the bytes those tensors really hold,
counted once per underlying storage so that views and slices are not
double-counted.

Three numbers matter downstream and they are deliberately kept apart:

    reported    what a method says its footprint is (its own accounting)
    structural  what its returned tensors actually occupy
    allocator   what the CUDA caching allocator saw during the call

A method whose ``reported`` tracks a compression knob while its ``structural``
does not is not lying; it is reporting an analytic figure for storage it never
built. Telling those apart is the whole point of the audit.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch import Tensor

# Containers we recurse into. Anything else is inspected via __dict__/__slots__.
_MAX_DEPTH = 12


@dataclass
class Footprint:
    """Bytes held by the tensors reachable from an object.

    Two totals, because they answer different questions. ``total_bytes`` counts
    whole storages and is what the allocator cannot hand to anyone else.
    ``logical_bytes`` counts what the tensors address. They diverge when a small
    tensor is a view into a large buffer, which is how a cache can look evicted
    while still pinning everything it evicted.
    """

    total_bytes: int = 0
    logical_bytes: int = 0
    n_tensors: int = 0
    n_storages: int = 0
    by_dtype: Dict[str, int] = field(default_factory=dict)

    def add(self, dtype: str, nbytes: int) -> None:
        self.total_bytes += nbytes
        self.n_storages += 1
        self.by_dtype[dtype] = self.by_dtype.get(dtype, 0) + nbytes

    @property
    def view_overhead(self) -> float:
        """Storage held per byte addressed. Above 1.0 means views pin extra."""
        if self.logical_bytes <= 0:
            return 0.0
        return self.total_bytes / self.logical_bytes

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "logical_bytes": self.logical_bytes,
            "view_overhead": round(self.view_overhead, 4),
            "n_tensors": self.n_tensors,
            "n_storages": self.n_storages,
            "by_dtype": dict(sorted(self.by_dtype.items())),
        }


def inner_tensor_names(t: Tensor) -> Optional[List[str]]:
    """Attribute names of the tensors a subclass really stores, or None.

    A quantized tensor subclass advertises the dtype and element count of the
    tensor it stands in for, and its storage reports that same size, so counting
    it directly charges the uncompressed footprint. The packed bytes live in
    inner tensors reached through the standard subclass protocol.
    """
    flatten = getattr(t, "__tensor_flatten__", None)
    if flatten is None:
        return None
    try:
        names, _ctx = flatten()
    except Exception:  # a subclass free to refuse; fall back to counting it whole
        return None
    return list(names) or None


def iter_tensors(obj: Any, max_depth: int = _MAX_DEPTH) -> Iterator[Tensor]:
    """Yield every torch tensor reachable from ``obj``, each object visited once.

    Tensor subclasses are opened rather than counted, so what is charged is the
    storage a subclass actually holds and not the shape it presents.
    """
    seen: Set[int] = set()
    # Identity alone is not a safe key: attribute access on a tensor subclass can
    # hand back a temporary, and once it is collected a later temporary can land
    # on the same address and be skipped as already seen. Holding a reference to
    # everything visited keeps ids unique for the length of the walk.
    keepalive: List[Any] = []
    stack: List[Tuple[Any, int]] = [(obj, 0)]
    while stack:
        node, depth = stack.pop()
        if node is None or depth > max_depth:
            continue
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        keepalive.append(node)

        if isinstance(node, Tensor):
            inner = inner_tensor_names(node)
            if inner:
                stack.extend((getattr(node, name, None), depth + 1) for name in inner)
            else:
                yield node
            continue
        if isinstance(node, (str, bytes, int, float, bool)):
            continue
        if isinstance(node, dict):
            stack.extend((v, depth + 1) for v in node.values())
            stack.extend((k, depth + 1) for k in node)
            continue
        if isinstance(node, (list, tuple, set, frozenset)):
            stack.extend((v, depth + 1) for v in node)
            continue

        # Plain objects: walk their attributes, including __slots__ classes.
        attrs = getattr(node, "__dict__", None)
        if isinstance(attrs, dict):
            stack.extend((v, depth + 1) for v in attrs.values())
        for slot in getattr(type(node), "__slots__", ()) or ():
            if isinstance(slot, str):
                stack.append((getattr(node, slot, None), depth + 1))


def measure_structural(obj: Any) -> Footprint:
    """Bytes held by tensors reachable from ``obj``, one count per storage.

    Two tensors that are views of the same buffer are charged once. This is the
    number a serving system would actually pay.
    """
    fp = Footprint()
    counted: Set[Tuple[int, int]] = set()
    for t in iter_tensors(obj):
        fp.n_tensors += 1
        fp.logical_bytes += t.numel() * t.element_size()
        try:
            storage = t.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
        except (RuntimeError, AttributeError):  # meta / sparse / fake tensors
            key = (id(t), t.numel() * t.element_size())
        if key[0] == 0 or key in counted:
            continue
        counted.add(key)
        fp.add(str(t.dtype).replace("torch.", ""), key[1])
    return fp


@contextlib.contextmanager
def allocator_probe(device: Optional[torch.device] = None) -> Iterator[Dict[str, int]]:
    """Record the peak CUDA allocation across a block.

    Yields a dict filled in on exit. On CPU the peak is unavailable and the
    fields stay at -1 rather than reporting a number the platform cannot give.
    """
    out: Dict[str, int] = {"peak_bytes": -1, "delta_bytes": -1}
    use_cuda = device is not None and device.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
    try:
        yield out
    finally:
        if use_cuda:
            torch.cuda.synchronize(device)
            out["peak_bytes"] = int(torch.cuda.max_memory_allocated(device))
            out["delta_bytes"] = int(torch.cuda.memory_allocated(device) - before)


_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})


def float_byte_fraction(fp: Footprint) -> float:
    """Share of the measured bytes that sit in a floating-point dtype.

    A method claiming integer quantization whose storage is almost entirely
    floating point has simulated the arithmetic and kept the original layout.
    The fraction rather than a flag, because index and scale tensors are
    legitimately integer and would otherwise mask the answer -- they are also
    small, so a genuinely packed cache stays far below 1.0.
    """
    if fp.total_bytes <= 0:
        return 0.0
    float_bytes = sum(n for name, n in fp.by_dtype.items() if name in _FLOAT_DTYPES)
    return float_bytes / fp.total_bytes


def is_simulated(fp: Footprint, threshold: float = 0.99) -> bool:
    """True when the returned storage is float-typed and therefore uncompressed."""
    return float_byte_fraction(fp) >= threshold
