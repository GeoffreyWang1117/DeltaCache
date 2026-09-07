"""Why the experiment cleanup blocks rebind instead of calling exec("del ...").

The suite drops its large tensors in a `finally` block before `clear_gpu()`,
which begins with `gc.collect()`. The collector can only reclaim what nothing
references, so the drop has to actually happen. It used to be written as
`exec(f"del {name}")` in a loop, which looks like it frees five tensors per
sample and frees none.

These tests pin the semantics rather than the code, so the pattern cannot be
reintroduced as a "simplification".
"""

from __future__ import annotations

import contextlib
import gc
import weakref


class _Payload:
    """Stands in for a cached tensor: something we want collected on demand."""


def test_exec_del_does_not_unbind_a_function_local() -> None:
    """The original approach. exec sees a copy of locals, so the del is lost."""
    observed = {}

    def run() -> None:
        full_k = _Payload()
        observed["ref"] = weakref.ref(full_k)
        for name in ["full_k"]:
            # The defect under test, reproduced verbatim.
            with contextlib.suppress(Exception):
                exec(f"del {name}")
        # Still bound, and still referenced, so a collection here frees nothing.
        gc.collect()
        observed["alive_inside"] = observed["ref"]() is not None
        observed["still_named"] = "full_k" in locals()

    run()
    assert observed["alive_inside"] is True
    assert observed["still_named"] is True


def test_rebinding_to_none_drops_the_reference() -> None:
    """The fix. Assignment is a real store, so the payload becomes collectable."""
    observed = {}

    def run() -> None:
        full_k = _Payload()
        observed["ref"] = weakref.ref(full_k)
        full_k = None
        gc.collect()
        observed["alive_inside"] = observed["ref"]() is not None

    run()
    assert observed["alive_inside"] is False


def test_rebinding_is_safe_when_the_name_was_never_assigned() -> None:
    """`del` would raise here. The cleanup runs in `finally`, so it must not.

    An exception thrown before the tensors are built leaves those names unbound,
    and a cleanup path that raises would mask the original error.
    """

    def run() -> bool:
        try:
            raise RuntimeError("failed before the cache was built")
        except RuntimeError:
            pass
        finally:
            full_k = full_v = attns = layers = cache = None  # noqa: F841
        return True

    assert run() is True


def test_del_on_an_unassigned_name_would_raise() -> None:
    """Shows what the rebinding avoids, so the choice is not mistaken for style."""

    def run() -> None:
        try:
            raise RuntimeError("failed early")
        except RuntimeError:
            pass
        finally:
            del full_k  # noqa: F821 - unbound on purpose

    try:
        run()
    except (UnboundLocalError, NameError):
        return
    raise AssertionError("expected del of an unassigned local to raise")
