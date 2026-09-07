"""Locating and loading vendored upstream implementations.

Two problems recur across every tier-B adapter and are solved once here.

**Finding the code.** Upstream projects are shallow-cloned under ``baselines/``.
``MANIFEST`` pins each to the commit it was measured at, because a result that
cannot name the revision it came from is not reproducible.

**Loading the code.** These repositories were written against older releases of
transformers and their modules import names that have since moved, so importing
them fails outright. ``load_from_source`` executes only the definitions actually
needed, without running the surrounding module. What runs is their source,
unedited: patching upstream would make the measurement about the patch.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def repo_root() -> Path:
    """The DeltaCache checkout this file lives in."""
    return Path(__file__).resolve().parents[2]


def baselines_root() -> Path:
    return repo_root() / "baselines"


@dataclass(frozen=True)
class Upstream:
    """One vendored third-party implementation."""

    key: str
    url: str
    subdir: str
    licence: str

    @property
    def path(self) -> Path:
        return baselines_root() / self.subdir

    def commit(self) -> Optional[str]:
        """The revision actually on disk, read at run time rather than trusted.

        A hash written into a manifest by hand drifts; this asks git.
        """
        try:
            out = subprocess.run(
                ["git", "-C", str(self.path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() or None

    def require(self) -> Path:
        if not self.path.exists():
            raise FileNotFoundError(
                f"upstream '{self.key}' is not vendored at {self.path}; "
                f"clone it with: git clone --depth 1 {self.url} {self.path}"
            )
        return self.path


MANIFEST: Dict[str, Upstream] = {
    u.key: u
    for u in (
        Upstream("h2o", "https://github.com/FMInference/H2O", "h2o_official", "MIT"),
        Upstream("kivi", "https://github.com/jy-yuan/KIVI", "kivi_official", "MIT"),
        Upstream("snapkv", "https://github.com/FasterDecoding/SnapKV", "snapkv_official", "MIT"),
        Upstream(
            "pyramidkv",
            "https://github.com/Zefan-Cai/KVCache-Factory",
            "pyramidkv_official",
            "MIT",
        ),
    )
}


def load_from_source(
    path: Path,
    names: Sequence[str],
    namespace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute selected top-level definitions from a source file, unmodified.

    Args:
        path: the upstream file.
        names: class or function names to take, in the order they must execute.
        namespace: globals the extracted code may reference. ``torch`` is
            supplied by default; add anything else the definitions close over.

    Raises if a name is missing rather than returning a partial result, because
    a silently absent method would be indistinguishable from one that measured
    as doing nothing.
    """
    if not path.exists():
        raise FileNotFoundError(f"upstream source not found: {path}")

    import torch

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = list(names)
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted
    ]
    missing = set(wanted) - {n.name for n in nodes}
    if missing:
        raise AttributeError(f"{sorted(missing)} not found in {path}")
    nodes.sort(key=lambda n: wanted.index(n.name))

    # A synthetic module name, registered before execution. Decorators that
    # inspect their class resolve it through sys.modules[cls.__module__], which
    # is how @dataclass reads field annotations; without this it fails on any
    # extracted dataclass.
    mod_name = f"_byte_audit_upstream_{path.stem}_{abs(hash(str(path))) & 0xFFFF:04x}"
    ns: Dict[str, Any] = {"torch": torch, "__name__": mod_name}
    if namespace:
        ns.update(namespace)

    holder = types.ModuleType(mod_name)
    holder.__dict__.update(ns)
    sys.modules[mod_name] = holder
    try:
        module = ast.Module(body=nodes, type_ignores=[])
        exec(compile(module, filename=str(path), mode="exec"), holder.__dict__)
        return {name: holder.__dict__[name] for name in wanted}
    finally:
        sys.modules.pop(mod_name, None)
