# Security audit — 2026-09-07

Scope: this repository, which the operator owns. The GitHub remote
(`GeoffreyWang1117/DeltaCache`) is **already public**, so history and secrets were
audited as exposed rather than as private.

There is no deployed service, no container, no web endpoint and no authentication in
this project, so the runtime-probe half of the standard procedure does not apply. It
is a research library plus experiment scripts. Everything below is code-path and
supply-chain analysis, and the report says so rather than implying coverage it does
not have.

---

## HIGH — arbitrary code execution from any model repository, by default

**Where:** `deltacache/hf_integration/llama_adapter.py:105`

```python
trust_remote_code: bool = True,      # was the default
```

The value flowed into both `AutoTokenizer.from_pretrained` and the model kwargs. In
transformers, `trust_remote_code=True` means *execute the Python shipped inside the
model repository*. It was the default on the library's public loader, so
`LlamaStyleAdapter.from_pretrained("some/model")` ran that repository's code without
the caller ever deciding to allow it. This is the fail-open pattern: the permissive
state is the one you get by saying nothing.

Repository-wide there were **75 sites setting it True and none setting it False**.

**Compounding it:** zero `revision=` pins anywhere in the codebase, against 152
bandit B615 hits. Even a trusted repository name resolves to whatever its default
branch points at on the day it is fetched, so the name is not a trust anchor and the
result is not reproducible.

**Status: FIXED and re-probed.** The default is now `False`, a `revision` parameter
is threaded through both calls, and the docstring states what turning it on does.
Verified that the models actually in use do not need it: `Qwen/Qwen3-0.6B` loads
with `trust_remote_code=False`. Archived experiment scripts were left alone.

Not exploited. Confirmed by reading the call path, not by loading hostile code.

## MEDIUM — dependency CVEs, and a lockfile that overstated them

`requirements.lock.txt` as first committed was `pip freeze` over the whole conda
environment. It swept in packages this project never declares (borb, pillow,
cryptography, vastai, aiohttp) and pinned the project to its own git URL, which also
made the file unusable as a `pip-audit` input.

| lockfile | packages | packages with CVEs | CVEs |
|---|---|---|---|
| `pip freeze` of the env | 84 | 3 | 27 |
| true dependency closure | 44 | 1 | 5 |

Twenty-one of the twenty-seven were in pillow and cryptography, neither of which is
a dependency of this project. **Status: FIXED.** The lockfile is now generated from
the declared roots and their transitive requirements.

**Remaining: transformers 4.57.6, five CVEs.** Reachability triage:

| CVE | Path | Reachable here |
|---|---|---|
| PYSEC-2025-217 | X-CLIP checkpoint conversion | No. Zero references. |
| PYSEC-2026-2288 | `Trainer` class | No. Zero references. |
| PYSEC-2026-2290 | LightGlue model loading | No. Zero references. |
| PYSEC-2026-2289 | model loading, fix 5.3.0 | Only via a hostile model repository |
| CVE-2026-9856 | path traversal on file write, fix 5.10.0 | Only via a hostile model repository |

Three are not reachable at all. The other two need an attacker-controlled model, so
the exposure was the same unpinned-revision path as the HIGH finding, and pinning
closes it. **Not upgrading transformers**: 4.57.6 is a deliberate pin
(`docs/ENV_REBUILD_DIAGNOSIS.md`) because 5.x breaks the suite's cache API, and the
fixes require 5.3 to 5.10. Pin revisions instead; revisit if a reachable CVE lands.

## LOW — predictable lock path in shared /tmp

`experiments/suite/single_process_guard.py:15` uses a fixed
`/tmp/deltacache_suite.lock` with mode `0o644`. On a multi-tenant box another local
user can create it first and deny the experiment runner its lock. Not exploitable on
this machine: `coder-gw` is the only account with uid >= 1000. It matters because
the module's own docstring says it exists for "rented servers". Left as-is;
noted for anyone running the suite where /tmp is shared.

## Not vulnerabilities, recorded so they are not re-triaged

**627 gitleaks hits in git history: all one false positive.** The `generic-api-key`
rule fired on the JSON field name `_checkpoint_key`; the 607 distinct "secrets" were
experiment identifiers of the form `qwen3-0.6b__ppl__cake__cr4.0__seq1024`. A
`.gitleaks.toml` now allowlists that value shape, targeted at the captured secret
rather than the file, so it cannot mask a real credential. **History is clean: 0.**

**81 gitleaks hits in the working tree, none ours.** Seventy-one are inside the
gitignored vendored clones. Ten are inside `experiments/suite/results/.longbench_cache/`,
which is the LongBench code-completion corpus: test RSA keys and sample JWTs that
ship with the dataset. That path is gitignored and never entered history.

**bandit: 8513 findings, 667 in our code, 0 confirmed vulnerabilities.**
435 are `assert` usage, 152 the unpinned-revision issue above, 45 `random` for
non-cryptographic sampling, and the three "hardcoded password" hits are the parameter
`token_selection="random"` matching on the word *token*.

**semgrep: 7 findings, all `exec-detected`**, matching bandit's B102 exactly.

## The `exec` calls, and why two of them stay

Five are `exec(f'del {vname}')` in experiment cleanup blocks. `vname` iterates a
hardcoded list, so there is no injection. They are, however, a **verified no-op**:
`exec` receives a copy of the function's locals in CPython, so the `del` frees
nothing, and the surrounding code believes it is releasing GPU memory before
`clear_gpu()`. That is a correctness bug, not a security one, and is left for a
separate change.

Two are the byte audit deliberately executing vendored upstream source. There were
three before; `adapters/h2o_official.py` carried a private copy of the loader and
now uses the shared `upstream.py:load_from_source`, so there is one implementation
to reason about. The trust boundary is documented in that module: it executes files
from `baselines/`, which are repositories the operator cloned from URLs in
`MANIFEST`. Nothing fetches code, derives a path from user input, or runs anything
not already on disk, and importing these modules normally would execute strictly
more of each file than this does.

## What was checked and found clean

CI workflow (`.github/workflows/ci.yml`): no secrets usage, no `github.event`
interpolation, no injectable `run` steps. No network-listening code outside one
archived benchmark. No Dockerfile or compose file, so no container or IaC surface.
No `pickle.load`, no `yaml.load`, no `shell=True`, no `os.system` in our code.
Environment-variable name drift: not applicable, no deploy config to drift against.

## Not verified

No runtime probing, because there is nothing deployed. Authenticated flows, headers,
redirects, TLS and proxy configuration are all out of scope for this project rather
than checked and passed. Vendored upstream code under `baselines/` was excluded from
triage: it is third-party, gitignored, and executed deliberately.

## Separately: CI has been failing

`ruff check deltacache/ tests/` is a CI step and reports 107 fixable findings on the
committed tree, almost all import ordering. This predates the audit and is unrelated
to it. The one file this audit touched was brought clean; the rest is untouched so
that a security change does not arrive inside a thousand-line reformat.
