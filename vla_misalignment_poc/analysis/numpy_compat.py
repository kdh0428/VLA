"""
Restore the numpy aliases that ORION's vendored code still uses.

ORION pins `numpy==1.23.0`, where `np.bool` / `np.float` / `np.int` / ... were deprecated
but still present. They were REMOVED in numpy 1.24, so the vendored pipeline raises
`AttributeError: module 'numpy' has no attribute 'bool'` (e.g.
`mmcv/datasets/pipelines/transforms_3d.py:632`).

We cannot simply pin numpy 1.23 here: the rest of this environment (scipy, scikit-learn,
numba 0.60, opencv) ships wheels built against a newer numpy ABI, and torch 2.8 — required
for RTX 5090 / sm_120 — expects >= 1.24 as well.

Each alias below is restored to EXACTLY what numpy 1.23 defined it as (the plain Python
builtin, per the numpy 1.20 deprecation note), so behaviour is identical to running under
the pinned version. Importing this module before mmcv is therefore a compatibility shim,
not a semantic change. It is idempotent and never overwrites an attribute that still
exists.
"""
from __future__ import annotations

import numpy as np

# numpy 1.23 value -> attribute name. All of these were plain aliases of builtins
# (or, for `unicode`, of `str`) and carried no numpy-specific behaviour.
_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "long": int,
    "unicode": str,
}

_applied: list[str] = []


def apply() -> list[str]:
    """Install the missing aliases; returns the names that had to be restored."""
    if _applied:
        return _applied
    for name, target in _ALIASES.items():
        if not hasattr(np, name):
            setattr(np, name, target)
            _applied.append(name)
    return _applied


# Applied on import so `import analysis.numpy_compat` is enough.
apply()
