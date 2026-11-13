"""Hexperiment: an ML kit that catches data leakage.

If your model looks too good to be true, it's haunted.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from hexperiment.errors import HexInputError
from hexperiment.models import Finding, Verdict

if TYPE_CHECKING:
    from hexperiment.auditor import audit
    from hexperiment.data import load, split
    from hexperiment.runs import load_history

__all__ = [
    "audit",
    "load",
    "split",
    "load_history",
    "HexInputError",
    "Verdict",
    "Finding",
]


def _home_module(name: str) -> str | None:
    """Module that defines a lazily resolved public function, or ``None``."""
    if name == "audit":
        return "hexperiment.auditor"
    if name in ("load", "split"):
        return "hexperiment.data"
    if name == "load_history":
        return "hexperiment.runs"
    return None


def __getattr__(name: str) -> Any:
    # The functions that pull in pandas and scikit-learn are resolved on first
    # access, so that importing the package stays cheap (Requirement 4.8).
    module_name = _home_module(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
