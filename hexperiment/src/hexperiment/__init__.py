"""Hexperiment: a leakage-proof ML kit.

If your model looks too good to be true, it's haunted.
"""

from hexperiment.auditor import audit
from hexperiment.data import load, split
from hexperiment.errors import HexInputError
from hexperiment.models import Finding, Verdict
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
