"""Hexperiment: a leakage-proof ML kit.

If your model looks too good to be true, it's haunted.
"""

from hexperiment.data import load, split
from hexperiment.errors import HexInputError
from hexperiment.models import Finding, Verdict

__all__ = ["Finding", "HexInputError", "Verdict", "load", "split"]
