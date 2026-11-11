"""The split-order hint: fits and statistics that come before the split (Requirement 8).

The check reads the text of the training script with ``ast``. It never executes or
imports the script, and it compares line numbers only, not data flow. It reports
warnings and never an error: the canary test is the authority on leaks.
"""

from __future__ import annotations

import ast

from hexperiment.models import CheckResult, Finding, Outcome, Severity

CHECK = "split_order"
FIX = "this runs before the split; make sure it does not use test rows"
UNAPPLIED_FIX = (
    "Call hexperiment.split(df) in train.py and fit preprocessing after it, so the "
    "order of the split and the fits can be checked."
)

_SPLIT_NAMES = frozenset({"split", "train_test_split"})
_QUALIFIED_SPLIT = "hexperiment.split"
_FIT_METHODS = frozenset({"fit", "fit_transform", "fit_resample"})
_STATISTIC_METHODS = frozenset(
    {"mean", "median", "std", "quantile", "min", "max", "transform"}
)


def check_split_order(source: str | None, script_name: str) -> CheckResult:
    """Warn about fits and column statistics on lines before the first split call.

    The first call to ``split``, ``hexperiment.split`` or ``train_test_split``
    (smallest line number) is found first (Requirement 8.1). A call to ``fit``,
    ``fit_transform`` or ``fit_resample`` on any receiver, or to ``mean``,
    ``median``, ``std``, ``quantile``, ``min``, ``max`` or ``transform`` as a
    method of a DataFrame or Series expression, on an earlier line is a ``warning``
    located at ``<script_name>:<line>`` (Requirement 8.2). A bare ``min(...)`` or
    ``max(...)`` and a call rooted at an imported module alias (``np.mean``) are
    ignored. If the source is unavailable or cannot be parsed, or holds no split
    call, the result is one ``warning`` and the outcome ``inconclusive``
    (Requirement 8.3).
    """
    if source is None:
        return _inconclusive(script_name, "the script text could not be read")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return _inconclusive(script_name, "the script could not be parsed")
    calls = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    first_split = next((node for node in calls if _is_split_call(node)), None)
    if first_split is None:
        return _inconclusive(script_name, "the script has no call to a split function")
    modules = _imported_modules(tree)
    findings = tuple(
        _finding(script_name, name, node.lineno, first_split.lineno)
        for node in calls
        if node.lineno < first_split.lineno
        and (name := _suspect_method(node, modules)) is not None
    )
    return CheckResult(CHECK, None, Outcome.PASSED, findings)


def _inconclusive(script_name: str, reason: str) -> CheckResult:
    finding = Finding(
        check=CHECK,
        ghost=None,
        severity=Severity.WARNING,
        location=script_name,
        message=f"the split-order hint could not be applied: {reason}",
        evidence={},
        fix=UNAPPLIED_FIX,
    )
    return CheckResult(CHECK, None, Outcome.INCONCLUSIVE, (finding,))


def _finding(script_name: str, call: str, line: int, first_split_line: int) -> Finding:
    return Finding(
        check=CHECK,
        ghost=None,
        severity=Severity.WARNING,
        location=f"{script_name}:{line}",
        message=f"{call}() is called before the first split",
        evidence={"call": call, "first_split_line": first_split_line},
        fix=FIX,
    )


def _is_split_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _SPLIT_NAMES
    if isinstance(func, ast.Attribute):
        return func.attr == "train_test_split" or _dotted_name(func) == _QUALIFIED_SPLIT
    return False


def _dotted_name(node: ast.expr) -> str | None:
    """Return ``a.b.c`` for a chain of names and attributes, else ``None``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _imported_modules(tree: ast.AST) -> frozenset[str]:
    """Names bound by ``import x``, ``import x.y`` and ``import x as y``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return frozenset(names)


def _suspect_method(node: ast.Call, modules: frozenset[str]) -> str | None:
    """Return the method name when the call is a fit or a column statistic."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr in _FIT_METHODS:
        return func.attr
    if func.attr in _STATISTIC_METHODS and not _rooted_at_module(func.value, modules):
        return func.attr
    return None


def _rooted_at_module(receiver: ast.expr, modules: frozenset[str]) -> bool:
    while isinstance(receiver, ast.Attribute):
        receiver = receiver.value
    return isinstance(receiver, ast.Name) and receiver.id in modules
