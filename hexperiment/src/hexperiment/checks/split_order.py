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
_READER_NAMES = frozenset(
    {
        "read_csv",
        "read_parquet",
        "read_excel",
        "read_json",
        "read_feather",
        "read_table",
        "read_fwf",
        "read_pickle",
        "read_sql",
        "read_orc",
        "loadtxt",
        "genfromtxt",
    }
)
READER_MESSAGE = (
    "this reads data without hexperiment.load(); the canary test cannot scramble "
    "this copy"
)
READER_FIX = (
    "load the data with hexperiment.load() and derive everything from that frame"
)
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

    A call to a data reader (``read_csv`` and the others in Requirement 8.4)
    anywhere in the script is a ``warning`` at its line: the canary test cannot
    scramble a copy of the data that ``hexperiment.load`` did not return. These
    warnings do not depend on the split call.
    """
    if source is None:
        return _inconclusive(script_name, "the script text could not be read")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return _inconclusive(script_name, "the script could not be parsed")
    modules = _imported_modules(tree)
    targets = _import_targets(tree)
    calls = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    readers = tuple(
        _reader_finding(script_name, name, node.lineno)
        for node in calls
        if (name := _reader_name(node, targets)) is not None
    )
    first_split = next((node for node in calls if _is_split_call(node, targets)), None)
    if first_split is None:
        return _inconclusive(
            script_name, "the script has no call to a split function", readers
        )
    nested = _argument_calls(first_split)
    hints = tuple(
        _finding(script_name, name, node.lineno, first_split.lineno)
        for node in calls
        if (node.lineno < first_split.lineno or id(node) in nested)
        and (name := _suspect_method(node, modules)) is not None
    )
    return CheckResult(CHECK, None, Outcome.PASSED, _by_line(hints + readers))


def _inconclusive(
    script_name: str, reason: str, readers: tuple[Finding, ...] = ()
) -> CheckResult:
    finding = Finding(
        check=CHECK,
        ghost=None,
        severity=Severity.WARNING,
        location=script_name,
        message=f"the split-order hint could not be applied: {reason}",
        evidence={},
        fix=UNAPPLIED_FIX,
    )
    return CheckResult(CHECK, None, Outcome.INCONCLUSIVE, (finding, *readers))


def _by_line(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
    # sorted is stable, so findings on one line keep their order.
    return tuple(sorted(findings, key=lambda f: int(f.location.rpartition(":")[2])))


def _reader_finding(script_name: str, call: str, line: int) -> Finding:
    return Finding(
        check=CHECK,
        ghost=None,
        severity=Severity.WARNING,
        location=f"{script_name}:{line}",
        message=READER_MESSAGE,
        evidence={"call": call},
        fix=READER_FIX,
    )


def _reader_name(node: ast.Call, targets: dict[str, str]) -> str | None:
    """Return the name of the data reader that the call invokes, else ``None``.

    Only the callee name counts (``pd.read_csv`` and ``read_csv`` both are
    ``read_csv``). A bare name bound by ``from pandas import read_csv as rc`` is
    resolved to the name it stands for (Requirement 8.4).
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = targets.get(func.id, func.id).rpartition(".")[2]
    else:
        return None
    return name if name in _READER_NAMES else None


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


def _is_split_call(node: ast.Call, targets: dict[str, str]) -> bool:
    """Whether the call is ``split``, ``hexperiment.split`` or ``train_test_split``.

    A bare ``split`` or ``train_test_split`` name and any ``.train_test_split``
    attribute count as before. An import alias is resolved first, so
    ``hx.split`` (``import hexperiment as hx``), ``hsplit`` (``from hexperiment
    import split as hsplit``) and ``tts`` (``from sklearn.model_selection import
    train_test_split as tts``) count too (Requirement 8.1).
    """
    func = node.func
    if isinstance(func, ast.Name) and func.id in _SPLIT_NAMES:
        return True
    if isinstance(func, ast.Attribute) and func.attr == "train_test_split":
        return True
    dotted = _dotted_name(func)
    if dotted is None:
        return False
    root, _, rest = dotted.partition(".")
    if root in targets:
        dotted = targets[root] + (f".{rest}" if rest else "")
    return dotted == _QUALIFIED_SPLIT or dotted.rpartition(".")[2] == "train_test_split"


def _argument_calls(call: ast.Call) -> frozenset[int]:
    """Identities of the calls nested in the arguments of ``call``.

    They run before ``call`` does, so they count as earlier calls even when they
    are on the same line or below it (Requirement 8.2).
    """
    arguments = [*call.args, *(keyword.value for keyword in call.keywords)]
    return frozenset(
        id(node)
        for argument in arguments
        for node in ast.walk(argument)
        if isinstance(node, ast.Call)
    )


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


def _import_targets(tree: ast.AST) -> dict[str, str]:
    """Map each name an import binds to the dotted name it stands for.

    ``import hexperiment as hx`` gives ``hx -> hexperiment``; ``from hexperiment
    import split as hsplit`` gives ``hsplit -> hexperiment.split``.
    """
    targets: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    targets[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    targets[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                targets[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return targets


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
