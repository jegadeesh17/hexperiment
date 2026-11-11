"""Structural rules of the package, checked on its source with ``ast``.

The tests walk the package directory, so a module added later is checked without
changing this file (Requirements 2.1, 15.2).
"""

import ast
from pathlib import Path

import pytest

import hexperiment

PACKAGE_DIR = Path(hexperiment.__file__).resolve().parent

PUBLIC_API = {
    "audit",
    "load",
    "split",
    "load_history",
    "HexInputError",
    "Verdict",
    "Finding",
}

# Modules that must stay free of file, process, clock and config access.
PURE_FILES = {"verdict.py", "card.py", "models.py", "errors.py", "hook_event.py"}
FORBIDDEN_IN_PURE = {"os", "pathlib", "shutil", "subprocess", "tomllib", "datetime"}

MUTABLE_CALLS = {
    "list",
    "dict",
    "set",
    "bytearray",
    "deque",
    "defaultdict",
    "OrderedDict",
    "Counter",
}
MUTABLE_NODES = (
    ast.List,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
)


def package_modules() -> list[Path]:
    return sorted(PACKAGE_DIR.rglob("*.py"))


def module_id(path: Path) -> str:
    return path.relative_to(PACKAGE_DIR).as_posix()


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_roots(tree: ast.AST) -> set[str]:
    """Top-level module names imported anywhere in ``tree``, nested imports too."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def calls_open(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "open":
                return True
    return False


def is_pure_module(path: Path) -> bool:
    relative = path.relative_to(PACKAGE_DIR)
    return relative.parts[0] == "checks" or relative.name in PURE_FILES


def mutable_module_variables(tree: ast.Module) -> list[str]:
    """Names bound at module level to a list, dict, set or similar mutable value."""
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.AugAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if not is_mutable_value(value):
            continue
        found.extend(
            target.id
            for target in targets
            if isinstance(target, ast.Name) and not is_dunder(target.id)
        )
    return found


def is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def is_mutable_value(value: ast.expr) -> bool:
    if isinstance(value, MUTABLE_NODES):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        return name in MUTABLE_CALLS
    return False


def test_the_walk_finds_the_modules_of_the_package():
    found = {module_id(path) for path in package_modules()}
    assert {
        "__init__.py",
        "auditor.py",
        "canary.py",
        "card.py",
        "checks/__init__.py",
        "checks/metric.py",
        "config.py",
        "console.py",
        "data.py",
        "errors.py",
        "models.py",
        "runs.py",
        "verdict.py",
    } <= found


def test_all_is_exactly_the_public_api():
    assert sorted(hexperiment.__all__) == sorted(PUBLIC_API)
    for name in PUBLIC_API:
        assert hasattr(hexperiment, name)


def test_all_in_the_source_is_a_literal_list_of_the_public_names():
    tree = parse(PACKAGE_DIR / "__init__.py")
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    ]
    assert len(values) == 1
    assert sorted(values[0]) == sorted(PUBLIC_API)


def test_only_canary_imports_subprocess():
    importers = [
        module_id(path)
        for path in package_modules()
        if "subprocess" in imported_roots(parse(path))
    ]
    assert importers == ["canary.py"]


def imported_modules(path: Path) -> set[str]:
    """Every module a file imports, as dotted names, including ``from x import y``."""
    modules: set[str] = set()
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


@pytest.mark.parametrize("name", ["hook.py", "hook_event.py", "workspace.py"])
def test_the_hook_modules_do_not_import_the_cli(name):
    assert "hexperiment.cli" not in imported_modules(PACKAGE_DIR / name)


def test_only_console_calls_print():
    printers = [
        module_id(path)
        for path in package_modules()
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            for node in ast.walk(parse(path))
        )
    ]
    assert printers == ["console.py"]


def test_cli_and_auditor_print_through_the_console_module():
    for name in ("cli.py", "auditor.py"):
        imports = [
            node
            for node in ast.walk(parse(PACKAGE_DIR / name))
            if isinstance(node, ast.ImportFrom) and node.module == "hexperiment.console"
        ]
        assert [alias.name for node in imports for alias in node.names] == [
            "print_text"
        ]


@pytest.mark.parametrize(
    "path", [p for p in package_modules() if is_pure_module(p)], ids=module_id
)
def test_pure_modules_import_no_io_modules_and_never_call_open(path):
    tree = parse(path)
    assert imported_roots(tree) & FORBIDDEN_IN_PURE == set()
    assert not calls_open(tree)


def test_the_pure_modules_include_every_check():
    checked = {module_id(p) for p in package_modules() if is_pure_module(p)}
    assert {
        "checks/__init__.py",
        "checks/metric.py",
        "checks/row_overlap.py",
        "checks/split_order.py",
        "checks/target_proxy.py",
        "checks/temporal.py",
        "verdict.py",
        "card.py",
        "models.py",
        "errors.py",
        "hook_event.py",
    } <= checked


@pytest.mark.parametrize("path", package_modules(), ids=module_id)
def test_no_module_defines_a_module_level_mutable_variable(path):
    assert mutable_module_variables(parse(path)) == []


# The helpers above must catch what they claim to catch.


def test_imported_roots_sees_nested_and_from_imports():
    tree = ast.parse("import os.path\ndef f():\n    from subprocess import run\n")
    assert imported_roots(tree) == {"os", "subprocess"}


def test_calls_open_sees_builtin_and_method_calls():
    assert calls_open(ast.parse("open('x')"))
    assert calls_open(ast.parse("path.open()"))
    assert not calls_open(ast.parse("x = opened(1)"))


@pytest.mark.parametrize(
    "source",
    ["x = []", "x = {}", "x = {1}", "x = list()", "x: dict = {}", "x = [i for i in y]"],
)
def test_mutable_module_variables_sees_mutable_values(source):
    assert mutable_module_variables(ast.parse(source)) == ["x"]


@pytest.mark.parametrize(
    "source",
    ["x = ()", "x = 1", "x = frozenset()", "__all__ = ['a']", "def f():\n    y = []"],
)
def test_mutable_module_variables_accepts_immutable_values(source):
    assert mutable_module_variables(ast.parse(source)) == []
