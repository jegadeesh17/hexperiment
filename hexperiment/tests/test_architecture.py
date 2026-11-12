"""Structural rules of the package, checked on its source with ``ast``.

The tests walk the package directory, so a module added later is checked without
changing this file (Requirements 2.1, 15.2).
"""

import ast
import subprocess
import sys
import tomllib
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
PURE_FILES = {
    "verdict.py",
    "card.py",
    "models.py",
    "errors.py",
    "hook_event.py",
    "ghosts.py",
    # The gallery's pure modules (each task adds the modules it creates).
    "newlines.py",
    "markup.py",
    "records.py",
    "summary.py",
    "difftext.py",
    "modelcard.py",
    "pages.py",
    "links.py",
    "site.py",
    # gallery/benchmark.py is deliberately NOT listed: it reads no file and no clock,
    # but it imports ``datetime`` to parse ``created_at`` (hexperiment-benchmark,
    # Requirement 4.1: ISO 8601 UTC), which FORBIDDEN_IN_PURE forbids.
}
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
    return module_names(parse(path))


def module_names(tree: ast.AST) -> set[str]:
    """Dotted names imported by the import statements in ``tree``."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


@pytest.mark.parametrize(
    "name", ["hook.py", "hook_event.py", "workspace.py", "mcp_server/service.py"]
)
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
        "ghosts.py",
    } <= checked


@pytest.mark.parametrize("path", package_modules(), ids=module_id)
def test_no_module_defines_a_module_level_mutable_variable(path):
    assert mutable_module_variables(parse(path)) == []


# The MCP SDK is optional (Requirements 4.1, 7.3).

SDK_ROOTS = {"mcp", "anyio"}
SERVER_MODULE = "mcp_server/server.py"


def test_only_the_server_module_imports_the_sdk():
    importers = [
        module_id(path)
        for path in package_modules()
        if imported_roots(parse(path)) & SDK_ROOTS
    ]
    assert importers == [SERVER_MODULE]


def test_no_module_outside_the_mcp_package_imports_it():
    importers = [
        module_id(path)
        for path in package_modules()
        if not module_id(path).startswith("mcp_server/")
        and any(
            name == "hexperiment.mcp_server"
            or name.startswith("hexperiment.mcp_server.")
            for name in imported_modules(path)
        )
    ]
    assert importers == []


def test_the_mcp_package_marker_has_no_module_level_sdk_or_server_import():
    tree = parse(PACKAGE_DIR / "mcp_server" / "__init__.py")
    top_level = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)],
        type_ignores=[],
    )
    assert imported_roots(top_level) & SDK_ROOTS == set()
    assert not any(
        name.startswith("hexperiment.mcp_server") for name in module_names(top_level)
    )


def test_importing_hexperiment_does_not_load_the_sdk():
    code = (
        "import sys, hexperiment\n"
        "loaded = [m for m in sys.modules if m == 'mcp' or m.startswith('mcp.')]\n"
        "server = 'hexperiment.mcp_server.server' in sys.modules\n"
        "sys.exit(1 if loaded or server else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def project_table() -> dict:
    path = PACKAGE_DIR.parents[1] / "pyproject.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def test_the_runtime_dependencies_stay_the_three_of_the_core():
    assert sorted(project_table()["dependencies"]) == [
        "numpy",
        "pandas",
        "scikit-learn",
    ]


def test_the_sdk_is_pinned_in_the_mcp_extra_and_reached_through_dev():
    extras = project_table()["optional-dependencies"]
    assert extras["mcp"] == ["mcp==2.3.0"]
    assert "hexperiment[mcp]" in extras["dev"]


def test_the_console_script_for_the_server_is_declared():
    scripts = project_table()["scripts"]
    assert scripts["hexperiment-mcp"] == "hexperiment.mcp_server:main"


# The gallery package (hexperiment-gallery, Requirements 4.1, 4.5).

GALLERY_FORBIDDEN_IMPORTS = {"pandas", "sklearn", "scipy", "mcp"}


def gallery_modules() -> list[Path]:
    return [p for p in package_modules() if module_id(p).startswith("gallery/")]


def test_the_walk_finds_the_gallery_modules():
    found = {module_id(path) for path in gallery_modules()}
    assert {
        "gallery/__init__.py",
        "gallery/newlines.py",
        "gallery/markup.py",
        "gallery/records.py",
    } <= found


@pytest.mark.parametrize("path", gallery_modules(), ids=module_id)
def test_no_gallery_module_imports_the_cli_or_a_heavy_library(path):
    tree = parse(path)
    assert imported_roots(tree) & GALLERY_FORBIDDEN_IMPORTS == set()
    assert "hexperiment.cli" not in module_names(tree)


def literal_calls_without_a_string_literal(tree: ast.AST) -> list[int]:
    """Line numbers of calls to ``literal(...)`` whose argument is not a str literal."""
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name != "literal":
            continue
        argument = node.args[0] if len(node.args) == 1 and not node.keywords else None
        if not (isinstance(argument, ast.Constant) and isinstance(argument.value, str)):
            bad.append(node.lineno)
    return bad


def test_every_call_to_literal_has_a_string_literal_argument():
    offenders = {
        module_id(path): lines
        for path in gallery_modules()
        if (lines := literal_calls_without_a_string_literal(parse(path)))
    }
    assert offenders == {}


def test_only_the_markup_module_builds_markup_directly():
    builders = [
        module_id(path)
        for path in gallery_modules()
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Markup"
            for node in ast.walk(parse(path))
        )
    ]
    assert builders == ["gallery/markup.py"]


def test_the_literal_rule_catches_a_computed_argument():
    assert literal_calls_without_a_string_literal(ast.parse("literal(x)")) == [1]
    assert literal_calls_without_a_string_literal(ast.parse("m.literal(f'{x}')")) == [1]
    assert literal_calls_without_a_string_literal(ast.parse("literal('<br>')")) == []


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


# Start-up cost (Requirement 4.8). Each test starts a fresh interpreter, because the
# test process has long since imported everything.


def loaded_after(code: str, names: tuple[str, ...]) -> list[str]:
    """Run ``code`` in a fresh interpreter; return which of ``names`` it loaded."""
    script = (
        "import sys\n"
        f"{code}\n"
        f"print(','.join(m for m in {names!r} if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return [name for name in result.stdout.strip().split(",") if name]


def test_importing_the_server_module_loads_neither_sklearn_nor_scipy():
    loaded = loaded_after(
        "import hexperiment.mcp_server.server", ("sklearn", "scipy")
    )
    assert loaded == []


def test_importing_hexperiment_loads_neither_pandas_nor_sklearn_nor_scipy():
    loaded = loaded_after("import hexperiment", ("pandas", "sklearn", "scipy"))
    assert loaded == []


def test_the_lazy_public_functions_resolve_by_attribute_access():
    code = (
        "import hexperiment\n"
        "assert 'load' in dir(hexperiment)\n"
        "assert 'audit' in dir(hexperiment)\n"
        "assert 'pandas' not in sys.modules\n"
        "from hexperiment.data import load\n"
        "assert hexperiment.load is load\n"
        "assert hexperiment.split.__module__ == 'hexperiment.data'\n"
        "assert hexperiment.audit.__module__ == 'hexperiment.auditor'\n"
        "assert hexperiment.load_history.__module__ == 'hexperiment.runs'\n"
    )
    assert loaded_after(code, ("pandas",)) == ["pandas"]


def test_an_unknown_package_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        hexperiment.no_such_name  # noqa: B018
