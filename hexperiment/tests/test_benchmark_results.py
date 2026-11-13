"""The committed benchmark results, their README block and the isolation rules.

Nothing here runs a case or starts a process. The tests fail (not skip) when
``results.json`` is absent: run task 17 of the benchmark plan first.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from hexbench.execution import RunRecord
from hexbench.mechanisms import MECHANISMS, NOT_MEASURED
from hexbench.report import BEGIN, END, render_block
from hexbench.summary import STATUSES, CaseRecord, build_results

PACKAGE_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PACKAGE_DIR / "benchmark"
RESULTS_PATH = BENCHMARK_DIR / "results.json"
README_PATH = BENCHMARK_DIR / "README.md"
SRC_DIR = PACKAGE_DIR / "src" / "hexperiment"
DOI = "10.1016/j.patter.2023.100804"
ALLOWED_THIRD_PARTY = {"hexbench", "numpy", "pandas"}
NUMPY_PANDAS_MODULES = {"datagen.py", "cases.py"}


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    if not RESULTS_PATH.is_file():
        pytest.fail(
            f"{RESULTS_PATH} is missing. Run the complete benchmark "
            "(task 17 of .kiro/specs/hexperiment-benchmark/tasks.md) and commit it."
        )
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme() -> str:
    if not README_PATH.is_file():
        pytest.fail(f"{README_PATH} is missing (task 10 and task 17).")
    return README_PATH.read_text(encoding="utf-8")


# --- results.json ---------------------------------------------------------------------


def test_header_fields(results: dict[str, Any]) -> None:
    assert results["format"] == "hexperiment-benchmark/1"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", results["created_at"])
    datetime.strptime(results["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    assert results["complete"] is True
    assert results["only"] is None
    assert DOI in results["source"]


def test_environment_keys_and_types(results: dict[str, Any]) -> None:
    env = results["environment"]
    texts = ("hexperiment", "python", "numpy", "pandas", "scikit-learn", "os")
    for key in texts:
        assert isinstance(env[key], str) and env[key], key
    assert isinstance(env["cpu_count"], int) and not isinstance(env["cpu_count"], bool)
    assert isinstance(env["run_seconds"], int | float)
    assert not isinstance(env["run_seconds"], bool)
    assert set(env) == {*texts, "cpu_count", "run_seconds"}


def test_mechanisms_match_the_code(results: dict[str, Any]) -> None:
    items = results["mechanisms"]
    assert [i["key"] for i in items] == [m.key for m in MECHANISMS]
    for item, mech in zip(items, MECHANISMS, strict=True):
        assert item["leak_type"] == mech.leak_type, mech.key
        assert item["mechanism"] == mech.mechanism, mech.key
        assert item["designed_to_catch"] == list(mech.designed_to_catch), mech.key
        assert item["leak_type_name"], mech.key


def test_leak_type_names_match_the_code(results: dict[str, Any]) -> None:
    from hexbench.mechanisms import LEAK_TYPE_NAMES

    for item in results["mechanisms"]:
        assert item["leak_type_name"] == LEAK_TYPE_NAMES[item["leak_type"]]


def test_mechanism_counts_are_consistent(results: dict[str, Any]) -> None:
    for item in results["mechanisms"]:
        key = item["key"]
        assert item["scripts"] == 3, key
        assert set(item["statuses"]) == set(STATUSES), key
        assert sum(item["statuses"].values()) == item["scripts"], key
        assert item["rejected"] == item["statuses"]["REJECTED"], key
        assert all(n <= item["rejected"] for n in item["by_ghost"].values()), key
        for field in ("mean_score_leaky", "mean_score_clean"):
            assert item[field] is None or isinstance(item[field], int | float), key


def test_clean_section(results: dict[str, Any]) -> None:
    clean = results["clean"]
    assert clean["scripts"] == 39
    assert set(clean["statuses"]) == set(STATUSES)
    assert sum(clean["statuses"].values()) == clean["scripts"]
    assert clean["false_alarms"] == clean["statuses"]["REJECTED"]
    assert all(n <= clean["false_alarms"] for n in clean["by_ghost"].values())


def test_not_measured_matches_the_code(results: dict[str, Any]) -> None:
    expected = [
        {
            "leak_type": n.leak_type,
            "leak_type_name": n.leak_type_name,
            "reason": n.reason,
        }
        for n in NOT_MEASURED
    ]
    assert results["not_measured"] == expected


def _record(case: dict[str, Any]) -> CaseRecord:
    def run(item: dict[str, Any]) -> RunRecord:
        return RunRecord(
            status=item["status"],
            score=item["score"],
            ghosts=tuple(item["ghosts"]),
            warnings=tuple(item["warnings"]),
            seconds=item["seconds"],
            message=item["message"],
        )

    return CaseRecord(
        case_id=case["id"],
        mechanism=case["mechanism"],
        model=case["model"],
        data_seed=case["data_seed"],
        leaky=run(case["leaky"]),
        clean=run(case["clean"]),
    )


def test_cases_are_39_and_agree_with_the_counts(results: dict[str, Any]) -> None:
    cases = results["cases"]
    assert len(cases) == 39
    assert len({c["id"] for c in cases}) == 39
    for item in results["mechanisms"]:
        mine = [c for c in cases if c["mechanism"] == item["key"]]
        assert len(mine) == item["scripts"]
        for status in STATUSES:
            counted = sum(1 for c in mine if c["leaky"]["status"] == status)
            assert counted == item["statuses"][status], (item["key"], status)


def test_build_results_reproduces_the_file(results: dict[str, Any]) -> None:
    rebuilt = build_results(
        [_record(c) for c in results["cases"]],
        mechanisms=MECHANISMS,
        not_measured=NOT_MEASURED,
        environment=results["environment"],
        created_at=results["created_at"],
        only=results["only"],
    )
    assert rebuilt["mechanisms"] == results["mechanisms"]
    assert rebuilt["clean"] == results["clean"]


# --- README.md ------------------------------------------------------------------------


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}\s*$", text, re.MULTILINE)
    assert match, f"README.md has no section '## {heading}'"
    rest = text[match.end() :]
    following = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: following.start()] if following else rest


def test_readme_block_equals_render_block(results: dict[str, Any], readme: str) -> None:
    assert readme.count(BEGIN) == 1 and readme.count(END) == 1
    start = readme.index(BEGIN)
    end = readme.index(END) + len(END)
    assert readme[start:end] == render_block(results)


def test_readme_statements(readme: str) -> None:
    assert DOI in readme
    assert "Kapoor and Narayanan" in readme
    assert "synthetic" in readme
    assert "written by the authors of the checks" in readme
    assert "L3.3" in readme
    assert "## Changes after the first complete run" in readme


def test_readme_names_every_missed_mechanism(
    results: dict[str, Any], readme: str
) -> None:
    missed = [m["key"] for m in results["mechanisms"] if m["rejected"] < m["scripts"]]
    section = _section(readme, "Missed or partly caught")
    _section(readme, "What we expected before the first run")
    for key in missed:
        assert f"`{key}`" in section, f"{key} is missed but not named in the README"


# --- isolation (Requirement 6.2) ------------------------------------------------------


def test_nothing_benchmark_under_src() -> None:
    names = [p.name for p in SRC_DIR.rglob("*") if "__pycache__" not in p.parts]
    assert not [n for n in names if n in ("benchmark", "hexbench")]
    for path in SRC_DIR.rglob("*.py"):
        assert "hexbench" not in path.read_text(encoding="utf-8"), path


def test_pyproject_dependencies_and_package_data() -> None:
    data = tomllib.loads((PACKAGE_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    assert sorted(data["project"]["dependencies"]) == [
        "numpy",
        "pandas",
        "scikit-learn",
    ]
    package_data = data["tool"]["setuptools"]["package-data"]
    assert "benchmark" not in json.dumps(package_data)
    assert "hexbench" not in json.dumps(package_data)


def _benchmark_modules() -> list[Path]:
    return sorted(
        p for p in BENCHMARK_DIR.rglob("*.py") if "__pycache__" not in p.parts
    )


def _imported_modules(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_benchmark_imports_are_restricted() -> None:
    modules = _benchmark_modules()
    assert modules
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _imported_modules(tree):
            top = name.split(".")[0]
            where = f"{path.name} imports {name}"
            if top in sys.stdlib_module_names:
                if top == "subprocess":
                    assert path.name == "execution.py", where
                continue
            assert top in ALLOWED_THIRD_PARTY, where
            assert top != "sklearn", where
            if top in ("numpy", "pandas"):
                assert path.name in NUMPY_PANDAS_MODULES, where


def test_benchmark_never_calls_print() -> None:
    for path in _benchmark_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", f"{path.name}:{node.lineno} calls print"
