"""The benchmark results file and its table on the landing page.

Pure in behaviour: nothing here reads a file or the clock. The module imports
``datetime`` to parse ``created_at``, so it is not in the architecture test's
``PURE_FILES``. The file format is ``hexperiment-benchmark``, Requirement 4; the
gallery validates every key it shows and ignores the others (Requirement 6).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from hexperiment.errors import HexInputError
from hexperiment.gallery.markup import Child, Markup, el, spaced
from hexperiment.gallery.records import Benchmark, BenchmarkMechanism, NotMeasured

BENCHMARK_FORMAT = "hexperiment-benchmark/1"

BENCHMARK_CAVEAT = (
    "The data is synthetic and the cases were written by the authors of the "
    "checks, so this shows that the checks see the leaks they were built for, "
    "not how often they find leaks in real projects."
)

_REGENERATE = "Regenerate the file with the benchmark runner."
_ADD_KEY = "Add the key, or regenerate the file."
_TOP_KEYS = (
    "created_at",
    "environment",
    "source",
    "mechanisms",
    "clean",
    "not_measured",
)
_ENVIRONMENT_KEYS = ("hexperiment", "python", "scikit-learn")
_MECHANISM_KEYS = (
    "leak_type",
    "leak_type_name",
    "mechanism",
    "designed_to_catch",
    "scripts",
    "rejected",
    "by_ghost",
)
_CLEAN_KEYS = ("scripts", "false_alarms")
_NOT_MEASURED_KEYS = ("leak_type", "leak_type_name", "reason")
_JSON_TYPES = (
    (bool, "a boolean"),
    (int, "a number"),
    (float, "a number"),
    (str, "a string"),
    (list, "a list"),
    (dict, "an object"),
    (type(None), "null"),
)


def _fail(source: str, problem: str, fix: str) -> HexInputError:
    return HexInputError(f"{source}: {problem}.", fix)


def _json_type(value: object) -> str:
    for kind, name in _JSON_TYPES:
        if type(value) is kind:
            return name
    return type(value).__name__


def _is_int(value: object) -> bool:
    return type(value) is int


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not allowed in JSON")


def _load(text: str, source: str) -> dict[str, Any]:
    fix = "Write the file as JSON, or regenerate it with the benchmark runner."
    try:
        data = json.loads(text.removeprefix("\ufeff"), parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise _fail(
            source,
            f"the file is not valid JSON ({error.msg}, line {error.lineno}, "
            f"column {error.colno})",
            fix,
        ) from error
    except (ValueError, RecursionError) as error:
        raise _fail(source, f"the file is not valid JSON ({error})", fix) from error
    if not isinstance(data, dict):
        raise _fail(
            source, f"the top level must be a JSON object, not {_json_type(data)}", fix
        )
    return data


def _check_header(data: Mapping[str, Any], source: str) -> None:
    if "format" not in data:
        raise _fail(
            source,
            "format is missing",
            "Regenerate the file with the benchmark runner of this version.",
        )
    if data["format"] != BENCHMARK_FORMAT:
        raise _fail(
            source,
            f"format is {data['format']!r}, but this gallery reads "
            f"{BENCHMARK_FORMAT!r}",
            "Regenerate the file with the matching version, or build the gallery "
            "with the version that wrote it.",
        )
    if type(data.get("complete")) is not bool:
        raise _fail(source, "complete is missing or is not true or false", _REGENERATE)
    if data["complete"] is False:
        raise _fail(
            source,
            "complete is false: the file was written with --only and covers some "
            "mechanisms only",
            "Run the full benchmark without --only and pass that results file.",
        )


def _require_keys(
    obj: Mapping[str, Any], keys: tuple[str, ...], path: str, source: str
) -> None:
    for key in keys:
        if key not in obj:
            raise _fail(source, f"{path}{key} is missing", _ADD_KEY)


def _parse_created_at(value: object, source: str) -> tuple[str, str]:
    fix = "Write the time of the benchmark run in UTC."
    problem = (
        "created_at must be an ISO 8601 time in UTC (2026-11-11T19:50:51Z or "
        f"ending in +00:00), not {value!r}"
    )
    if not isinstance(value, str):
        raise _fail(source, problem, fix)
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as error:
        raise _fail(source, problem, fix) from error
    if moment.tzinfo is None or moment.utcoffset() != timedelta(0):
        raise _fail(source, problem, fix)
    return value, moment.astimezone(UTC).date().isoformat()


def _parse_environment(value: object, source: str) -> tuple[str, str, str]:
    fix = 'Write the versions that ran the benchmark, for example "0.1.0".'
    if not isinstance(value, dict):
        raise _fail(
            source, f"environment must be an object, not {_json_type(value)}", fix
        )
    for key in _ENVIRONMENT_KEYS:
        if not _non_empty_string(value.get(key)):
            raise _fail(
                source,
                f"environment.{key} is missing or is not a non-empty string",
                fix,
            )
    return value["hexperiment"], value["python"], value["scikit-learn"]


def _parse_by_ghost(value: object, rejected: int, path: str, source: str) -> tuple:
    fix = "List only ghosts that rejected at least one script."
    if not isinstance(value, dict):
        raise _fail(source, f"{path} must be an object, not {_json_type(value)}", fix)
    for name, count in value.items():
        if not _is_int(count) or not 1 <= count <= rejected:
            raise _fail(
                source,
                f"{path}.{name} must be an integer from 1 to {rejected} (the value "
                f"of rejected), not {count!r}",
                fix,
            )
    return tuple(sorted(value.items(), key=lambda pair: (-pair[1], pair[0])))


def _parse_mechanism(item: object, index: int, source: str) -> BenchmarkMechanism:
    path = f"mechanisms[{index}]"
    if not isinstance(item, dict):
        raise _fail(
            source,
            f"{path} must be an object, not {_json_type(item)}",
            _ADD_KEY,
        )
    _require_keys(item, _MECHANISM_KEYS, f"{path}.", source)
    for key in ("leak_type", "leak_type_name", "mechanism"):
        if not _non_empty_string(item[key]):
            raise _fail(
                source,
                f"{path}.{key} must be a non-empty string",
                "Write the leakage type ID, its name and the mechanism sentence.",
            )
    designed = item["designed_to_catch"]
    if not isinstance(designed, list) or not all(map(_non_empty_string, designed)):
        raise _fail(
            source,
            f"{path}.designed_to_catch must be a list of non-empty strings",
            "List the ghost names, or an empty list when no check is designed for it.",
        )
    scripts = item["scripts"]
    if not _is_int(scripts) or scripts < 1:
        raise _fail(
            source,
            f"{path}.scripts must be an integer of at least 1, not {scripts!r}",
            "A rate needs at least one script.",
        )
    rejected = item["rejected"]
    if not _is_int(rejected) or not 0 <= rejected <= scripts:
        raise _fail(
            source,
            f"{path}.rejected must be an integer from 0 to {scripts} (the value of "
            f"scripts), not {rejected!r}",
            "Count only the scripts the run rejected.",
        )
    by_ghost = _parse_by_ghost(item["by_ghost"], rejected, f"{path}.by_ghost", source)
    return BenchmarkMechanism(
        leak_type=item["leak_type"],
        leak_type_name=item["leak_type_name"],
        mechanism=item["mechanism"],
        designed_to_catch=tuple(designed),
        scripts=scripts,
        rejected=rejected,
        by_ghost=by_ghost,
    )


def _parse_mechanisms(value: object, source: str) -> tuple[BenchmarkMechanism, ...]:
    if not isinstance(value, list) or not value:
        raise _fail(
            source,
            "mechanisms must be a non-empty list",
            "Regenerate the file; a benchmark without mechanisms shows nothing.",
        )
    return tuple(_parse_mechanism(item, i, source) for i, item in enumerate(value))


def _parse_clean(value: object, source: str) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise _fail(
            source, f"clean must be an object, not {_json_type(value)}", _ADD_KEY
        )
    _require_keys(value, _CLEAN_KEYS, "clean.", source)
    scripts = value["scripts"]
    if not _is_int(scripts) or scripts < 1:
        raise _fail(
            source,
            f"clean.scripts must be an integer of at least 1, not {scripts!r}",
            "A rate needs at least one clean script.",
        )
    alarms = value["false_alarms"]
    if not _is_int(alarms) or not 0 <= alarms <= scripts:
        raise _fail(
            source,
            f"clean.false_alarms must be an integer from 0 to {scripts} (the value "
            f"of clean.scripts), not {alarms!r}",
            "Count only clean scripts that were rejected.",
        )
    return scripts, alarms


def _parse_not_measured(value: object, source: str) -> tuple[NotMeasured, ...]:
    fix = "Write the ID, name and reason of each type that was not measured."
    if not isinstance(value, list):
        raise _fail(
            source, f"not_measured must be a list, not {_json_type(value)}", fix
        )
    items = []
    for index, item in enumerate(value):
        path = f"not_measured[{index}]"
        if not isinstance(item, dict):
            raise _fail(
                source, f"{path} must be an object, not {_json_type(item)}", fix
            )
        for key in _NOT_MEASURED_KEYS:
            if not _non_empty_string(item.get(key)):
                raise _fail(
                    source, f"{path}.{key} is missing or is not a non-empty string", fix
                )
        items.append(
            NotMeasured(item["leak_type"], item["leak_type_name"], item["reason"])
        )
    return tuple(items)


def parse_benchmark(text: str, *, source: str) -> Benchmark:
    """Validate a results file; report the first problem as ``HexInputError``.

    Every message starts with ``source`` and names the key path.
    """
    data = _load(text, source)
    _check_header(data, source)
    _require_keys(data, _TOP_KEYS, "", source)
    created_at, day = _parse_created_at(data["created_at"], source)
    hexperiment, python, scikit = _parse_environment(data["environment"], source)
    if not _non_empty_string(data["source"]):
        raise _fail(
            source, "source must be a non-empty string", "Write the taxonomy citation."
        )
    mechanisms = _parse_mechanisms(data["mechanisms"], source)
    clean_scripts, false_alarms = _parse_clean(data["clean"], source)
    not_measured = _parse_not_measured(data["not_measured"], source)
    return Benchmark(
        created_at=created_at,
        day=day,
        hexperiment_version=hexperiment,
        python_version=python,
        scikit_learn_version=scikit,
        source=data["source"],
        mechanisms=mechanisms,
        clean_scripts=clean_scripts,
        false_alarms=false_alarms,
        not_measured=not_measured,
    )


def run_count(benchmark: Benchmark) -> int:
    """Every leaky and every clean script is one full run."""
    return sum(m.scripts for m in benchmark.mechanisms) + benchmark.clean_scripts


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%"


def _plural(count: int, singular: str) -> str:
    return f"{count} {singular if count == 1 else singular + 's'}"


def _ghost_list(names: tuple[str, ...], ghost_names: frozenset[str]) -> Child:
    if not names:
        return "No check is designed for it"
    parts: list[Child] = []
    for index, name in enumerate(names):
        if index:
            parts.append(", ")
        parts.append(el("em", name, class_="ghost") if name in ghost_names else name)
    return parts


def _by_ghost_text(mechanism: BenchmarkMechanism, ghost_names: frozenset[str]) -> Child:
    if not mechanism.by_ghost:
        return "none"
    parts: list[Child] = []
    for index, (name, count) in enumerate(mechanism.by_ghost):
        if index:
            parts.append(", ")
        shown: Child = el("em", name, class_="ghost") if name in ghost_names else name
        parts.append(spaced(shown, f"({count})"))
    return parts


def _mechanism_row(
    mechanism: BenchmarkMechanism,
    ghost_names: frozenset[str],
    group_cell: Markup | None,
) -> Markup:
    rate = percent(mechanism.rejected, mechanism.scripts)
    return el(
        "tr",
        group_cell,
        el("td", mechanism.mechanism),
        el("td", _ghost_list(mechanism.designed_to_catch, ghost_names)),
        el("td", f"rejected {mechanism.rejected} of {mechanism.scripts} ({rate})"),
        el("td", _by_ghost_text(mechanism, ghost_names)),
    )


def _groups(
    mechanisms: tuple[BenchmarkMechanism, ...],
) -> list[list[BenchmarkMechanism]]:
    """Mechanisms grouped by leakage type in order of first appearance."""
    groups: dict[str, list[BenchmarkMechanism]] = {}
    for mechanism in mechanisms:
        groups.setdefault(mechanism.leak_type, []).append(mechanism)
    return list(groups.values())


def _group_body(group: list[BenchmarkMechanism], ghost_names: frozenset[str]) -> Markup:
    first = group[0]
    cell = el(
        "th",
        f"{first.leak_type} {first.leak_type_name}",
        scope="rowgroup",
        rowspan=str(len(group)),
    )
    rows = [
        _mechanism_row(mechanism, ghost_names, cell if index == 0 else None)
        for index, mechanism in enumerate(group)
    ]
    return el("tbody", rows)


def _clean_body(benchmark: Benchmark) -> Markup:
    alarms = _plural(benchmark.false_alarms, "false alarm")
    total = _plural(benchmark.clean_scripts, "script")
    rate = percent(benchmark.false_alarms, benchmark.clean_scripts)
    return el(
        "tbody",
        el(
            "tr",
            el("th", "Clean twins", scope="row", colspan="3"),
            el("td", f"{alarms} in {total} ({rate})", colspan="2"),
        ),
    )


def benchmark_totals(benchmark: Benchmark) -> str:
    """The totals sentence, summed over the mechanisms of the file."""
    leaky = sum(m.scripts for m in benchmark.mechanisms)
    rejected = sum(m.rejected for m in benchmark.mechanisms)
    return (
        f"{rejected} of {leaky} leaky scripts rejected; "
        f"{_plural(benchmark.false_alarms, 'false alarm')} in "
        f"{benchmark.clean_scripts} clean twins."
    )


def benchmark_section(benchmark: Benchmark, ghost_names: frozenset[str]) -> Markup:
    """The ``<section>`` with the results table and the notes below it."""
    headers = (
        "Leakage type",
        "Mechanism",
        "Designed to catch",
        "Rejected",
        "Rejected by",
    )
    table = el(
        "table",
        el("caption", "Leaky scripts rejected, by leakage type and mechanism"),
        el("thead", el("tr", [el("th", h, scope="col") for h in headers])),
        [_group_body(group, ghost_names) for group in _groups(benchmark.mechanisms)],
        _clean_body(benchmark),
    )
    parts: list[Child] = [
        el("h2", "Benchmark", id="benchmark"),
        el(
            "p",
            "Each leaky script was run through hexperiment run; its clean twin is "
            "the same script with the leak removed.",
        ),
        el("p", benchmark_totals(benchmark)),
        el("p", BENCHMARK_CAVEAT),
        el(
            "div",
            table,
            class_="scroll",
            role="region",
            tabindex="0",
            aria_label="Benchmark results",
        ),
    ]
    if benchmark.not_measured:
        items = [
            el("li", f"{item.leak_type} {item.leak_type_name}: {item.reason}")
            for item in benchmark.not_measured
        ]
        parts += [el("h3", "Not measured"), el("ul", items)]
    parts += [
        el("p", f"Leakage types follow {benchmark.source}."),
        el(
            "p",
            f"Measured with hexperiment {benchmark.hexperiment_version}, Python "
            f"{benchmark.python_version} and scikit-learn "
            f"{benchmark.scikit_learn_version} on {benchmark.day}, in "
            f"{run_count(benchmark)} runs.",
        ),
    ]
    return el("section", parts, aria_labelledby="benchmark")
