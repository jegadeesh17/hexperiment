"""Loading and validation of ``hex.toml``, and the execution mode (from the env)."""

from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from hexperiment.errors import HexInputError

CONFIG_FILENAME = "hex.toml"
ENV_MODE = "HEX_MODE"
ENV_RUN_ID = "HEX_RUN_ID"

# Checks that a waiver may name. The canary, determinism, split_order and input
# findings can never be waived (Requirement 10.7).
WAIVABLE_CHECKS = ("row_overlap", "target_proxy", "temporal")
MIN_WAIVER_REASON_LENGTH = 10

_TABLE_KEYS = MappingProxyType(
    {
        "app": frozenset({"name", "description"}),
        "data": frozenset({"path"}),
        "target": frozenset({"column", "positive"}),
        "metric": frozenset({"name", "threshold"}),
        "split": frozenset({"test_fraction", "seed", "time_column"}),
        "checks": frozenset(
            {
                "proxy_auc_max",
                "overlap_max_fraction",
                "too_good_margin",
                "max_test_uses",
            }
        ),
        "run": frozenset({"timeout_seconds"}),
    }
)
_WAIVER_KEYS = frozenset({"check", "column", "reason"})
_TOP_LEVEL_KEYS = frozenset(_TABLE_KEYS) | {"waivers"}
_SUPPORTED_METRICS = ("roc_auc",)


class ExecMode(StrEnum):
    """How one execution of ``train.py`` behaves (set by ``HEX_MODE``)."""

    NORMAL = "normal"
    CANARY = "canary"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class AppConfig:
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class DataConfig:
    path: str


@dataclass(frozen=True, slots=True)
class TargetConfig:
    column: str
    positive: int | str | bool = 1


@dataclass(frozen=True, slots=True)
class MetricConfig:
    name: str
    threshold: float


@dataclass(frozen=True, slots=True)
class SplitConfig:
    test_fraction: float = 0.2
    seed: int = 0
    time_column: str | None = None


@dataclass(frozen=True, slots=True)
class ChecksConfig:
    proxy_auc_max: float = 0.95
    overlap_max_fraction: float = 0.0
    too_good_margin: float = 0.15
    max_test_uses: int = 5


@dataclass(frozen=True, slots=True)
class RunConfig:
    timeout_seconds: float = 120


@dataclass(frozen=True, slots=True)
class Waiver:
    check: str
    reason: str
    column: str | None = None


@dataclass(frozen=True, slots=True)
class HexConfig:
    app: AppConfig
    data: DataConfig
    target: TargetConfig
    metric: MetricConfig
    split: SplitConfig
    checks: ChecksConfig
    run: RunConfig
    waivers: tuple[Waiver, ...]
    app_dir: Path


def load_config(app_dir: Path) -> HexConfig:
    """Read and validate ``<app_dir>/hex.toml`` (Requirements 3.1 to 3.4)."""
    source = Path(app_dir) / CONFIG_FILENAME
    try:
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError:
        raise HexInputError(
            f"{source} does not exist.",
            f"Create {CONFIG_FILENAME} in the app folder with [data], [target] "
            "and [metric] tables.",
        ) from None
    except UnicodeDecodeError:
        raise HexInputError(
            f"{source} is not valid UTF-8 text.",
            f"Save {CONFIG_FILENAME} as UTF-8; TOML files must be UTF-8.",
        ) from None
    except tomllib.TOMLDecodeError as error:
        raise HexInputError(
            f"{source} is not valid TOML ({error}).",
            f"Fix the syntax error in {CONFIG_FILENAME}.",
        ) from None
    except OSError as error:
        raise HexInputError(
            f"{source} cannot be read ({error.strerror or error}).",
            "Check that the file is readable.",
        ) from None
    return parse_config(raw, source=source, app_dir=Path(app_dir))


def parse_config(raw: Mapping[str, Any], *, source: Path, app_dir: Path) -> HexConfig:
    """Validate parsed TOML and build a ``HexConfig``. Pure: reads no file."""
    _reject_unknown(raw, _TOP_LEVEL_KEYS, source, "table", prefix=None)
    tables = {name: _table(raw, name, source) for name in _TABLE_KEYS}
    return HexConfig(
        app=_parse_app(tables["app"], source, app_dir),
        data=_parse_data(tables["data"], source),
        target=_parse_target(tables["target"], source),
        metric=_parse_metric(tables["metric"], source),
        split=_parse_split(tables["split"], source),
        checks=_parse_checks(tables["checks"], source),
        run=_parse_run(tables["run"], source),
        waivers=_parse_waivers(raw.get("waivers", []), source),
        app_dir=app_dir,
    )


def current_mode(environ: Mapping[str, str] | None = None) -> ExecMode:
    """Return the execution mode: ``normal`` when ``HEX_MODE`` is unset."""
    env = os.environ if environ is None else environ
    value = env.get(ENV_MODE)
    if value is None:
        return ExecMode.NORMAL
    try:
        return ExecMode(value)
    except ValueError:
        allowed = ", ".join(mode.value for mode in ExecMode)
        raise HexInputError(
            f"environment variable {ENV_MODE} is {value!r}, "
            "which is not an execution mode.",
            f"Unset {ENV_MODE} or set it to one of: {allowed}.",
        ) from None


def run_id_from_env(environ: Mapping[str, str] | None = None) -> str | None:
    """Return ``HEX_RUN_ID``, or ``None`` when it is unset (a standalone run)."""
    env = os.environ if environ is None else environ
    value = env.get(ENV_RUN_ID)
    if value is None:
        return None
    if not value.strip():
        raise HexInputError(
            f"environment variable {ENV_RUN_ID} is set but empty.",
            f"Unset {ENV_RUN_ID} for a standalone run, "
            "or let `hexperiment run` set it.",
        )
    return value


# --- table parsers ----------------------------------------------------------------


def _parse_app(table: Mapping[str, Any], source: Path, app_dir: Path) -> AppConfig:
    name = app_dir.resolve().name
    if "name" in table:
        name = _string(table["name"], "app.name", source)
    description = ""
    if "description" in table:
        description = _string(
            table["description"], "app.description", source, allow_empty=True
        )
    return AppConfig(name=name, description=description)


def _parse_data(table: Mapping[str, Any], source: Path) -> DataConfig:
    path = _require(table, "data", "path", source)
    return DataConfig(path=_string(path, "data.path", source))


def _parse_target(table: Mapping[str, Any], source: Path) -> TargetConfig:
    column = _string(
        _require(table, "target", "column", source), "target.column", source
    )
    positive = table.get("positive", 1)
    if not isinstance(positive, bool | int | str):
        raise HexInputError(
            f"target.positive in {source} must be an integer, a string or a boolean, "
            f"not {type(positive).__name__}.",
            "Set [target] positive to the value that marks the positive class, "
            "for example 1.",
        )
    return TargetConfig(column=column, positive=positive)


def _parse_metric(table: Mapping[str, Any], source: Path) -> MetricConfig:
    name = _string(_require(table, "metric", "name", source), "metric.name", source)
    if name not in _SUPPORTED_METRICS:
        raise HexInputError(
            f"metric.name in {source} is {name!r}, which is not supported.",
            f'Set [metric] name = "{_SUPPORTED_METRICS[0]}".',
        )
    threshold = _bounded(
        _require(table, "metric", "threshold", source),
        "metric.threshold",
        source,
        lo=0,
        hi=1,
    )
    return MetricConfig(name=name, threshold=threshold)


def _parse_split(table: Mapping[str, Any], source: Path) -> SplitConfig:
    d = SplitConfig()
    fraction = d.test_fraction
    if "test_fraction" in table:
        fraction = _bounded(
            table["test_fraction"], "split.test_fraction", source, lo=0, hi=1
        )
    seed = d.seed
    if "seed" in table:
        seed = _integer(table["seed"], "split.seed", source, lo=0)
    time_column = d.time_column
    if "time_column" in table:
        time_column = _string(table["time_column"], "split.time_column", source)
    return SplitConfig(test_fraction=fraction, seed=seed, time_column=time_column)


def _parse_checks(table: Mapping[str, Any], source: Path) -> ChecksConfig:
    d = ChecksConfig()
    proxy_auc_max = d.proxy_auc_max
    if "proxy_auc_max" in table:
        proxy_auc_max = _bounded(
            table["proxy_auc_max"],
            "checks.proxy_auc_max",
            source,
            lo=0,
            hi=1,
            hi_inclusive=True,
        )
    overlap_max_fraction = d.overlap_max_fraction
    if "overlap_max_fraction" in table:
        overlap_max_fraction = _bounded(
            table["overlap_max_fraction"],
            "checks.overlap_max_fraction",
            source,
            lo=0,
            hi=1,
            lo_inclusive=True,
            hi_inclusive=True,
        )
    too_good_margin = d.too_good_margin
    if "too_good_margin" in table:
        too_good_margin = _bounded(
            table["too_good_margin"],
            "checks.too_good_margin",
            source,
            lo=0,
            lo_inclusive=True,
        )
    max_test_uses = d.max_test_uses
    if "max_test_uses" in table:
        max_test_uses = _integer(
            table["max_test_uses"], "checks.max_test_uses", source, lo=1
        )
    return ChecksConfig(
        proxy_auc_max=proxy_auc_max,
        overlap_max_fraction=overlap_max_fraction,
        too_good_margin=too_good_margin,
        max_test_uses=max_test_uses,
    )


def _parse_run(table: Mapping[str, Any], source: Path) -> RunConfig:
    timeout = RunConfig().timeout_seconds
    if "timeout_seconds" in table:
        timeout = _bounded(
            table["timeout_seconds"], "run.timeout_seconds", source, lo=0
        )
    return RunConfig(timeout_seconds=timeout)


def _parse_waivers(raw: Any, source: Path) -> tuple[Waiver, ...]:
    if not isinstance(raw, list) or not all(isinstance(w, Mapping) for w in raw):
        raise HexInputError(
            f"waivers in {source} must be an array of tables.",
            "Write each waiver as a [[waivers]] table with check and reason.",
        )
    return tuple(_parse_waiver(w, i, source) for i, w in enumerate(raw, start=1))


def _parse_waiver(item: Mapping[str, Any], index: int, source: Path) -> Waiver:
    label = f"waivers[{index}]"
    _reject_unknown(item, _WAIVER_KEYS, source, "key", prefix=label)
    check = _string(
        _require_waiver(item, label, "check", source), f"{label}.check", source
    )
    if check not in WAIVABLE_CHECKS:
        raise HexInputError(
            f"{label}.check in {source} is {check!r}, which cannot be waived.",
            f"Waive only one of: {', '.join(WAIVABLE_CHECKS)}. "
            "Fix the cause of any other finding instead.",
        )
    reason = _string(
        _require_waiver(item, label, "reason", source),
        f"{label}.reason",
        source,
        allow_empty=True,
    ).strip()
    if len(reason) < MIN_WAIVER_REASON_LENGTH:
        raise HexInputError(
            f"{label}.reason in {source} is too short: a waiver needs at least "
            f"{MIN_WAIVER_REASON_LENGTH} characters of justification.",
            "Explain why the finding is acceptable, for the reviewer who reads "
            "the model card.",
        )
    column: str | None = None
    if "column" in item:
        column = _string(item["column"], f"{label}.column", source)
        if check != "target_proxy":
            raise HexInputError(
                f"{label}.column in {source} is only allowed with "
                f'check = "target_proxy", but check is {check!r}.',
                "Remove the column key from this waiver.",
            )
    elif check == "target_proxy":
        raise HexInputError(
            f"{label}.column is missing from {source}: a target_proxy waiver must "
            "name the column it covers.",
            "Add `column = ...` with the name of the feature to this waiver, so "
            "that it cannot cover a leaking column that appears later.",
        )
    return Waiver(check=check, reason=reason, column=column)


# --- value helpers -----------------------------------------------------------------


def _missing(key: str, source: Path) -> HexInputError:
    table, _, name = key.partition(".")
    return HexInputError(
        f"required key {key} is missing from {source}.",
        f"Add `{name} = ...` under [{table}] in {CONFIG_FILENAME}.",
    )


def _require(table: Mapping[str, Any], table_name: str, key: str, source: Path) -> Any:
    if key not in table:
        raise _missing(f"{table_name}.{key}", source)
    return table[key]


def _require_waiver(item: Mapping[str, Any], label: str, key: str, source: Path) -> Any:
    if key not in item:
        raise HexInputError(
            f"{label}.{key} is missing from {source}.",
            f"Add `{key} = ...` to this [[waivers]] table.",
        )
    return item[key]


def _table(raw: Mapping[str, Any], name: str, source: Path) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise HexInputError(
            f"[{name}] in {source} must be a table, not {type(value).__name__}.",
            f"Write it as a [{name}] table with key = value lines.",
        )
    _reject_unknown(value, _TABLE_KEYS[name], source, "key", prefix=name)
    return value


def _reject_unknown(
    mapping: Mapping[str, Any],
    allowed: frozenset[str],
    source: Path,
    kind: str,
    *,
    prefix: str | None,
) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if not unknown:
        return
    names = ", ".join(f"{prefix}.{key}" if prefix else key for key in unknown)
    raise HexInputError(
        f"unknown {kind} {names} in {source}.",
        f"Check the spelling. Allowed here: {', '.join(sorted(allowed))}.",
    )


def _string(value: Any, name: str, source: Path, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        kind = "a string" if allow_empty else "a non-empty string"
        raise HexInputError(
            f"{name} in {source} must be {kind}.",
            f"Set {name} to a quoted text value.",
        )
    return value


def _integer(value: Any, name: str, source: Path, *, lo: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HexInputError(
            f"{name} in {source} must be an integer, not {value!r}.",
            f"Set {name} to a whole number.",
        )
    if value < lo:
        raise HexInputError(
            f"{name} in {source} is {value}, but it must be at least {lo}.",
            f"Set {name} to {lo} or more.",
        )
    return value


def _bounded(
    value: Any,
    name: str,
    source: Path,
    *,
    lo: float | None = None,
    hi: float | None = None,
    lo_inclusive: bool = False,
    hi_inclusive: bool = False,
) -> float:
    """Return ``value`` as a float after checking it lies in the given interval."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise HexInputError(
            f"{name} in {source} must be a finite number, not {value!r}.",
            f"Set {name} to a number.",
        )
    too_low = lo is not None and (value < lo if lo_inclusive else value <= lo)
    too_high = hi is not None and (value > hi if hi_inclusive else value >= hi)
    if too_low or too_high:
        parts = []
        if lo is not None:
            parts.append(f"{'at least' if lo_inclusive else 'greater than'} {lo:g}")
        if hi is not None:
            parts.append(f"{'at most' if hi_inclusive else 'less than'} {hi:g}")
        wanted = " and ".join(parts)
        raise HexInputError(
            f"{name} in {source} is {value}, but it must be {wanted}.",
            f"Set {name} to a value that is {wanted}.",
        )
    return float(value)
