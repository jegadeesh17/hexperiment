"""Frozen dataclasses that carry the inputs of a gallery build."""

from __future__ import annotations

from dataclasses import dataclass

from hexperiment.config import HexConfig
from hexperiment.models import Verdict


@dataclass(frozen=True, slots=True)
class AppSource:
    config: HexConfig  # name, description, target, metric; app_dir is never rendered
    history: tuple[Verdict, ...]  # oldest first, never empty
    last_verdict_json: bytes  # runs/<last run>/verdict.json, LF line endings
    approved_card: bytes | None  # model_card.md of the last approved run, LF endings
    train_py: str | None  # LF line endings; None unless haunted_py is not None
    haunted_py: str | None  # LF line endings


@dataclass(frozen=True, slots=True)
class BenchmarkMechanism:
    leak_type: str  # "L1.2"
    leak_type_name: str  # the paper's name
    mechanism: str  # one sentence
    designed_to_catch: tuple[str, ...]  # ghost names, empty when no check is designed
    scripts: int  # at least 1
    rejected: int  # 0 to scripts
    by_ghost: tuple[tuple[str, int], ...]  # by count (descending), then name


@dataclass(frozen=True, slots=True)
class NotMeasured:
    leak_type: str
    leak_type_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class Benchmark:
    created_at: str  # as in the file
    day: str  # YYYY-MM-DD, the UTC date of created_at
    hexperiment_version: str
    python_version: str
    scikit_learn_version: str
    source: str  # the taxonomy citation
    mechanisms: tuple[BenchmarkMechanism, ...]  # file order
    clean_scripts: int
    false_alarms: int
    not_measured: tuple[NotMeasured, ...]  # file order


@dataclass(frozen=True, slots=True)
class GalleryInputs:
    apps: tuple[AppSource, ...]  # in the order given on the command line
    benchmark: Benchmark | None
    repo_url: str | None
    stylesheet: bytes
    fonts: tuple[tuple[str, bytes], ...] = ()  # (file name, bytes), sorted by name
