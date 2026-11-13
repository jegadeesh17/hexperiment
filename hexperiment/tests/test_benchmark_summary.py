"""The results object, the README block and the splice (pure; nothing is run)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexbench.execution import RunRecord
from hexbench.mechanisms import MECHANISMS, NOT_MEASURED
from hexbench.report import BEGIN, END, NO_CHECK, render_block, splice_block
from hexbench.summary import SOURCE, STATUSES, CaseRecord, build_results

ENVIRONMENT = {
    "hexperiment": "0.1.0",
    "python": "3.13.1",
    "numpy": "2.2.1",
    "pandas": "2.2.3",
    "scikit-learn": "1.6.0",
    "os": "Windows-11",
    "cpu_count": 8,
    "run_seconds": 12.5,
}
FIRST, SECOND = MECHANISMS[0], MECHANISMS[1]


def run(
    status: str = "APPROVED",
    score: float | None = 0.7,
    ghosts: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    seconds: float = 3.04,
    message: str | None = None,
) -> RunRecord:
    return RunRecord(status, score, ghosts, warnings, seconds, message)


def case(
    mechanism: str,
    model: str,
    leaky: RunRecord,
    clean: RunRecord | None = None,
) -> CaseRecord:
    return CaseRecord(
        f"{mechanism}-{model}", mechanism, model, 1000, leaky, clean or run()
    )


def build(records: Sequence[CaseRecord], only: str | None = None) -> dict[str, Any]:
    return build_results(
        records,
        mechanisms=MECHANISMS,
        not_measured=NOT_MEASURED,
        environment=ENVIRONMENT,
        created_at="2026-01-15T10:42:07Z",
        only=only,
    )  # type: ignore[return-value]


def three(
    key: str, leaky: Sequence[RunRecord], clean: Sequence[RunRecord] | None = None
) -> list[CaseRecord]:
    cleans = clean or [run()] * 3
    return [
        case(key, model, lk, cl)
        for model, lk, cl in zip(("lr", "rf", "hgb"), leaky, cleans, strict=True)
    ]


def rejected(*ghosts: str, score: float | None = 0.9) -> RunRecord:
    return run("REJECTED", score, tuple(sorted(ghosts)))


# --- build_results --------------------------------------------------------------------


def test_header_fields() -> None:
    result = build(three(FIRST.key, [rejected("A")] * 3))
    assert result["format"] == "hexperiment-benchmark/1"
    assert result["created_at"] == "2026-01-15T10:42:07Z"
    assert result["complete"] is True
    assert result["only"] is None
    assert result["environment"] == ENVIRONMENT
    assert result["source"] == SOURCE
    assert "doi:10.1016/j.patter.2023.100804" in result["source"]
    assert result["not_measured"][0]["leak_type"] == "L3.3"


def test_counts_and_item_fields() -> None:
    leaky = [rejected("The Doppelganger"), rejected("The Doppelganger"), run()]
    item = build(three(FIRST.key, leaky))["mechanisms"][0]
    assert item["key"] == FIRST.key
    assert item["leak_type"] == FIRST.leak_type
    assert item["mechanism"] == FIRST.mechanism
    assert item["designed_to_catch"] == list(FIRST.designed_to_catch)
    assert (item["scripts"], item["rejected"]) == (3, 2)
    assert item["statuses"] == {
        "APPROVED": 1,
        "REJECTED": 2,
        "UNVERIFIED": 0,
        "ERROR": 0,
    }


def test_by_ghost_counts_a_script_once_per_key() -> None:
    leaky = [
        rejected("The Doppelganger", "The Possession"),
        rejected("The Doppelganger"),
        rejected("The Possession"),
    ]
    item = build(three(FIRST.key, leaky))["mechanisms"][0]
    assert item["by_ghost"] == {"The Doppelganger": 2, "The Possession": 2}


def test_duplicate_keys_within_a_script_count_once() -> None:
    record = run("REJECTED", 0.9, ("X", "X"))
    item = build(three(FIRST.key, [record] * 3))["mechanisms"][0]
    assert item["by_ghost"] == {"X": 3}


def test_statuses_are_zero_filled_and_error_is_not_rejected() -> None:
    error = run("ERROR", None, message="boom")
    unverified = run("UNVERIFIED", 0.7)
    item = build(three(FIRST.key, [error, unverified, rejected("A")]))["mechanisms"][0]
    assert tuple(item["statuses"]) == STATUSES
    assert item["statuses"] == {
        "APPROVED": 0,
        "REJECTED": 1,
        "UNVERIFIED": 1,
        "ERROR": 1,
    }
    assert item["rejected"] == 1
    assert item["by_ghost"] == {"A": 1}


def test_means_cover_runs_with_a_score_and_round_to_four_decimals() -> None:
    leaky = [rejected("A", score=0.9), rejected("A", score=0.8), run("ERROR", None)]
    clean = [run(score=0.7), run(score=None), run(score=0.6)]
    item = build(three(FIRST.key, leaky, clean))["mechanisms"][0]
    assert item["mean_score_leaky"] == 0.85
    assert item["mean_score_clean"] == 0.65
    thirds = [rejected("A", score=s) for s in (0.1, 0.1, 0.2)]
    item = build(three(FIRST.key, thirds))["mechanisms"][0]
    assert item["mean_score_leaky"] == 0.1333


def test_means_are_null_without_scores() -> None:
    errors = [run("ERROR", None, message="x")] * 3
    item = build(three(FIRST.key, errors, errors))["mechanisms"][0]
    assert item["mean_score_leaky"] is None
    assert item["mean_score_clean"] is None


def test_clean_false_alarms_and_ghosts() -> None:
    cleans = [rejected("The Peeking Spirit"), run(), run("ERROR", None, message="m")]
    result = build(three(FIRST.key, [run()] * 3, cleans))
    assert result["clean"] == {
        "scripts": 3,
        "false_alarms": 1,
        "by_ghost": {"The Peeking Spirit": 1},
        "statuses": {"APPROVED": 1, "REJECTED": 1, "UNVERIFIED": 0, "ERROR": 1},
    }


def test_only_gives_an_incomplete_file_with_one_mechanism() -> None:
    result = build(three(SECOND.key, [rejected("A")] * 3), only=SECOND.key)
    assert result["complete"] is False
    assert result["only"] == SECOND.key
    assert [m["key"] for m in result["mechanisms"]] == [SECOND.key]


def test_mechanisms_follow_the_file_order_not_the_record_order() -> None:
    records = three(SECOND.key, [run()] * 3) + three(FIRST.key, [run()] * 3)
    keys = [m["key"] for m in build(records)["mechanisms"]]
    assert keys == [FIRST.key, SECOND.key]


def test_cases_agree_with_the_counts() -> None:
    leaky = [rejected("A"), run(), run("ERROR", None, message="boom")]
    result = build(three(FIRST.key, leaky))
    cases = result["cases"]
    assert [c["id"] for c in cases] == [f"{FIRST.key}-{m}" for m in ("lr", "rf", "hgb")]
    counts = [c["leaky"]["status"] for c in cases]
    item = result["mechanisms"][0]
    assert {s: counts.count(s) for s in STATUSES} == item["statuses"]
    assert cases[0]["leaky"] == {
        "status": "REJECTED",
        "score": 0.9,
        "ghosts": ["A"],
        "warnings": [],
        "seconds": 3.0,
        "message": None,
    }
    assert cases[2]["leaky"]["message"] == "boom"


def test_result_is_json_serialisable() -> None:
    result = build(three(FIRST.key, [rejected("A"), run(), run("ERROR", None)]))
    assert json.loads(json.dumps(result)) == result


def test_input_key_is_counted_as_rejected() -> None:
    record = run("REJECTED", None, ("input",))
    item = build(three(FIRST.key, [record] * 3))["mechanisms"][0]
    assert item["rejected"] == 3
    assert item["by_ghost"] == {"input": 3}


def test_unknown_status_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown run status"):
        build(three(FIRST.key, [run("MAYBE")] * 3))


_RUNS = st.builds(
    RunRecord,
    status=st.sampled_from(STATUSES),
    score=st.one_of(st.none(), st.floats(0, 1)),
    ghosts=st.lists(st.sampled_from(["A", "B", "input"]), max_size=3).map(tuple),
    warnings=st.just(()),
    seconds=st.floats(0, 300),
    message=st.none(),
)


@given(
    st.lists(
        st.tuples(st.sampled_from([m.key for m in MECHANISMS]), _RUNS, _RUNS),
        min_size=1,
        max_size=12,
    )
)
def test_counts_are_consistent_for_any_records(
    rows: list[tuple[str, RunRecord, RunRecord]],
) -> None:
    records = [case(key, f"m{i}", lk, cl) for i, (key, lk, cl) in enumerate(rows)]
    result = build(records)
    for item in result["mechanisms"]:
        assert sum(item["statuses"].values()) == item["scripts"]
        assert item["rejected"] == item["statuses"]["REJECTED"]
        assert all(n <= item["rejected"] for n in item["by_ghost"].values())
    clean = result["clean"]
    assert sum(clean["statuses"].values()) == clean["scripts"] == len(records)
    assert all(n <= clean["false_alarms"] for n in clean["by_ghost"].values())


# --- render_block and splice_block ----------------------------------------------------


def _full_results() -> dict[str, Any]:
    records: list[CaseRecord] = []
    for mechanism in MECHANISMS:
        leaky = [rejected("The Peeking Spirit")] * 3
        records += three(mechanism.key, leaky)
    return build(records)


def test_block_is_delimited_and_has_a_row_per_mechanism() -> None:
    block = render_block(_full_results())
    lines = block.splitlines()
    assert lines[0] == BEGIN
    assert lines[-1] == END
    rows = [
        line for line in lines if line.startswith("| L") and "Leak type" not in line
    ]
    assert len(rows) == len(MECHANISMS)
    assert "| L1.1 No test set |" in rows[0]
    assert "3 of 3 (100%)" in rows[0]
    assert "The Peeking Spirit (3)" in rows[0]


def test_block_says_when_no_check_is_designed() -> None:
    assert NO_CHECK in render_block(_full_results())


def test_block_has_the_clean_row_the_not_measured_list_and_the_citation() -> None:
    block = render_block(_full_results())
    assert "clean twins: 0 false alarms in 39 scripts" in block
    assert "- L3.3 Sampling bias in test distribution:" in block
    assert f"Source: {SOURCE}" in block
    assert (
        "Run with hexperiment 0.1.0, Python 3.13.1 and scikit-learn 1.6.0 "
        "on 2026-01-15: 78 runs." in block
    )


def test_block_marks_unverified_and_error_runs() -> None:
    leaky = [rejected("A"), run("UNVERIFIED"), run("ERROR", None, message="x")]
    block = render_block(build(three(FIRST.key, leaky)))
    assert "1 of 3 (33%), 1 unverified, 1 error" in block


def test_block_shows_n_a_for_missing_means() -> None:
    errors = [run("ERROR", None, message="x")] * 3
    block = render_block(build(three(FIRST.key, errors, errors)))
    assert "| n/a | n/a |" in block


def test_block_is_deterministic() -> None:
    results = _full_results()
    assert render_block(results) == render_block(results)


README = f"intro\n\n{BEGIN}\nold\n{END}\n\noutro\n"


def test_splice_replaces_the_block_and_keeps_the_rest() -> None:
    block = render_block(_full_results())
    spliced = splice_block(README, block)
    assert spliced.startswith("intro\n\n" + BEGIN)
    assert spliced.endswith(END + "\n\noutro\n")
    assert "old" not in spliced
    assert block in spliced


def test_splice_is_idempotent() -> None:
    block = render_block(_full_results())
    once = splice_block(README, block)
    assert splice_block(once, block) == once


@pytest.mark.parametrize(
    "readme",
    [
        "no markers",
        f"{BEGIN}\nonly begin",
        f"only end\n{END}",
        f"{BEGIN}\n{END}\n{BEGIN}\n{END}",
        f"{BEGIN}\n{END}\n{END}",
    ],
)
def test_splice_rejects_a_missing_or_repeated_marker(readme: str) -> None:
    with pytest.raises(ValueError, match="exactly once"):
        splice_block(readme, "x")
