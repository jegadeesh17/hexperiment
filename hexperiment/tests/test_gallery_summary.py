import math
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gallery_builder import (
    approved,
    make_finding,
    make_source,
    make_verdict,
    rejected,
)
from hexperiment.canary import PEEKING_SPIRIT
from hexperiment.errors import HexInputError
from hexperiment.gallery.summary import (
    RESERVED_SLUGS,
    assign_slugs,
    choose_hero,
    ghosts_found,
    score_text,
    slugify,
    summarize_app,
    suspect_lines,
)
from hexperiment.models import Severity, Status

PROPERTY = settings(max_examples=50, deadline=None)


def summary(name: str, runs):
    return summarize_app(slugify(name) or "x", make_source(name, runs))


# slugs


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Hotel Cancellations", "hotel-cancellations"),
        ("--a--", "a"),
        ("a !!! b", "a-b"),
        ("Model 2", "model-2"),
        ("größe", "gr-e"),
        ("!!!", ""),
        ("", ""),
    ],
)
def test_slugify_examples(name, slug):
    assert slugify(name) == slug


def test_assign_slugs_returns_the_slugs_in_order():
    pairs = [("apps/a", "Alpha One"), ("apps/b", "Beta")]
    assert assign_slugs(pairs) == ("alpha-one", "beta")


def test_an_empty_slug_is_refused_with_the_fix():
    with pytest.raises(HexInputError, match=r"apps/a.*empty.*\[app\] name"):
        assign_slugs([("apps/a", "!!!")])


def test_assets_is_refused():
    with pytest.raises(HexInputError, match=r"'assets'.*stylesheet.*Rename"):
        assign_slugs([("apps/a", "Assets")])


@pytest.mark.parametrize("name", ["CON", "nul", "com1", "LPT9", "aux", "prn"])
def test_a_windows_device_name_is_refused(name):
    with pytest.raises(HexInputError, match="device name"):
        assign_slugs([("apps/a", name)])


def test_the_reserved_slugs_hold_the_documented_names():
    assert {"assets", "con", "prn", "aux", "nul", "com1", "com9", "lpt1", "lpt9"} <= (
        RESERVED_SLUGS
    )
    assert "com0" not in RESERVED_SLUGS


def test_two_apps_with_the_same_slug_are_refused_naming_both():
    with pytest.raises(HexInputError, match=r"apps/a and apps/b.*'x'.*different"):
        assign_slugs([("apps/a", "X"), ("apps/b", "x!")])


def test_the_same_folder_twice_is_refused():
    with pytest.raises(HexInputError, match=r"apps/a and apps/a"):
        assign_slugs([("apps/a", "X"), ("apps/a", "X")])


@PROPERTY
@given(st.text())
def test_a_slug_matches_the_pattern_and_is_idempotent(name):
    slug = slugify(name)
    assert re.fullmatch(r"([a-z0-9]+(-[a-z0-9]+)*)?", slug)
    assert slugify(slug) == slug


# score text, ghosts, suspect lines


def test_score_text_has_four_decimals_or_no_score():
    assert score_text(1.0) == "1.0000"
    assert score_text(0.90964) == "0.9096"
    assert score_text(None) == "no score"


def test_ghosts_found_follow_catalogue_order_and_include_warnings_and_waivers():
    runs = [
        make_verdict(
            Status.REJECTED,
            findings=[
                make_finding(ghost=PEEKING_SPIRIT),
                make_finding(ghost="The Possession", severity=Severity.WAIVED),
                make_finding(ghost="The Shapeshifter", severity=Severity.WARNING),
                make_finding(ghost=None, check="metric"),
            ],
        ),
        make_verdict(findings=[make_finding(ghost="The Possession")]),
    ]
    assert ghosts_found(runs) == (
        "The Possession",
        PEEKING_SPIRIT,
        "The Shapeshifter",
    )


def test_a_ghost_outside_the_catalogue_goes_last_in_alphabetical_order():
    runs = [
        make_verdict(
            findings=[
                make_finding(ghost="Zed"),
                make_finding(ghost="Abe"),
                make_finding(ghost="The Prophet"),
            ]
        )
    ]
    assert ghosts_found(runs) == ("The Prophet", "Abe", "Zed")


def test_no_ghost_gives_an_empty_tuple():
    assert ghosts_found([make_verdict()]) == ()


def test_suspect_lines_are_parsed_from_the_real_fix_text():
    fix = (
        "Fit every step on training rows only. Select features on the training "
        "rows only. Suspect lines: train.py:100, train.py:104."
    )
    assert suspect_lines(make_finding(fix=fix)) == ("train.py:100", "train.py:104")


def test_suspect_lines_of_a_fix_without_them_is_empty():
    assert suspect_lines(make_finding(fix="Drop the column.")) == ()


def test_the_committed_hotel_run_pins_the_suspect_line_format():
    from pathlib import Path

    from hexperiment.runs import load_history

    app = Path(__file__).resolve().parents[2] / "apps" / "hotel-cancellations"
    if not (app / "runs").is_dir():
        pytest.skip("example app not present")
    findings = [
        f for v in load_history(app) for f in v.findings if f.ghost == PEEKING_SPIRIT
    ]
    assert findings
    assert any(suspect_lines(f) for f in findings)


# summaries


def test_summarize_app_copies_the_source_and_picks_runs():
    runs = [rejected(1.0), make_verdict(Status.UNVERIFIED, value=0.9), approved(0.91)]
    result = summarize_app("demo", make_source("Demo", runs, target="is_canceled"))
    assert result.name == "Demo"
    assert result.target_column == "is_canceled"
    assert result.metric_name == "roc_auc"
    assert result.threshold == 0.85
    assert result.first is runs[0]
    assert result.last is runs[2]
    assert result.first_rejected is runs[0]
    assert result.last_approved is runs[2]
    assert result.runs == tuple(runs)
    assert result.ghosts == ("The Possession",)


def test_summarize_app_without_rejected_or_approved_runs():
    result = summarize_app("d", make_source("d", [make_verdict(Status.UNVERIFIED)]))
    assert result.first_rejected is None
    assert result.last_approved is None


# hero


def test_the_highest_first_rejected_score_wins():
    a = summary("a", [rejected(0.95), approved(0.9)])
    b = summary("b", [rejected(1.0), approved(0.5)])
    assert choose_hero([a, b]).app is b
    assert choose_hero([b, a]).app is b


def test_the_hero_carries_the_first_rejected_and_last_approved_runs():
    first, second = rejected(1.0), approved(0.9)
    third = approved(0.9096)
    hero = choose_hero([summary("a", [first, second, third])])
    assert hero.rejected is first
    assert hero.approved is third


def test_a_tie_goes_to_the_larger_drop():
    a = summary("a", [rejected(1.0), approved(0.95)])
    b = summary("b", [rejected(1.0), approved(0.80)])
    assert choose_hero([a, b]).app is b


def test_a_tie_in_score_and_drop_goes_to_the_app_given_first():
    a = summary("a", [rejected(1.0), approved(0.9)])
    b = summary("b", [rejected(1.0), approved(0.9)])
    assert choose_hero([a, b]).app is a
    assert choose_hero([b, a]).app is b


def test_four_decimal_rounding_decides_a_tie():
    a = summary("a", [rejected(0.99996), approved(0.9)])
    b = summary("b", [rejected(1.0), approved(0.5)])
    # Both read 1.0000 (0.99996 rounds up), so the larger drop decides.
    assert choose_hero([a, b]).app is b


def test_rounding_can_separate_scores_that_are_close():
    a = summary("a", [rejected(0.99994), approved(0.1)])
    b = summary("b", [rejected(0.9999), approved(0.9)])
    # 0.99994 reads 0.9999, equal to b; a has the larger drop.
    assert choose_hero([a, b]).app is a


def test_no_approved_run_anywhere_still_chooses_the_highest_score_without_honest():
    a = summary("a", [rejected(0.9)])
    b = summary("b", [rejected(0.95)])
    hero = choose_hero([a, b])
    assert hero.app is b
    assert hero.approved is None


def test_no_approved_run_anywhere_and_a_tie_goes_to_the_app_given_first():
    a = summary("a", [rejected(0.9)])
    b = summary("b", [rejected(0.9)])
    assert choose_hero([a, b]).app is a


def test_an_app_without_an_approved_run_beats_a_lower_score_with_one():
    a = summary("a", [rejected(0.99)])
    b = summary("b", [rejected(0.9), approved(0.5)])
    assert choose_hero([a, b]).app is a


def test_no_rejected_run_gives_no_hero():
    assert choose_hero([summary("a", [approved(0.9)])]) is None
    assert choose_hero([]) is None


def test_a_rejected_run_without_a_score_is_never_chosen():
    a = summary("a", [rejected(None)])
    b = summary("b", [rejected(0.6)])
    assert choose_hero([a, b]).app is b
    assert choose_hero([a]) is None


def test_the_first_rejected_run_is_used_not_a_later_one():
    runs = [rejected(0.7), rejected(0.99)]
    hero = choose_hero([summary("a", runs)])
    assert hero.rejected is runs[0]


@PROPERTY
@given(
    st.lists(
        st.tuples(
            st.floats(0, 1, allow_nan=False) | st.none(),
            st.floats(0, 1, allow_nan=False) | st.none(),
        ),
        max_size=6,
    )
)
def test_the_hero_is_a_member_and_no_other_candidate_has_a_larger_key(pairs):
    summaries = []
    for index, (first, last) in enumerate(pairs):
        runs = [rejected(first)] + ([approved(last)] if last is not None else [])
        summaries.append(summarize_app(f"app{index}", make_source(f"app{index}", runs)))

    def key(index):
        first, last = pairs[index]
        score = float(f"{first:.4f}")
        drop = -math.inf if last is None else round(score - float(f"{last:.4f}"), 4)
        return (score, drop, -index)

    candidates = [i for i, (first, _) in enumerate(pairs) if first is not None]
    hero = choose_hero(summaries)
    if not candidates:
        assert hero is None
        return
    chosen = summaries.index(hero.app)
    assert chosen in candidates
    assert all(key(i) <= key(chosen) for i in candidates)
