import dataclasses
import re
from html import unescape

from gallery_builder import make_verdict
from hexperiment import card
from hexperiment.gallery.modelcard import model_card_section
from hexperiment.models import (
    CheckSummary,
    DataSummary,
    Outcome,
    SplitSignature,
    WaiverRecord,
)


def build(verdict):
    return model_card_section(verdict).html


def rich_verdict(**changes):
    verdict = make_verdict(value=0.9096, time_column="when")
    verdict = dataclasses.replace(
        verdict,
        checks=(
            CheckSummary("target_proxy", "The Possession", Outcome.PASSED),
            CheckSummary("metric", None, Outcome.PASSED),
        ),
        data=DataSummary(
            80,
            20,
            0.375,
            0.25,
            ("2020-01-01", "2020-06-30"),
            ("2020-07-01", "2020-12-31"),
        ),
        waivers=(
            WaiverRecord("target_proxy", "price", "price is known at booking", 1),
        ),
        test_set_uses=3,
    )
    return dataclasses.replace(verdict, **changes)


def test_every_value_of_requirement_3_4_appears():
    verdict = rich_verdict()
    text = unescape(build(verdict))
    for expected in (
        "APPROVED",
        "full",
        verdict.run_id,
        verdict.created_at,
        "0.1.0",
        "0.9096",
        "0.8500",
        "passed",
        "data.csv",
        "0.2",
        "when",
        "80 (positive rate 37.5%)",
        "20 (positive rate 25.0%)",
        "2020-01-01 to 2020-06-30",
        "2020-07-01 to 2020-12-31",
        "target_proxy",
        "The Possession",
        "3 time(s)",
    ):
        assert expected in text


def test_the_card_is_groups_of_a_heading_and_its_content():
    html = build(rich_verdict())
    assert html.startswith('<div class="card"><div class="card-group"><h3>Status</h3>')
    titles = re.findall(r'<div class="card-group[^"]*"><h3>([^<]*)</h3>', html)
    assert titles == [
        "Status",
        "Metric",
        "Split",
        "Data",
        "Checks",
        "Waivers",
        "Test-set uses",
    ]
    assert html.count("<h3>") == html.count('<div class="card-group') == 7
    assert "model-card" not in html


def test_only_the_groups_with_a_table_are_wide():
    html = build(rich_verdict())
    wide = re.findall(r'<div class="card-group wide"><h3>([^<]*)</h3>', html)
    assert wide == ["Checks", "Waivers"]
    assert html.count('class="card-group"') == 5


def test_the_section_does_not_link_the_card_file():
    html = build(rich_verdict())
    assert "model_card.md" not in html
    assert "<a " not in html


def test_waivers_are_listed_with_their_reasons():
    text = unescape(build(rich_verdict()))
    assert "price is known at booking" in text
    assert "No waivers applied." not in text


def test_no_waivers_says_so():
    text = build(make_verdict())
    assert "No waivers applied." in text


def test_without_time_column_no_time_ranges():
    text = build(make_verdict(time_column=None))
    assert "Time column" in text
    assert "none" in text
    assert "time range" not in text


def test_a_failed_metric_without_value():
    verdict = make_verdict(value=None)
    text = build(verdict)
    assert "not computed" in text
    assert "failed" in text


def test_a_verdict_without_data():
    text = build(dataclasses.replace(make_verdict(), data=None))
    assert "Not available" in text


def test_tables_scroll_in_their_own_labelled_box():
    text = build(rich_verdict())
    assert text.count('class="scroll" role="region" tabindex="0"') == 2
    assert 'aria-label="Checks"' in text
    assert 'aria-label="Waivers"' in text


def test_formatted_values_also_appear_in_the_card_text():
    verdict = rich_verdict()
    markdown = card.render(verdict, ())
    for value in ("0.9096", "0.8500", "37.5%", "25.0%", "80", "20", "3 time(s)"):
        assert value in markdown
        assert value in unescape(build(verdict))
    assert "2020-01-01 to 2020-06-30" in markdown
    assert "2020-01-01 to 2020-06-30" in unescape(build(verdict))


def test_hostile_waiver_reason_is_escaped():
    hostile = "<script>alert(1)</script>"
    verdict = rich_verdict(
        waivers=(WaiverRecord("target_proxy", None, hostile, 1),),
        split=SplitSignature("<b>.csv", 0.2, 0, None),
    )
    text = build(verdict)
    assert "<script>" not in text
    assert "<b>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "all columns" in text
