"""The model card of one verdict as a page section (pure).

The number formats are those of ``card.py`` (positive rates ``.1%``, metric and
threshold ``.4f``, counts as integers). ``card.py`` keeps its helpers private, so the
formats are repeated here and a test compares the two outputs (Requirement 3.4).
"""

from __future__ import annotations

from hexperiment.gallery.markup import Child, Markup, el
from hexperiment.models import Verdict


def _pairs(pairs: list[tuple[str, Child]]) -> Markup:
    items: list[Child] = []
    for term, value in pairs:
        items.append(el("dt", term))
        items.append(el("dd", value))
    return el("dl", items)


def _scroll(label: str, table: Markup) -> Markup:
    return el(
        "div", table, class_="scroll", role="region", tabindex="0", aria_label=label
    )


def _metric_pairs(verdict: Verdict) -> list[tuple[str, Child]]:
    metric = verdict.metric
    if metric.value is None:
        value, outcome = "not computed", "failed"
    else:
        value = f"{metric.value:.4f}"
        outcome = "passed" if metric.passed else "failed"
    return [
        (metric.name, value),
        ("Threshold", f"{metric.threshold:.4f}"),
        ("Result", outcome),
    ]


def _split_pairs(verdict: Verdict) -> list[tuple[str, Child]]:
    split = verdict.split
    return [
        ("Data file", el("code", split.data_path)),
        ("Test fraction", str(split.test_fraction)),
        ("Seed", str(split.seed)),
        ("Time column", el("code", split.time_column) if split.time_column else "none"),
    ]


def _data_pairs(verdict: Verdict) -> list[tuple[str, Child]]:
    data = verdict.data
    if data is None:
        return [("Data", "Not available: the run stopped on an input error.")]
    pairs: list[tuple[str, Child]] = [
        (
            "Training rows",
            f"{data.n_train} (positive rate {data.positive_rate_train:.1%})",
        ),
        ("Test rows", f"{data.n_test} (positive rate {data.positive_rate_test:.1%})"),
    ]
    if verdict.split.time_column is not None:
        for label, span in (
            ("Training time range", data.train_time_range),
            ("Test time range", data.test_time_range),
        ):
            if span is not None:
                pairs.append((label, f"{span[0]} to {span[1]}"))
    return pairs


def _checks_table(verdict: Verdict) -> Markup:
    if not verdict.checks:
        return el("p", "No checks ran.")
    headers = ("Check", "Ghost", "Outcome")
    head = el("thead", el("tr", [el("th", h, scope="col") for h in headers]))
    rows = [
        el(
            "tr",
            el("td", c.name),
            el("td", el("em", c.ghost, class_="ghost") if c.ghost else "-"),
            el("td", c.outcome.value),
        )
        for c in verdict.checks
    ]
    return _scroll("Checks", el("table", head, el("tbody", rows)))


def _waivers_table(verdict: Verdict) -> Markup:
    if not verdict.waivers:
        return el("p", "No waivers applied.")
    headers = ("Check", "Column", "Reason", "Findings waived")
    head = el("thead", el("tr", [el("th", h, scope="col") for h in headers]))
    rows = [
        el(
            "tr",
            el("td", w.check),
            el("td", w.column or "all columns"),
            el("td", w.reason),
            el("td", str(w.findings_waived)),
        )
        for w in verdict.waivers
    ]
    return _scroll("Waivers", el("table", head, el("tbody", rows)))


def _group(title: str, content: Markup, *, wide: bool = False) -> Markup:
    """One ``h3`` with the content that follows it."""
    return el(
        "div",
        el("h3", title),
        content,
        class_="card-group wide" if wide else "card-group",
    )


def model_card_section(verdict: Verdict) -> Markup:
    """The card of ``verdict``: status, metric, split, data, checks, waivers, uses.

    Each heading and its content form one ``div.card-group``; the groups that hold
    a table are ``wide``. The link to ``model_card.md`` is on the page, not here.
    """
    status = _pairs(
        [
            ("Verdict", verdict.status.value),
            ("Mode", verdict.mode.value),
            ("Run id", el("code", verdict.run_id)),
            ("Date", verdict.created_at),
            ("hexperiment version", verdict.hexperiment_version),
        ]
    )
    uses = (
        f"The test set has been evaluated {verdict.test_set_uses} time(s), "
        "including this run."
    )
    return el(
        "div",
        _group("Status", status),
        _group("Metric", _pairs(_metric_pairs(verdict))),
        _group("Split", _pairs(_split_pairs(verdict))),
        _group("Data", _pairs(_data_pairs(verdict))),
        _group("Checks", _checks_table(verdict), wide=True),
        _group("Waivers", _waivers_table(verdict), wide=True),
        _group("Test-set uses", el("p", uses)),
        class_="card",
    )
