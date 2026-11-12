"""The pages of the gallery as markup (pure).

This module holds the page shell, the status mark, link helpers and the case page.
Everything that comes from an app goes through the builder in ``markup.py``, which
escapes it.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from types import MappingProxyType
from urllib.parse import urlsplit

from hexperiment.gallery.difftext import diff_markup, diff_scripts
from hexperiment.gallery.markup import Child, Markup, code, el, prose, spaced
from hexperiment.gallery.modelcard import model_card_section
from hexperiment.gallery.records import AppSource
from hexperiment.gallery.summary import AppSummary, score_text
from hexperiment.models import Finding, Severity, Status, Verdict

FONTS_URL = (
    "https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,200;0,400;0,600;1,400"
    "&family=IBM+Plex+Mono:wght@400;500&display=swap"
)

LANDING_PATH = "index.html"
STYLESHEET_PATH = "assets/gallery.css"
LANDING_TITLE = "Hexperiment: audited model runs"

_UTC_TIME = re.compile(r"(\d{4}-\d\d-\d\d)T(\d\d:\d\d:\d\d)Z")
_SEVERITY_ORDER = MappingProxyType(
    {Severity.ERROR: 0, Severity.WAIVED: 1, Severity.WARNING: 2}
)


def relative_href(from_path: str, to_path: str) -> str:
    """Link from the page ``from_path`` to the file ``to_path`` (both site keys)."""
    return posixpath.relpath(to_path, start=posixpath.dirname(from_path) or ".")


def repo_link_label(repo_url: str) -> str:
    """"Source on GitHub" for github.com, "Source code" for any other host."""
    host = urlsplit(repo_url).hostname
    return "Source on GitHub" if host == "github.com" else "Source code"


def status_mark(status: Status | str) -> Markup:
    """The status word with a decorative mark (hidden from assistive technology)."""
    word = Status(status).value
    return el(
        "span",
        el("span", class_="mark", aria_hidden="true"),
        word,
        class_=f"status status-{word.lower()}",
    )


def _repo_link(repo_url: str | None) -> Markup | None:
    if repo_url is None:
        return None
    return el("a", repo_link_label(repo_url), href=repo_url, class_="source")


def page_shell(
    *, title: str, body: Markup, path: str, repo_url: str | None
) -> Markup:
    """The document around ``body`` for the page that will be stored at ``path``."""
    head = el(
        "head",
        el("meta", charset="utf-8"),
        el("meta", name="viewport", content="width=device-width, initial-scale=1"),
        el("title", title),
        el("link", rel="stylesheet", href=FONTS_URL),
        el("link", rel="stylesheet", href=relative_href(path, STYLESHEET_PATH)),
    )
    home = relative_href(path, LANDING_PATH)
    page = el(
        "body",
        el("a", "Skip to main content", href="#main", class_="skip"),
        el(
            "header",
            el("a", "Hexperiment", href=home, class_="site"),
            _repo_link(repo_url),
        ),
        el("main", body, id="main", tabindex="-1"),
        el("footer", _repo_link(repo_url), el("p", "Released under the MIT licence.")),
    )
    return el("html", head, page, lang="en")


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def _utc_text(created_at: str) -> tuple[str, str | None]:
    """The time as shown, and the value for ``datetime`` when it is well formed."""
    match = _UTC_TIME.fullmatch(created_at)
    if match is None:
        return created_at, None
    return f"{match.group(1)} {match.group(2)} UTC", created_at


def _time_element(created_at: str) -> Markup:
    shown, machine = _utc_text(created_at)
    return el("time", shown, datetime=machine)


def _finding_item(finding: Finding) -> Markup:
    name: Child = (
        el("em", finding.ghost, class_="ghost") if finding.ghost else finding.check
    )
    heading: list[Child] = [name, finding.severity.value]
    if finding.location:
        heading += ["at", code(finding.location)]
    parts: list[Child] = [
        el("p", spaced(*heading)),
        el("p", prose(finding.message)),
        el("p", spaced("Fix:", prose(finding.fix))),
    ]
    if finding.severity is Severity.WAIVED:
        reason = finding.waiver_reason or ""
        parts.append(el("p", spaced("Waiver reason:", prose(reason))))
    return el("li", parts, class_=f"finding finding-{finding.severity.value}")


def _findings_list(verdict: Verdict) -> Markup:
    if not verdict.findings:
        return el("p", "No findings.", class_="no-findings")
    ordered = sorted(verdict.findings, key=lambda f: _SEVERITY_ORDER[f.severity])
    return el("ul", [_finding_item(f) for f in ordered], class_="findings")


def _run_item(verdict: Verdict) -> Markup:
    uses = f"test set used {_plural(verdict.test_set_uses, 'time')}"
    head = spaced(
        status_mark(verdict.status),
        verdict.mode.value,
        score_text(verdict.metric.value),
        _time_element(verdict.created_at),
        uses,
    )
    findings = _findings_list(verdict)
    return el("li", el("p", head, class_="run-head"), findings, class_="run")


def _record(runs: Sequence[Verdict]) -> Markup:
    return el(
        "section",
        el("h2", "The record"),
        el("ol", [_run_item(run) for run in runs], class_="record"),
    )


def _fix_section(source: AppSource) -> Markup | None:
    if source.haunted_py is None or source.train_py is None:
        return None
    old_name, new_name = "haunted/train.py", "train.py"
    diff = diff_scripts(source.haunted_py, source.train_py)
    if not diff.blocks:
        body: list[Child] = [el("p", "The two scripts are identical.")]
    else:
        counts = (
            f": {_plural(diff.removed, 'line')} removed, "
            f"{_plural(diff.added, 'line')} added."
        )
        body = [
            el("p", code(old_name), " against ", code(new_name), counts),
            diff_markup(diff, old_name=old_name, new_name=new_name),
        ]
    return el("section", el("h2", "The fix"), body)


def _card_section(summary: AppSummary, path: str) -> Markup:
    approved = summary.last_approved
    if approved is None:
        content: Child = el("p", "No approved run yet.")
    else:
        href = relative_href(path, f"{summary.slug}/model_card.md")
        content = model_card_section(approved, card_href=href)
    return el("section", el("h2", "Model card of the last approved run"), content)


def _raw_files(summary: AppSummary, path: str) -> Markup:
    def link(name: str) -> Markup:
        href = relative_href(path, f"{summary.slug}/{name}")
        return el("li", el("a", name, href=href))

    links = [link("verdict.json")]
    if summary.last_approved is not None:
        links.append(link("model_card.md"))
    return el("section", el("h2", "Raw files"), el("ul", links))


def _facts(summary: AppSummary) -> Markup:
    return el(
        "p",
        "target ",
        code(summary.target_column),
        f", metric {summary.metric_name}, threshold {score_text(summary.threshold)}, "
        f"{_plural(len(summary.runs), 'recorded run')}",
        class_="facts",
    )


def case_page(summary: AppSummary, source: AppSource, repo_url: str | None) -> Markup:
    """The whole page ``<slug>/index.html`` of one app."""
    path = f"{summary.slug}/index.html"
    body = el(
        "div",
        el("p", el("a", "All cases", href=relative_href(path, LANDING_PATH))),
        el("h1", summary.name),
        el("p", summary.description) if summary.description else None,
        _facts(summary),
        _record(summary.runs),
        _fix_section(source),
        _card_section(summary, path),
        _raw_files(summary, path),
    )
    return page_shell(
        title=f"Hexperiment: {summary.name}", body=body, path=path, repo_url=repo_url
    )
