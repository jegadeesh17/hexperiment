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

from hexperiment.gallery.benchmark import benchmark_section
from hexperiment.gallery.difftext import diff_markup, diff_scripts
from hexperiment.gallery.markup import Child, Markup, code, el, prose, spaced
from hexperiment.gallery.modelcard import model_card_section
from hexperiment.gallery.records import AppSource, Benchmark
from hexperiment.gallery.summary import AppSummary, Hero, score_text, suspect_lines
from hexperiment.ghosts import CATALOG
from hexperiment.models import Finding, Severity, Status, Verdict

FONTS_URL = (
    "https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,200;0,400;0,600;1,400"
    "&family=IBM+Plex+Mono:wght@400;500&display=swap"
)

LANDING_PATH = "index.html"
STYLESHEET_PATH = "assets/gallery.css"
LANDING_TITLE = "Hexperiment: audited model runs"
MOTTO = "If your model looks too good to be true, it's haunted."
CHECK_STEPS = (
    "Train normally and keep what the model learned.",
    "Train again with the test rows scrambled at load time.",
    "Compare the two. In a script without leakage nothing the model learned "
    "depends on the test rows, so nothing moves.",
)
# Reviewed by the owner: every claim must be true of the repository when published.
HOW_IT_WAS_BUILT = (
    "Hexperiment was written against Kiro specs. A hook runs the fast check each "
    "time a training script is saved, and a workflow repeats fix, retrain and the "
    "full check until the verdict is APPROVED. The kit ships as a Kiro power."
)
_COUNT_WORDS = (
    "No",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
)

_UTC_TIME = re.compile(r"(\d{4}-\d\d-\d\d)T(\d\d:\d\d:\d\d)Z")
_SEVERITY_ORDER = MappingProxyType(
    {Severity.ERROR: 0, Severity.WAIVED: 1, Severity.WARNING: 2}
)


def relative_href(from_path: str, to_path: str) -> str:
    """Link from the page ``from_path`` to the file ``to_path`` (both site keys)."""
    return posixpath.relpath(to_path, start=posixpath.dirname(from_path) or ".")


def repo_link_label(repo_url: str) -> str:
    """ "Source on GitHub" for github.com, "Source code" for any other host."""
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


def page_shell(*, title: str, body: Markup, path: str, repo_url: str | None) -> Markup:
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


# landing page


def _metric_label(name: str) -> str:
    return "ROC AUC" if name == "roc_auc" else name


def _finding_phrase(finding: Finding) -> Markup:
    name: Child = (
        el("em", finding.ghost, class_="ghost") if finding.ghost else finding.check
    )
    parts: list[Child] = [name]
    if finding.location:
        parts += [" at ", code(finding.location)]
    lines = suspect_lines(finding)
    if lines:
        shown: list[Child] = []
        for index, line in enumerate(lines):
            shown += [", " if index else "", code(line)]
        parts += [" (suspect lines ", shown, ")"]
    return el("span", parts)


def _joined(phrases: Sequence[Child]) -> list[Child]:
    """``a``, ``a and b`` or ``a, b and c``."""
    joined: list[Child] = []
    for index, phrase in enumerate(phrases):
        if index:
            joined.append(" and " if index == len(phrases) - 1 else ", ")
        joined.append(phrase)
    return joined


def hero_sentence(hero: Hero) -> Markup:
    """The paragraph under the haunted number, written from the rejected run."""
    rejected = hero.rejected
    score = score_text(rejected.metric.value)
    label = _metric_label(rejected.metric.name)
    parts: list[Child] = [
        f"{hero.app.name} scored {label} {score} on its first rejected run. "
    ]
    errors = [f for f in rejected.findings if f.severity is Severity.ERROR]
    if errors:
        phrases = [_finding_phrase(f) for f in errors]
        parts += ["The audit found ", _joined(phrases), ". "]
    else:
        parts.append("The audit rejected it. ")
    parts.append("Its honest score is" if hero.approved else "No approved run yet.")
    return el("p", parts, class_="lede")


def _hero_section(hero: Hero | None) -> Markup:
    motto = el("h1", MOTTO, id="motto")
    if hero is None:
        body: list[Child] = [el("p", "No run has been rejected yet."), motto]
    else:
        figure = score_text(hero.rejected.metric.value)
        body = [
            el("p", figure, class_="haunted", data_value=figure),
            hero_sentence(hero),
        ]
        if hero.approved is not None:
            body.append(
                el("p", score_text(hero.approved.metric.value), class_="honest")
            )
        body.append(motto)
    return el("section", body, class_="hero", aria_labelledby="motto")


def _case_item(summary: AppSummary) -> Markup:
    ghosts: Child = "No ghost found"
    if summary.ghosts:
        names: list[Child] = []
        for index, name in enumerate(summary.ghosts):
            names += [", " if index else "", el("em", name, class_="ghost")]
        ghosts = names
    href = relative_href(LANDING_PATH, f"{summary.slug}/index.html")
    return el(
        "li",
        el("h3", summary.name),
        el("p", summary.description) if summary.description else None,
        el("p", spaced(status_mark(summary.first.status), "on its first run")),
        el("p", ghosts),
        el(
            "p",
            f"from {score_text(summary.first.metric.value)} to "
            f"{score_text(summary.last.metric.value)}",
        ),
        el("p", spaced("Last run:", status_mark(summary.last.status))),
        el("p", _plural(len(summary.runs), "recorded run")),
        el("p", el("a", "Read the case", href=href)),
        class_="case",
    )


def _count_heading(count: int) -> str:
    word = _COUNT_WORDS[count] if count < len(_COUNT_WORDS) else str(count)
    return f"{word} case" if count == 1 else f"{word} cases"


def _cases_section(summaries: Sequence[AppSummary]) -> Markup:
    return el(
        "section",
        el("h2", _count_heading(len(summaries))),
        el("ul", [_case_item(s) for s in summaries], class_="cases"),
    )


def _steps_section() -> Markup:
    return el(
        "section",
        el("h2", "How the check works"),
        el("ol", [el("li", step) for step in CHECK_STEPS], class_="steps"),
    )


def _ghosts_section() -> Markup:
    entries: list[Child] = []
    for entry in CATALOG:
        if entry.ghost is None:
            continue
        entries.append(el("dt", el("em", entry.ghost, class_="ghost")))
        entries.append(
            el(
                "dd",
                el("p", prose(entry.meaning)),
                el("p", spaced("Why it matters:", prose(entry.why_it_matters))),
                el("p", spaced("How to fix it:", prose(entry.how_to_fix))),
            )
        )
    return el("section", el("h2", "The ghosts"), el("dl", entries))


def landing_page(
    summaries: Sequence[AppSummary],
    hero: Hero | None,
    benchmark: Benchmark | None,
    repo_url: str | None,
) -> Markup:
    """The whole page ``index.html``."""
    ghost_names = frozenset(e.ghost for e in CATALOG if e.ghost is not None)
    body = el(
        "div",
        _hero_section(hero),
        _cases_section(summaries),
        _steps_section(),
        benchmark_section(benchmark, ghost_names) if benchmark is not None else None,
        _ghosts_section(),
        el("section", el("h2", "How it was built"), el("p", HOW_IT_WAS_BUILT)),
    )
    return page_shell(
        title=LANDING_TITLE, body=body, path=LANDING_PATH, repo_url=repo_url
    )
