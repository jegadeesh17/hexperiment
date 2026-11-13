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

from hexperiment.gallery import art
from hexperiment.gallery.benchmark import benchmark_section
from hexperiment.gallery.difftext import diff_markup, diff_scripts
from hexperiment.gallery.markup import Child, Markup, code, el, prose, spaced
from hexperiment.gallery.modelcard import model_card_section
from hexperiment.gallery.records import AppSource, Benchmark
from hexperiment.gallery.summary import AppSummary, Hero, canary_counts, score_text
from hexperiment.ghosts import CATALOG
from hexperiment.models import Finding, Severity, Status, Verdict

LANDING_PATH = "index.html"
STYLESHEET_PATH = "assets/gallery.css"
LANDING_TITLE = "Hexperiment: audited model runs"
MOTTO = "If your model looks too good to be true, it\u2019s haunted."
GHOST_LINE = MappingProxyType(
    {
        "The Doppelganger": "The same rows sit in training and test.",
        "The Possession": (
            "One column gives the answer away, like a status filled in after "
            "the booking ends."
        ),
        "The Peeking Spirit": "Training saw the test rows. The canary test catches it.",
        "The Prophet": "Training uses rows from after the test period.",
        "The Shapeshifter": (
            "Identical runs give different results, so nothing can be verified."
        ),
    }
)
# Fixed copy, word for word from the reference page (mockups/landing.html).
_DEFINITION_HEAD = "Hexperiment is a Kiro power that catches data leaks in "
_DEFINITION_TAIL = (
    " test answers that sneak into training and fake a high score. "
    "Kiro\u2019s agent then fixes them."
)
_LEAK_TITLE = "A leak makes a broken model look perfect."
_LEAK_LEDE = (
    "A model learns from training rows, then gets graded on test rows it has "
    "never seen. A data leak lets information from the test rows slip into "
    "training. The model has seen the answers, so it aces the test and fails on "
    "real data.",
    "It\u2019s like a unit test that reads its expected value from the code it "
    "tests: it always passes and proves nothing. When an AI agent writes the "
    "training code, a leak is easy to miss. Nothing crashes. The score just "
    "goes up.",
)
_HOW_TITLE = "It scrambles the answers and watches what moves."
_HOW_LEDE = (
    "Hexperiment runs your training script twice. The second time, it scrambles "
    "only the test rows. Honest code never uses the test rows to train, so "
    "nothing in training should change. This is the canary test."
)
_HOW_STEPS = (
    ("Train normally", "Hexperiment records exactly what training produced."),
    (
        "Scramble the test rows, train again",
        "Same script, same seed. Only the test rows are shuffled.",
    ),
    ("Compare", "If any training row changed, the test rows leaked into training."),
)
_ROWS_KEYS = (
    "8 training rows, 2 test rows",
    "the test rows are scrambled",
    "orange rows changed: a leak",
)
_GHOSTS_LEDE = (
    "Each check names the leak it found, so a failed run reads like a story. A "
    "model is approved only when a full run passes every check and reaches its "
    "target score."
)
_CASES_LEDE = (
    "Every number here comes from a recorded run. Open a case file to see each "
    "run and every finding."
)
_CASES_NOTE = (
    "The honest score is lower. That is the point: it is an honest estimate of "
    "how the model will do on new data."
)
_KIRO_TITLE = "It lives inside Kiro."
_KIRO_LEDE = (
    "Hexperiment installs as a Kiro power. Kiro\u2019s agent gets the rules, "
    "the tools, and a loop that does not stop until the leak is gone."
)
_KIRO_ENTRIES = (
    (
        "Steering",
        "Before it writes training code, the agent reads the rules: split first, "
        "no columns from the future, always end with an audit.",
    ),
    (
        "Save hook",
        "Saving a training script runs a fast check and clears the old verdict, "
        "so an old approval stops counting once the code changes.",
    ),
    (
        "MCP tools",
        "The agent can list apps, run the audit, read the verdict and its "
        "history, and ask what a ghost means.",
    ),
    (
        "Guardrails",
        "The agent cannot edit the verdict, the model card, the thresholds or "
        "the data. It has to fix the cause.",
    ),
)
_LOOP_TITLE = "Then it fixes the leak and checks again."
_LOOP_COMMAND = "/workflow run exorcise"
_LOOP_LEDE = (
    " hands an app to the ghost-hunter agent and repeats until the full audit "
    "approves it."
)
_LOOP_STEPS = (
    (
        "Edit",
        "The ghost hunter changes the training code to remove the cause of each "
        "finding.",
    ),
    ("Audit", "The full audit runs again, canary test included."),
    (
        "Repeat",
        "Still rejected? It reads the new findings and tries again, up to five times.",
    ),
    (
        "Hand back",
        "Approved, or out of tries: it stops and shows you the latest verdict.",
    ),
)
_END_LEDE = "Open a case file to see every recorded run"
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


def _repo_link(repo_url: str | None, class_: str | None = None) -> Markup | None:
    if repo_url is None:
        return None
    return el("a", repo_link_label(repo_url), href=repo_url, class_=class_)


def page_shell(*, title: str, body: Child, path: str, repo_url: str | None) -> Markup:
    """The document around ``body`` for the page that will be stored at ``path``.

    The shell writes the head, the skip link and the sprite; each page writes its
    own ``header``, ``main`` and ``footer``. ``repo_url`` is kept for symmetry with
    the pages.
    """
    head = el(
        "head",
        el("meta", charset="utf-8"),
        el("meta", name="viewport", content="width=device-width, initial-scale=1"),
        el("title", title),
        el("link", rel="stylesheet", href=relative_href(path, STYLESHEET_PATH)),
    )
    page = el(
        "body",
        el("a", "Skip to content", href="#content", class_="skip"),
        art.SPRITE,
        body,
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
    name: Child = el("em", finding.ghost) if finding.ghost else finding.check
    what: list[Child] = [name]
    if finding.location:
        what += [" at ", code(finding.location)]
    what.append(el("span", finding.severity.value, class_="sev"))
    parts: list[Child] = [
        el("p", what, class_="what"),
        el("p", prose(finding.message), class_="msg"),
        el("p", spaced("Fix:", prose(finding.fix)), class_="fix"),
    ]
    if finding.severity is Severity.WAIVED:
        reason = finding.waiver_reason or ""
        parts.append(el("p", spaced("Waiver reason:", prose(reason)), class_="waiver"))
    return el("li", parts, class_=f"finding finding-{finding.severity.value}")


def _findings_list(verdict: Verdict) -> Markup:
    if not verdict.findings:
        return el("p", "No findings.", class_="clear")
    ordered = sorted(verdict.findings, key=lambda f: _SEVERITY_ORDER[f.severity])
    return el("ul", [_finding_item(f) for f in ordered], class_="findings")


def _run_item(verdict: Verdict) -> Markup:
    score: list[Child]
    if verdict.metric.value is None:
        score = [", no score"]
    else:
        score = [
            ", score ",
            el("span", score_text(verdict.metric.value), class_="score"),
        ]
    head = el(
        "p",
        status_mark(verdict.status),
        f", {verdict.mode.value} check",
        score,
        class_="run-head",
    )
    meta = el(
        "p",
        _time_element(verdict.created_at),
        f", test set used {_plural(verdict.test_set_uses, 'time')}",
        class_="run-meta",
    )
    return el("li", el("div", head, meta, _findings_list(verdict)), class_="run")


def _record(runs: Sequence[Verdict]) -> Markup:
    return el(
        "section",
        el(
            "div",
            el("h2", "Every recorded run", id="record-title"),
            el(
                "p",
                "Oldest first. An error stops a model; a warning is worth a look.",
                class_="lede",
            ),
            el("ol", [_run_item(run) for run in runs], class_="record"),
            class_="wrap",
        ),
        class_="page",
        aria_labelledby="record-title",
    )


def _fix_section(source: AppSource) -> Markup | None:
    if source.haunted_py is None or source.train_py is None:
        return None
    old_name, new_name = "haunted/train.py", "train.py"
    diff = diff_scripts(source.haunted_py, source.train_py)
    if not diff.blocks:
        body: list[Child] = [el("p", "The two scripts are identical.", class_="lede")]
    else:
        counts = (
            f": {_plural(diff.removed, 'line')} removed, "
            f"{_plural(diff.added, 'line')} added."
        )
        body = [
            el("p", code(old_name), " against ", code(new_name), counts, class_="lede"),
            diff_markup(diff, old_name=old_name, new_name=new_name),
        ]
    return el(
        "section",
        el("div", el("h2", "The fix", id="fix-title"), body, class_="wrap"),
        class_="page",
        aria_labelledby="fix-title",
    )


def _raw_files(summary: AppSummary, path: str) -> Markup:
    def link(name: str) -> Markup:
        return el("a", name, href=relative_href(path, f"{summary.slug}/{name}"))

    if summary.last_approved is None:
        links: list[Child] = [link("verdict.json"), "."]
    else:
        links = [link("verdict.json"), " and ", link("model_card.md"), "."]
    return el("p", "Raw files: ", links, class_="raw")


def _card_section(summary: AppSummary, path: str) -> Markup:
    approved = summary.last_approved
    content: Child
    if approved is None:
        content = el("p", "No approved run yet.")
    else:
        content = model_card_section(approved)
    return el(
        "section",
        el(
            "div",
            el("h2", "Model card of the last approved run", id="card-title"),
            content,
            _raw_files(summary, path),
            class_="wrap",
        ),
        class_="page",
        aria_labelledby="card-title",
    )


def _case_score(which: str, run: Verdict) -> Markup:
    return el(
        "div",
        el(
            "span",
            score_text(run.metric.value),
            class_=f"v {_score_class(run.status)}",
        ),
        el("span", f"{which} run, ", status_mark(run.status), class_="k"),
    )


def _facts(summary: AppSummary) -> Markup:
    facts: list[tuple[str, Child]] = [
        ("target", code(summary.target_column)),
        ("metric", summary.metric_name),
        ("threshold", score_text(summary.threshold)),
        ("recorded runs", str(len(summary.runs))),
    ]
    return el(
        "dl",
        [el("div", el("dt", term), el("dd", value)) for term, value in facts],
        class_="facts",
    )


def _case_head(summary: AppSummary, home: str, repo_url: str | None) -> Markup:
    return el(
        "section",
        el(
            "header",
            el(
                "div",
                el("a", "Hexperiment", href=home, class_="wordmark"),
                el(
                    "nav",
                    el("a", "All cases", href=f"{home}#cases"),
                    _repo_link(repo_url),
                    aria_label="Site",
                ),
                class_="wrap",
            ),
            class_="site-head",
        ),
        art.CORNER_MOON,
        el(
            "div",
            el("h1", summary.name, id="case-title"),
            el("p", summary.description, class_="desc")
            if summary.description
            else None,
            el(
                "div",
                _case_score("first", summary.first),
                el("span", class_="dash", aria_hidden="true"),
                _case_score("last", summary.last),
                class_="scores",
            ),
            _facts(summary),
            class_="wrap",
            id="content",
            tabindex="-1",
        ),
        class_="page case-head",
        aria_labelledby="case-title",
    )


def case_page(summary: AppSummary, source: AppSource, repo_url: str | None) -> Markup:
    """The whole page ``<slug>/index.html`` of one app."""
    path = f"{summary.slug}/index.html"
    home = relative_href(path, LANDING_PATH)
    page = [
        el(
            "main",
            _case_head(summary, home, repo_url),
            _record(summary.runs),
            _fix_section(source),
            _card_section(summary, path),
        ),
        el(
            "div",
            art.GRAVEYARD_CASE,
            el(
                "footer",
                el(
                    "div",
                    el("p", MOTTO, class_="motto"),
                    el(
                        "span",
                        el("a", "All cases", href=f"{home}#cases"),
                        ". Released under the MIT licence.",
                    ),
                    class_="wrap",
                ),
                class_="site-foot",
            ),
            class_="end",
        ),
    ]
    return page_shell(
        title=f"Hexperiment: {summary.name}", body=page, path=path, repo_url=repo_url
    )


# landing page


def _joined(phrases: Sequence[Child]) -> list[Child]:
    """``a``, ``a and b`` or ``a, b and c``."""
    joined: list[Child] = []
    for index, phrase in enumerate(phrases):
        if index:
            joined.append(" and " if index == len(phrases) - 1 else ", ")
        joined.append(phrase)
    return joined


def _count_word(count: int) -> str:
    return _COUNT_WORDS[count] if count < len(_COUNT_WORDS) else str(count)


def _screen(
    section_id: str,
    title_id: str,
    *content: Child,
    cls: str = "page",
    wrap: str = "wrap",
) -> Markup:
    """One screen: a ``section`` with its content in ``div.wrap``."""
    return el(
        "section",
        el("div", content, class_=wrap),
        class_=cls,
        id=section_id,
        aria_labelledby=title_id,
    )


def _case_href(hero: Hero | None) -> str:
    if hero is None:
        return "#cases"
    return relative_href(LANDING_PATH, f"{hero.app.slug}/index.html")


def _trail(hero: Hero | None) -> Markup:
    if hero is None:
        return el("p", "No run has been rejected yet.", class_="l")
    rejected = hero.rejected
    number = list(hero.app.runs).index(rejected) + 1
    when = "its first run" if number == 1 else f"run {number}"
    items: list[Markup] = [
        el(
            "li",
            el(
                "span",
                score_text(rejected.metric.value),
                art.GHOST_RISE,
                class_="n o",
            ),
            el("p", f"the score {hero.app.name} claimed on {when}", class_="l"),
        )
    ]
    counts = canary_counts(rejected)
    if counts is not None:
        changed, total = counts
        items.append(
            el(
                "li",
                el("span", f"{changed:,}", class_="n"),
                el(
                    "p",
                    f"of {total:,} training rows changed when we scrambled only "
                    "the test rows",
                    class_="l",
                ),
            )
        )
    if hero.approved is not None:
        items.append(
            el(
                "li",
                el("span", score_text(hero.approved.metric.value), class_="n g"),
                el(
                    "p",
                    "the honest score once the leaks were removed",
                    class_="l",
                ),
            )
        )
    else:
        items.append(el("li", el("p", "No approved run yet.", class_="l")))
    trail: list[Child] = []
    for index, item in enumerate(items):
        if index:
            trail.append(el("li", class_="dash", aria_hidden="true"))
        trail.append(item)
    return el("ol", trail, class_="trail")


def _hero_section(hero: Hero | None, repo_url: str | None) -> Markup:
    definition = el(
        "p",
        el(
            "strong",
            _DEFINITION_HEAD,
            el("span", "machine-learning", class_="nowrap"),
            " code:",
        ),
        _DEFINITION_TAIL,
        class_="def",
    )
    return el(
        "header",
        art.LANDSCAPE,
        el(
            "div",
            el(
                "div",
                el("a", "Hexperiment", href="#top", class_="wordmark"),
                el(
                    "nav",
                    el("a", "How it works", href="#how"),
                    _repo_link(repo_url),
                    aria_label="Site",
                ),
                class_="site-head",
            ),
            class_="wrap hero-nav",
        ),
        el(
            "div",
            el(
                "div",
                el("h1", "Is your model too good to be\u00a0true?", id="question"),
                definition,
                el("a", "Open the case file", href=_case_href(hero), class_="cta"),
                class_="hero-copy",
            ),
            class_="wrap hero-main",
            id="content",
            tabindex="-1",
        ),
        el("div", _trail(hero), class_="wrap hero-trail"),
        class_="page hero",
        id="top",
        aria_labelledby="question",
    )


def _problem_screen() -> Markup:
    return el(
        "section",
        el(
            "div",
            el(
                "div",
                el("h2", _LEAK_TITLE, id="leak-title"),
                [el("p", text, class_="lede") for text in _LEAK_LEDE],
            ),
            art.LEAK_FIGURE,
            class_="wrap two",
        ),
        class_="page",
        id="leak",
        aria_labelledby="leak-title",
    )


def _canary_screen(hero: Hero | None) -> Markup:
    third = _HOW_STEPS[2][1]
    counts = canary_counts(hero.rejected) if hero is not None else None
    if hero is not None and counts is not None:
        third += f" In {hero.app.name}, {counts[0]:,} of {counts[1]:,} changed."
    texts = (_HOW_STEPS[0][1], _HOW_STEPS[1][1], third)
    rows = (art.ROWS_TRAIN, art.ROWS_SCRAMBLED, art.ROWS_CHANGED)
    steps = [
        el(
            "li",
            el("h3", _HOW_STEPS[index][0]),
            el("p", texts[index]),
            rows[index],
            el("p", _ROWS_KEYS[index], class_="rows-key"),
        )
        for index in range(3)
    ]
    return _screen(
        "how",
        "how-title",
        el("h2", _HOW_TITLE, id="how-title"),
        el("p", _HOW_LEDE, class_="lede"),
        el("ol", steps, class_="steps"),
    )


def _ghosts_screen() -> Markup:
    names = [e.ghost for e in CATALOG if e.ghost is not None]
    entries = [
        el("div", el("dt", name), el("dd", GHOST_LINE[name]), class_="ghost-entry")
        for name in names
    ]
    return el(
        "section",
        el(
            "div",
            el("h2", f"{_count_word(len(names))} ghosts it hunts.", id="ghosts-title"),
            el("p", _GHOSTS_LEDE, class_="lede"),
            el("dl", entries, class_="ghosts"),
            class_="wrap",
        ),
        art.PEEK,
        class_="page",
        id="ghosts",
        aria_labelledby="ghosts-title",
    )


def _was_fixed(summary: AppSummary) -> bool:
    """A rejected run is followed later in the history by an approved run."""
    statuses = [run.status for run in summary.runs]
    return any(
        status is Status.REJECTED and Status.APPROVED in statuses[index + 1 :]
        for index, status in enumerate(statuses)
    )


def _score_class(status: Status) -> str:
    return {Status.REJECTED: "o", Status.APPROVED: "g"}.get(status, "u")


def _score_block(which: str, run: Verdict) -> Markup:
    return el(
        "div",
        el(
            "span",
            score_text(run.metric.value),
            class_=f"v {_score_class(run.status)}",
        ),
        el("span", f"{which} run, {run.status.value.lower()}", class_="k"),
    )


def _found(summary: AppSummary) -> Markup:
    runs = f"{_plural(len(summary.runs), 'recorded run')}."
    if not summary.ghosts:
        return el("p", f"No ghost found. {runs}", class_="found")
    names = [el("em", name) for name in summary.ghosts]
    return el("p", "Found ", _joined(names), f". {runs}", class_="found")


def _case_item(summary: AppSummary) -> Markup:
    href = relative_href(LANDING_PATH, f"{summary.slug}/index.html")
    return el(
        "li",
        el(
            "div",
            el("h3", summary.name),
            el("p", summary.description, class_="desc")
            if summary.description
            else None,
            _found(summary),
        ),
        el(
            "div",
            _score_block("first", summary.first),
            el("span", "\u2192", class_="to", aria_hidden="true"),
            _score_block("last", summary.last),
            class_="pair",
        ),
        el(
            "a",
            "Open the case file",
            href=href,
            aria_label=f"Open the {summary.name} case file",
        ),
        class_="case",
    )


def _shown_score(run: Verdict | None) -> float | None:
    if run is None or run.metric.value is None:
        return None
    return float(score_text(run.metric.value))


def _honest_is_lower(summary: AppSummary) -> bool:
    first = _shown_score(summary.first_rejected)
    last = _shown_score(summary.last_approved)
    return first is not None and last is not None and last < first


def _cases_screen(summaries: Sequence[AppSummary]) -> Markup:
    count = len(summaries)
    if all(_was_fixed(s) for s in summaries):
        title = (
            "One haunted app, caught and fixed."
            if count == 1
            else f"{_count_word(count)} haunted apps, caught and fixed."
        )
    else:
        title = "One case." if count == 1 else f"{_count_word(count)} cases."
    note = (
        el("p", _CASES_NOTE, class_="note")
        if all(_honest_is_lower(s) for s in summaries)
        else None
    )
    return _screen(
        "cases",
        "cases-title",
        el("h2", title, id="cases-title"),
        el("p", _CASES_LEDE, class_="lede"),
        el("ul", [_case_item(s) for s in summaries], class_="cases"),
        note,
    )


def _kiro_screen() -> Markup:
    entries = [
        el("div", el("dt", term), el("dd", text)) for term, text in _KIRO_ENTRIES
    ]
    return _screen(
        "kiro",
        "kiro-title",
        el("h2", _KIRO_TITLE, id="kiro-title"),
        el("p", _KIRO_LEDE, class_="lede"),
        el("dl", entries, class_="kiro"),
    )


def _loop_screen() -> Markup:
    steps = [el("li", el("h3", head), el("p", text)) for head, text in _LOOP_STEPS]
    return _screen(
        "loop",
        "loop-title",
        el("h2", _LOOP_TITLE, id="loop-title"),
        el("p", code(_LOOP_COMMAND), _LOOP_LEDE, class_="lede"),
        el("ol", steps, class_="steps loop-steps"),
    )


def _end_screen(hero: Hero | None, repo_url: str | None) -> Markup:
    lede = f"{_END_LEDE}, or read the source." if repo_url else f"{_END_LEDE}."
    links = el(
        "div",
        el("a", "Open the case file", href=_case_href(hero), class_="cta"),
        _repo_link(repo_url),
        class_="links",
    )
    return el(
        "section",
        el(
            "div",
            el("h2", MOTTO, id="end-title"),
            el("p", lede, class_="lede"),
            links,
            class_="wrap",
        ),
        art.GRAVEYARD_LANDING,
        el(
            "footer",
            el("div", "Released under the MIT licence.", class_="wrap"),
            class_="site-foot",
        ),
        class_="page end",
        id="end",
        aria_labelledby="end-title",
    )


def landing_page(
    summaries: Sequence[AppSummary],
    hero: Hero | None,
    benchmark: Benchmark | None,
    repo_url: str | None,
) -> Markup:
    """The whole page ``index.html``."""
    ghost_names = frozenset(e.ghost for e in CATALOG if e.ghost is not None)
    benchmark_screen = (
        _screen(
            "benchmark-page",
            "benchmark",
            benchmark_section(benchmark, ghost_names),
        )
        if benchmark is not None
        else None
    )
    body = [
        _hero_section(hero, repo_url),
        el(
            "main",
            _problem_screen(),
            _canary_screen(hero),
            _ghosts_screen(),
            _cases_screen(summaries),
            benchmark_screen,
            _kiro_screen(),
            _loop_screen(),
            _end_screen(hero, repo_url),
        ),
    ]
    return page_shell(
        title=LANDING_TITLE, body=body, path=LANDING_PATH, repo_url=repo_url
    )
