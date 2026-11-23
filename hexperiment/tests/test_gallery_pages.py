import dataclasses
import re
from html import unescape
from html.parser import HTMLParser

import pytest

from gallery_builder import (
    approved,
    make_finding,
    make_source,
    make_verdict,
    rejected,
)
from hexperiment.gallery.markup import ART_TAGS, TAGS, el
from hexperiment.gallery.pages import (
    GHOST_LINE,
    LANDING_TITLE,
    MOTTO,
    case_page,
    landing_page,
    page_shell,
    relative_href,
    repo_link_label,
    status_mark,
)
from hexperiment.gallery.records import Benchmark, BenchmarkMechanism
from hexperiment.gallery.summary import choose_hero, slugify, summarize_app
from hexperiment.ghosts import CATALOG
from hexperiment.models import Severity, Status

REPO = "https://github.com/example/example"


class Page(HTMLParser):
    """Collects start tags with attributes and the text of a page."""

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.starts: list[tuple[str, dict[str, str | None]]] = []
        self.chunks: list[str] = []
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.starts.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.chunks.append(data)

    @property
    def text(self) -> str:
        return "".join(self.chunks)

    def tags(self, name):
        return [attrs for tag, attrs in self.starts if tag == name]


def build(source=None, *, repo_url=None, slug="demo"):
    source = source or make_source()
    summary = summarize_app(slug, source)
    return case_page(summary, source, repo_url).html


def squeeze(html: str) -> str:
    """The text of the page with tags removed and white space folded."""
    return " ".join(Page(html).text.split())


# shell


def test_shell_head_and_landmarks():
    html = build()
    page = Page(html)
    assert '<html lang="en">' in html
    assert page.starts[1] == ("head", {})
    assert page.starts[2] == ("meta", {"charset": "utf-8"})
    assert page.tags("meta")[1]["name"] == "viewport"
    assert "<title>Hexperiment: demo</title>" in html
    assert page.tags("main") == [{}]
    assert len(page.tags("header")) == len(page.tags("footer")) == 1
    content = [d for d in page.tags("div") if d.get("id") == "content"]
    assert content == [{"class": "wrap", "id": "content", "tabindex": "-1"}]
    assert [a for a in page.tags("h1")] == [{"id": "case-title"}]


def test_the_only_stylesheet_link_is_relative_and_nothing_names_another_origin():
    html = build()
    hrefs = [a["href"] for a in Page(html).tags("link")]
    assert hrefs == ["../assets/gallery.css"]
    assert "preconnect" not in html
    assert "googleapis" not in html
    assert "http" not in html


def test_skip_link_comes_first_in_the_body():
    page = Page(build())
    names = [tag for tag, _ in page.starts]
    after_body = page.starts[names.index("body") + 1]
    assert after_body == ("a", {"class": "skip", "href": "#content"})


def test_the_sprite_follows_the_skip_link():
    html = build()
    assert html.index('class="skip"') < html.index('<svg class="sprite"')
    assert html.count('<symbol id="ghost"') == 1


def test_titles_differ_between_apps():
    a = build(make_source("alpha"), slug="alpha")
    b = build(make_source("beta"), slug="beta")
    assert "<title>Hexperiment: alpha</title>" in a
    assert "<title>Hexperiment: beta</title>" in b


def test_repo_link_in_the_navigation_for_github():
    html = build(repo_url=REPO)
    links = [a for a in Page(html).tags("a") if a.get("href") == REPO]
    assert len(links) == 1
    assert html.count("Source on GitHub") == 1
    assert f'<a href="../index.html#cases">All cases</a><a href="{REPO}">' in html


def test_repo_link_label_for_another_host():
    html = build(repo_url="https://example.org/x")
    assert html.count("Source code") == 1
    assert "Source on GitHub" not in html


def test_no_repo_link_without_the_option():
    html = build()
    assert "Source on GitHub" not in html
    assert "Source code" not in html


def test_repo_link_label_function():
    assert repo_link_label(REPO) == "Source on GitHub"
    assert repo_link_label("https://gitlab.com/a/b") == "Source code"
    assert repo_link_label("https://github.com.evil.org/a") == "Source code"


def test_shell_of_the_landing_path_uses_plain_stylesheet_path():
    html = page_shell(
        title="t", body=el("p", "x"), path="index.html", repo_url=None
    ).html
    assert 'href="assets/gallery.css"' in html


@pytest.mark.parametrize(
    ("src", "dst", "expected"),
    [
        ("index.html", "assets/gallery.css", "assets/gallery.css"),
        ("demo/index.html", "assets/gallery.css", "../assets/gallery.css"),
        ("demo/index.html", "index.html", "../index.html"),
        ("demo/index.html", "demo/verdict.json", "verdict.json"),
        ("index.html", "demo/index.html", "demo/index.html"),
    ],
)
def test_relative_href(src, dst, expected):
    assert relative_href(src, dst) == expected


# status mark


@pytest.mark.parametrize("status", ["REJECTED", "APPROVED", "UNVERIFIED"])
def test_status_mark(status):
    html = status_mark(status).html
    assert html == (
        f'<span class="status status-{status.lower()}">'
        f'<span class="mark" aria-hidden="true"></span>{status}</span>'
    )


# the record


def runs_source():
    return make_source(
        runs=[
            rejected(1.0),
            make_verdict(
                Status.UNVERIFIED,
                "fast",
                0.909,
                run_id="20261111-195053-000003",
                created_at="2026-11-12T08:01:02Z",
                test_set_uses=2,
            ),
            approved(0.9096),
        ]
    )


def compact(html: str) -> str:
    """The markup with the white space between tags removed."""
    return re.sub(r">\s+<", "><", html)


def test_every_field_of_a_timeline_entry():
    html = compact(build(runs_source()))
    page = Page(html)
    items = [a for a in page.tags("li") if a.get("class") == "run"]
    assert len(items) == 3
    assert '<ol class="record">' in html
    assert (
        '<p class="run-head"><span class="status status-rejected">'
        '<span class="mark" aria-hidden="true"></span>REJECTED</span>'
        ', full check, score <span class="score">1.0000</span></p>'
        '<p class="run-meta"><time datetime="2026-11-11T19:50:51Z">'
        "2026-11-11 19:50:51 UTC</time>, test set used 1 time</p>"
    ) in html
    assert (
        ', fast check, score <span class="score">0.9090</span></p>'
        '<p class="run-meta"><time datetime="2026-11-12T08:01:02Z">'
        "2026-11-12 08:01:02 UTC</time>, test set used 2 times</p>"
    ) in html
    assert html.count('<div><p class="run-head">') == 3
    text = squeeze(html)
    record = text.split("Every recorded run")[1]
    assert record.index("REJECTED") < record.index("UNVERIFIED")
    assert record.index("UNVERIFIED") < record.index("APPROVED")


def test_a_note_is_shown_under_its_run_and_is_escaped():
    source = make_source(
        runs=[rejected(1.0), approved(0.9096)],
        notes=["Made by <b>hand</b> & \"quotes\" of the ghost hunter's", None],
    )
    html = compact(build(source))
    assert (
        "test set used 1 time</p>"
        '<p class="run-note">Made by &lt;b&gt;hand&lt;/b&gt; &amp; &quot;quotes&quot; '
        "of the ghost hunter&#x27;s</p>"
    ) in html
    assert "<b>hand</b>" not in html
    assert html.count('class="run-note"') == 1


def test_a_run_without_a_note_shows_none():
    html = build(runs_source())
    assert "run-note" not in html


def test_the_record_section_and_its_lede():
    html = compact(build(runs_source()))
    assert (
        '<section class="page" aria-labelledby="record-title"><div class="wrap">'
        '<h2 id="record-title">Every recorded run</h2>'
        '<p class="lede">Oldest first. An error stops a model; a warning is worth '
        "a look.</p>"
    ) in html


def test_finding_shows_ghost_severity_location_message_and_fix():
    html = compact(build(runs_source()))
    assert (
        '<li class="finding finding-error"><p class="what"><em>The Possession</em>'
        ' at <code>reservation_status</code><span class="sev">error</span></p>'
        '<p class="msg">reservation_status alone predicts the target with AUC '
        "1.0000</p>"
        '<p class="fix">Fix: Check whether this column is recorded after the '
        "outcome."
    ) in html


def test_finding_without_ghost_shows_check_name():
    finding = make_finding(
        check="split_order",
        ghost=None,
        severity=Severity.WARNING,
        location="train.py:3",
    )
    source = make_source(runs=[make_verdict(findings=[finding])])
    html = compact(build(source))
    assert (
        '<p class="what">split_order at <code>train.py:3</code>'
        '<span class="sev">warning</span></p>'
    ) in html
    assert 'class="finding finding-warning"' in html


def test_waived_finding_shows_reason():
    finding = make_finding(severity=Severity.WAIVED, waiver_reason="known at booking")
    html = compact(build(make_source(runs=[make_verdict(findings=[finding])])))
    assert '<span class="sev">waived</span>' in html
    assert '<p class="waiver">Waiver reason: known at booking</p>' in html
    assert 'class="finding finding-waived"' in html


def test_only_waived_findings_have_a_waiver_paragraph():
    html = build(runs_source())
    assert 'class="waiver"' not in html


def test_findings_are_listed_errors_then_waived_then_warnings():
    warning = make_finding(
        check="metric", ghost=None, severity=Severity.WARNING, location="w-loc"
    )
    waived = make_finding(severity=Severity.WAIVED, location="v-loc", waiver_reason="r")
    error = make_finding(location="e-loc")
    source = make_source(runs=[make_verdict(findings=[warning, waived, error])])
    text = squeeze(build(source))
    assert text.index("e-loc") < text.index("v-loc") < text.index("w-loc")


def test_a_run_without_findings():
    assert '<p class="clear">No findings.</p>' in build()
    assert 'class="findings"' not in build()


def test_a_run_without_score():
    source = make_source(runs=[make_verdict(Status.REJECTED, value=None)])
    html = compact(build(source))
    assert "REJECTED</span>, full check, no score</p>" in html
    assert 'class="score"' not in html


def test_time_that_is_not_utc_iso_is_shown_as_written():
    source = make_source(runs=[make_verdict(created_at="yesterday")])
    html = build(source)
    assert "<time>yesterday</time>" in html


def test_fact_line_and_heading():
    source = make_source(
        "hotel", description="Will it be cancelled?", target="is_canceled"
    )
    html = compact(build(source, slug="hotel"))
    assert '<h1 id="case-title">hotel</h1>' in html
    assert '<p class="desc">Will it be cancelled?</p>' in html
    assert (
        '<dl class="facts">'
        "<div><dt>target</dt><dd><code>is_canceled</code></dd></div>"
        "<div><dt>metric</dt><dd>roc_auc</dd></div>"
        "<div><dt>threshold</dt><dd>0.8500</dd></div>"
        "<div><dt>recorded runs</dt><dd>1</dd></div></dl>"
    ) in html
    assert (
        '<a href="../index.html" class="wordmark">Hexperiment</a>'
        '<nav aria-label="Site"><a href="../index.html#cases">All cases</a></nav>'
    ) in html


def test_the_head_section_and_its_scores():
    source = make_source(runs=[rejected(1.0), approved(0.9096)])
    html = compact(build(source))
    assert (
        '<section class="page case-head" aria-labelledby="case-title">'
        '<header class="site-head"><div class="wrap">'
    ) in html
    assert html.index('class="corner"') < html.index('id="content"')
    assert (
        '<div class="scores">'
        '<div><span class="v o">1.0000</span><span class="k">first run, '
        '<span class="status status-rejected"><span class="mark" '
        'aria-hidden="true"></span>REJECTED</span></span></div>'
        '<span class="dash" aria-hidden="true"></span>'
        '<div><span class="v g">0.9096</span><span class="k">last run, '
        '<span class="status status-approved"><span class="mark" '
        'aria-hidden="true"></span>APPROVED</span></span></div></div>'
    ) in html


@pytest.mark.parametrize(
    ("run", "cls", "shown"),
    [
        (rejected(0.9), "o", "0.9000"),
        (approved(0.9), "g", "0.9000"),
        (make_verdict(Status.UNVERIFIED, "fast", None), "u", "no score"),
    ],
)
def test_the_score_classes_of_the_head(run, cls, shown):
    html = build(make_source(runs=[run]))
    assert f'<span class="v {cls}">{shown}</span>' in html


def test_empty_description_is_left_out():
    html = build(make_source(description=""))
    assert "<p></p>" not in html
    assert 'class="desc"' not in html


def test_the_case_page_has_one_main_without_an_id_and_the_end_block():
    html = compact(build())
    assert html.count("<main>") == 1
    assert html.index("</main>") < html.index('<div class="end">')
    assert html.index('<div class="end"><svg class="graveyard"') > 0
    assert (
        '<footer class="site-foot"><div class="wrap">'
        f'<p class="motto">{MOTTO}</p>'
        '<span><a href="../index.html#cases">All cases</a>. '
        "Released under the MIT licence.</span></div></footer></div>"
    ) in html


# fix, card, raw files


def test_no_fix_section_without_haunted_script():
    assert "The fix" not in build()


def test_fix_section_with_diff():
    source = make_source(haunted="a\nscale()\nb\n", train="a\nb\n")
    html = compact(build(source))
    text = squeeze(html)
    assert (
        '<section class="page" aria-labelledby="fix-title"><div class="wrap">'
        '<h2 id="fix-title">The fix</h2><p class="lede">'
    ) in html
    assert (
        "haunted/train.py against train.py (the honest script kept in the "
        "repository): 1 line removed, 0 lines added."
    ) in text
    assert 'class="diff"' in html
    assert 'class="line line-removed"' in html


def test_identical_scripts_sentence():
    source = make_source(haunted="a\n", train="a\n")
    html = build(source)
    assert '<p class="lede">The two scripts are identical.</p>' in html
    assert 'class="diff"' not in html


def test_no_approved_run_yet():
    source = make_source(runs=[rejected(1.0)])
    html = compact(build(source))
    assert "<p>No approved run yet.</p>" in html
    assert "model_card.md" not in html
    assert '<p class="raw">Raw files: <a href="verdict.json">verdict.json</a>.</p>' in (
        html
    )


def test_model_card_of_the_last_approved_run_and_raw_links():
    source = make_source(runs=[approved(0.91), rejected(0.99), approved(0.9096)])
    html = compact(build(source))
    assert (
        '<section class="page" aria-labelledby="card-title"><div class="wrap">'
        '<h2 id="card-title">Model card of the last approved run</h2>'
        '<div class="card"><div class="card-group"><h3>Status</h3>'
    ) in html
    assert "0.9096" in html
    assert (
        '<p class="raw">Raw files: <a href="verdict.json">verdict.json</a> and '
        '<a href="model_card.md">model_card.md</a>.</p></div></section></main>'
    ) in html
    assert html.count('href="model_card.md"') == 1
    assert "Raw files</h2>" not in html


# hostile strings

HOSTILE = [
    "<script>alert(1)</script>",
    '"><img src=x onerror=alert(1)>',
    "`<b>`",
    "&amp;",
    "\x00",
    "\ud800",
]


def hostile_page():
    findings = [
        make_finding(
            location=text,
            message=text,
            fix=text,
            check=text,
            severity=severity,
            waiver_reason=text,
            ghost=None,
        )
        for text in HOSTILE
        for severity in (Severity.ERROR, Severity.WAIVED)
    ]
    source = make_source(
        runs=[make_verdict(findings=findings)],
        name="<script>alert(1)</script>",
        description='"><img src=x onerror=alert(1)>',
        target="&amp;",
        haunted="\n".join(HOSTILE) + "\nx = 1\n",
        train="\n".join(HOSTILE[::-1]) + "\nx = 2\n",
    )
    return build(source, repo_url=REPO)


def test_hostile_strings_cannot_create_tags_or_attributes():
    page = Page(hostile_page())
    assert page.starts
    for tag, attrs in page.starts:
        assert tag in TAGS | ART_TAGS
        assert tag not in {"script", "img", "style"}
        for name in attrs:
            assert not name.startswith("on")
            assert name != "style"


def test_hostile_strings_appear_as_text():
    text = Page(hostile_page()).text
    assert "<script>alert(1)</script>" in text
    assert '"><img src=x onerror=alert(1)>' in text
    assert "<b>" in text
    assert "&amp;" in text
    assert "\ufffd" in text
    assert "\x00" not in text
    assert not re.search(r"[\ud800-\udfff]", text)


def test_hostile_page_encodes_as_utf8():
    hostile_page().encode("utf-8")
    assert "<script>" not in hostile_page()
    assert unescape("&lt;script&gt;") == "<script>"


# landing page

CANARY_FIX = "Fit scalers on training rows only. Suspect lines: train.py:100."


def canary_finding(rows_changed=95309, train_rows=95457):
    finding = make_finding(
        check="canary",
        ghost="The Peeking Spirit",
        location="train.py",
        message="Scrambling the test rows changed training.",
        fix=CANARY_FIX,
    )
    evidence = {}
    if rows_changed is not None:
        evidence["rows_changed"] = rows_changed
    if train_rows is not None:
        evidence["train_rows"] = train_rows
    return dataclasses.replace(finding, evidence=evidence)


def hotel_source(**canary):
    first = make_verdict(
        Status.REJECTED,
        value=1.0,
        findings=[
            make_finding(location="reservation_status"),
            canary_finding(**canary),
        ],
        run_id="20261111-195051-000001",
    )
    return make_source(
        "hotel-cancellations",
        runs=[
            first,
            make_verdict(
                Status.UNVERIFIED, "fast", 0.909, run_id="20261111-195052-000002"
            ),
            approved(0.909),
            approved(0.9096),
        ],
        description="Predict whether a hotel booking will be cancelled.",
        target="is_canceled",
    )


def flight_source():
    return make_source(
        "flight-delays",
        runs=[rejected(0.7971, ghost="The Prophet"), approved(0.6193)],
        description="Predict flight delays.",
    )


def landing(*sources, benchmark=None, repo_url=None):
    sources = sources or (hotel_source(), flight_source())
    summaries = [summarize_app(slugify(s.config.app.name), s) for s in sources]
    hero = choose_hero(summaries)
    return landing_page(summaries, hero, benchmark, repo_url).html


def sample_benchmark():
    mechanism = BenchmarkMechanism(
        leak_type="L1.1",
        leak_type_name="No test set",
        mechanism="The model is scored on training rows.",
        designed_to_catch=("The Peeking Spirit",),
        scripts=12,
        rejected=9,
        by_ghost=(("The Peeking Spirit", 9),),
    )
    return Benchmark(
        created_at="2026-11-11T19:50:51Z",
        day="2026-11-11",
        hexperiment_version="0.1.0",
        python_version="3.13.1",
        scikit_learn_version="1.6.0",
        source="a taxonomy",
        mechanisms=(mechanism,),
        clean_scripts=12,
        false_alarms=0,
        not_measured=(),
    )


def trail_items(html):
    found = re.search(r'<ol class="trail">(.*?)</ol>', html, re.S)
    assert found
    return [
        " ".join(" ".join(Page(item).chunks).split())
        for item in re.findall(r"<li>(.*?)</li>", found.group(1), re.S)
    ]


def test_landing_is_the_whole_document_with_the_shell():
    html = landing()
    page = Page(html)
    assert html.startswith('<html lang="en">')
    assert f"<title>{LANDING_TITLE}</title>" in html
    assert [a["href"] for a in page.tags("link")] == ["assets/gallery.css"]
    assert "googleapis" not in html
    assert page.tags("main") == [{}]
    assert page.tags("footer") == [{"class": "site-foot"}]
    names = [tag for tag, _ in page.starts]
    after_body = page.starts[names.index("body") + 1]
    assert after_body == ("a", {"class": "skip", "href": "#content"})


def test_screens_in_order_with_their_ids():
    html = landing(benchmark=sample_benchmark())
    ids = re.findall(r'<(?:header|section) class="page[^"]*" id="([^"]+)"', html)
    assert ids == [
        "top",
        "leak",
        "how",
        "ghosts",
        "cases",
        "benchmark-page",
        "kiro",
        "loop",
        "end",
    ]
    assert html.index("<main>") > html.index('id="top"')
    for title in ("leak", "how", "ghosts", "cases", "kiro", "loop", "end"):
        assert f'aria-labelledby="{title}-title"' in html
        assert f'id="{title}-title"' in html
    assert 'aria-labelledby="question"' in html
    assert 'aria-labelledby="benchmark"' in html


def test_one_h1_with_the_question_and_a_no_break_space():
    html = landing()
    assert re.findall(r"<h1[^>]*>(.*?)</h1>", html) == [
        "Is your model too good to be\u00a0true?"
    ]
    assert '<h1 id="question">' in html


def test_the_skip_link_target_holds_the_h1():
    html = landing()
    target = '<div class="wrap hero-main" id="content" tabindex="-1">'
    assert target in html
    assert html.index(target) < html.index("<h1") < html.index("hero-trail")
    assert html.count('id="content"') == 1


def test_the_definition():
    text = squeeze(landing())
    assert (
        "Hexperiment is a Kiro power that catches data leaks in machine-learning "
        "code: test answers that sneak into training and fake a high score. "
        "Kiro\u2019s agent then fixes them."
    ) in text
    assert (
        "<strong>Hexperiment is a Kiro power that catches data leaks in "
        '<span class="nowrap">machine-learning</span> code:</strong>'
    ) in landing()


def test_the_motto_uses_a_typographic_apostrophe():
    assert MOTTO == "If your model looks too good to be true, it\u2019s haunted."
    html = landing()
    assert f'<h2 id="end-title">{MOTTO}</h2>' in html
    assert "'" not in squeeze(html).replace("&#x27;", "")


def test_call_to_action_goes_to_the_hero_case_and_to_cases_without_a_hero():
    html = landing()
    ctas = [a for a in Page(html).tags("a") if a.get("class") == "cta"]
    assert [a["href"] for a in ctas] == ["hotel-cancellations/index.html"] * 2
    html = landing(make_source("x", runs=[approved(0.9)]))
    ctas = [a for a in Page(html).tags("a") if a.get("class") == "cta"]
    assert [a["href"] for a in ctas] == ["#cases"] * 2


def test_trail_with_three_numbers():
    html = landing()
    assert trail_items(html) == [
        "1.0000 the score hotel-cancellations claimed on its first run",
        "95,309 of 95,457 training rows changed when we scrambled only the test rows",
        "0.9096 the honest score once the leaks were removed",
    ]
    assert '<span class="n o">1.0000<span class="ghost-wrap"' in html
    assert '<span class="n">95,309</span>' in html
    assert '<span class="n g">0.9096</span>' in html
    assert html.count('<li class="dash" aria-hidden="true"></li>') == 2


def test_trail_without_canary_counts_has_two_numbers():
    html = landing(hotel_source(rows_changed=None))
    assert trail_items(html) == [
        "1.0000 the score hotel-cancellations claimed on its first run",
        "0.9096 the honest score once the leaks were removed",
    ]
    assert html.count('<li class="dash"') == 1


def test_trail_without_an_approved_run():
    html = landing(make_source("x", runs=[rejected(0.99)]))
    assert trail_items(html)[-1] == "No approved run yet."
    assert 'class="n g"' not in html


def test_trail_names_the_run_when_the_first_rejection_is_not_the_first_run():
    html = landing(make_source("x", runs=[approved(0.8), rejected(0.99)]))
    assert trail_items(html)[0] == "0.9900 the score x claimed on run 2"


def test_trail_without_a_hero():
    html = landing(make_source("x", runs=[approved(0.9)]))
    assert '<p class="l">No run has been rejected yet.</p>' in html
    assert '<ol class="trail">' not in html


def test_hero_uses_the_app_with_the_highest_first_rejected_score():
    html = landing(flight_source(), hotel_source())
    assert trail_items(html)[0].endswith("hotel-cancellations claimed on its first run")


def test_thousands_separators_in_the_trail_and_step_three():
    html = landing(hotel_source(rows_changed=1234567, train_rows=2345678))
    assert "1,234,567" in trail_items(html)[1]
    assert "1,234,567 of 2,345,678 changed." in squeeze(html)


def test_step_three_with_and_without_counts():
    text = squeeze(landing())
    assert (
        "If any training row changed, the test rows leaked into training. "
        "In hotel-cancellations, 95,309 of 95,457 changed."
    ) in text
    text = squeeze(landing(hotel_source(train_rows=None)))
    assert "leaked into training. Compare" not in text
    assert "leaked into training. In " not in text
    assert "If any training row changed, the test rows leaked into training." in text


def test_the_canary_steps_have_their_rows_and_keys():
    html = landing()
    assert html.count('<div class="rows" aria-hidden="true">') == 3
    text = squeeze(html)
    for key in (
        "8 training rows, 2 test rows",
        "the test rows are scrambled",
        "orange rows changed: a leak",
    ):
        assert key in text
    assert text.count("Train normally") == 1


def test_one_ghost_entry_per_catalogue_ghost_with_its_line():
    html = landing()
    names = [entry.ghost for entry in CATALOG if entry.ghost is not None]
    assert set(GHOST_LINE) == set(names)
    assert html.count('<div class="ghost-entry">') == len(names) == 5
    for name in names:
        assert f"<dt>{name}</dt>" in html
        assert f"<dd>{GHOST_LINE[name]}</dd>" in html
    assert '<h2 id="ghosts-title">Five ghosts it hunts.</h2>' in html
    assert html.index('<h2 id="ghosts-title">') < html.index('class="peek"')


def test_cases_are_an_unnumbered_list_and_the_page_has_several_ols():
    html = landing()
    assert '<ul class="cases">' in html
    assert len([a for a in Page(html).tags("li") if a.get("class") == "case"]) == 2
    assert "cases-title" in html


def test_cases_heading_forms():
    assert "Two haunted apps, caught and fixed." in landing()
    one = landing(hotel_source())
    assert "One haunted app, caught and fixed." in one
    mixed = landing(hotel_source(), make_source("x", runs=[approved(0.9)]))
    assert '<h2 id="cases-title">Two cases.</h2>' in mixed
    assert "One case." in landing(make_source("x", runs=[approved(0.9)]))
    ten = [make_source(f"app{n}", runs=[approved(0.9)]) for n in range(10)]
    assert "10 cases." in landing(*ten)


def test_an_approval_before_the_rejection_is_not_a_fix():
    source = make_source("x", runs=[approved(0.9), rejected(0.99)])
    assert "One case." in landing(source)


def test_case_found_text_ghosts_none_and_one_run():
    text = squeeze(landing())
    assert "Found The Possession and The Peeking Spirit. 4 recorded runs." in text
    assert "Found The Prophet. 2 recorded runs." in text
    html = landing(
        make_source("x", runs=[approved(0.9)]),
        make_source("y", runs=[approved(0.9)]),
    )
    assert "No ghost found. 1 recorded run." in squeeze(html)
    assert "1 recorded runs" not in squeeze(html)
    assert "<em>The Possession</em>" in landing()


def test_three_found_ghosts_are_joined_with_commas_and_and():
    first = make_verdict(
        Status.REJECTED,
        value=0.99,
        findings=[
            make_finding(ghost="The Doppelganger"),
            make_finding(ghost="The Possession"),
            make_finding(ghost="The Prophet"),
        ],
    )
    html = landing(make_source("x", runs=[first, approved(0.8)]))
    assert (
        "Found <em>The Doppelganger</em>, <em>The Possession</em> and "
        "<em>The Prophet</em>. 2 recorded runs." in html
    )


def test_score_pairs_and_their_classes():
    html = landing()
    assert (
        '<span class="v o">1.0000</span><span class="k">first run, rejected</span>'
        in html
    )
    assert (
        '<span class="v g">0.9096</span><span class="k">last run, approved</span>'
        in html
    )
    assert '<span class="to" aria-hidden="true">\u2192</span>' in html
    source = make_source(
        "x",
        runs=[
            rejected(0.9),
            make_verdict(
                Status.UNVERIFIED, "fast", None, run_id="20261111-195059-000009"
            ),
        ],
    )
    html = landing(source)
    assert (
        '<span class="v u">no score</span><span class="k">last run, unverified</span>'
        in html
    )


def test_case_link_name_starts_with_its_visible_text_and_has_the_app_name():
    html = landing()
    assert (
        '<a href="flight-delays/index.html" '
        'aria-label="Open the case file: flight-delays">Open the case file</a>'
    ) in html


def test_description_paragraph_is_left_out_when_empty():
    html = landing(make_source("x", runs=[approved(0.9)], description=""))
    assert 'class="desc"' not in html
    assert 'class="desc">Predict flight delays.' in landing()


def test_the_note_is_shown_only_when_every_honest_score_is_lower():
    note = (
        "The honest score is lower. That is the point: it is an honest estimate "
        "of how the model will do on new data."
    )
    assert note in landing()
    higher = make_source("x", runs=[rejected(0.7), approved(0.9)])
    assert note not in landing(hotel_source(), higher)
    assert note not in landing(make_source("y", runs=[rejected(0.9)]))
    equal = make_source("z", runs=[rejected(0.9), approved(0.9)])
    assert note not in landing(equal)


def test_kiro_and_loop_screens():
    html = landing()
    text = squeeze(html)
    assert "It lives inside Kiro." in text
    for term in ("Steering", "Save hook", "MCP tools", "Guardrails"):
        assert f"<dt>{term}</dt>" in html
    assert '<dl class="kiro">' in html
    assert "<code>/workflow run exorcise</code> hands an app to the" in html
    assert '<ol class="steps loop-steps">' in html
    for head in ("Edit", "Audit", "Repeat", "Hand back"):
        assert f"<h3>{head}</h3>" in html
    assert "Kiro\u2019s agent gets the rules" in text


def test_the_benchmark_screen_only_with_a_benchmark():
    assert "benchmark-page" not in landing()
    html = landing(benchmark=sample_benchmark())
    assert (
        '<section class="page" id="benchmark-page" aria-labelledby="benchmark">' in html
    )
    assert '<h2 id="benchmark">Benchmark</h2>' in html
    assert (
        html.index('id="cases"')
        < html.index("benchmark-page")
        < html.index('id="kiro"')
    )


def test_the_repository_link_only_with_a_url():
    assert "Source on GitHub" not in landing()
    assert "Open a case file to see every recorded run." in squeeze(landing())
    html = landing(repo_url=REPO)
    assert html.count("Source on GitHub") == 2
    assert html.count(f'href="{REPO}"') == 2
    assert "recorded run, or read the source." in squeeze(html)


def test_the_end_screen_holds_the_footer_inside_main():
    html = landing()
    assert html.index('class="graveyard"') < html.index('<footer class="site-foot">')
    assert html.index('<footer class="site-foot">') < html.index("</main>")
    assert "Released under the MIT licence." in html


def test_the_landing_page_reads_in_order_without_the_stylesheet():
    text = squeeze(landing())
    marks = [
        "Is your model too good",
        "A leak makes a broken model",
        "It scrambles the answers",
        "Five ghosts it hunts.",
        "Two haunted apps",
        "It lives inside Kiro.",
        "Then it fixes the leak",
        "If your model looks too good to be true",
    ]
    positions = [text.index(mark) for mark in marks]
    assert positions == sorted(positions)


def test_hostile_strings_on_the_landing_page():
    first = make_verdict(
        Status.REJECTED,
        value=0.99,
        findings=[
            make_finding(check=text, ghost=text, location=text) for text in HOSTILE
        ],
    )
    source = make_source(
        "<script>alert(1)</script>",
        runs=[first, approved(0.8)],
        description='"><img src=x onerror=alert(1)>',
    )
    summary = summarize_app("x", source)
    html = landing_page(
        [summary], choose_hero([summary]), sample_benchmark(), REPO
    ).html
    page = Page(html)
    for tag, attrs in page.starts:
        assert tag in TAGS | ART_TAGS
        assert tag not in {"script", "img", "style"}
        assert not any(name.startswith("on") or name == "style" for name in attrs)
    assert "<script>" not in html
    assert "<script>alert(1)</script>" in page.text
    assert (
        'aria-label="Open the case file: &lt;script&gt;alert(1)&lt;/script&gt;"' in html
    )
    assert "\x00" not in page.text
    html.encode("utf-8")
