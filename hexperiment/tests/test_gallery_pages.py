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
from hexperiment.gallery.markup import TAGS, el
from hexperiment.gallery.pages import (
    CHECK_STEPS,
    EXPLAINER,
    FONTS_URL,
    GHOST_GLOSS,
    LANDING_TITLE,
    METRIC_RANGE,
    MOTTO,
    case_page,
    hero_sentence,
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
    assert page.tags("main") == [{"id": "main", "tabindex": "-1"}]
    assert len(page.tags("header")) == len(page.tags("footer")) == 1


def test_fonts_link_is_the_only_external_reference_and_has_no_preconnect():
    html = build()
    hrefs = [a["href"] for a in Page(html).tags("link")]
    assert hrefs == [FONTS_URL, "../assets/gallery.css"]
    assert "&amp;family=IBM+Plex+Mono" in html
    assert "preconnect" not in html


def test_skip_link_comes_first_in_the_body():
    page = Page(build())
    names = [tag for tag, _ in page.starts]
    after_body = page.starts[names.index("body") + 1]
    assert after_body == ("a", {"class": "skip", "href": "#main"})


def test_titles_differ_between_apps():
    a = build(make_source("alpha"), slug="alpha")
    b = build(make_source("beta"), slug="beta")
    assert "<title>Hexperiment: alpha</title>" in a
    assert "<title>Hexperiment: beta</title>" in b


def test_repo_link_in_header_and_footer_for_github():
    html = build(repo_url=REPO)
    links = [a for a in Page(html).tags("a") if a.get("href") == REPO]
    assert len(links) == 2
    assert html.count("Source on GitHub") == 2


def test_repo_link_label_for_another_host():
    html = build(repo_url="https://example.org/x")
    assert html.count("Source code") == 2
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


def test_every_field_of_a_timeline_entry():
    html = build(runs_source())
    page = Page(html)
    items = [a for a in page.tags("li") if a.get("class") == "run"]
    assert len(items) == 3
    assert '<ol class="record">' in html
    text = squeeze(html)
    assert "REJECTED full 1.0000 2026-11-11 19:50:51 UTC test set used 1 time" in text
    unverified = "UNVERIFIED fast 0.9090 2026-11-12 08:01:02 UTC test set used 2 times"
    assert unverified in text
    assert '<time datetime="2026-11-12T08:01:02Z">' in html
    assert text.index("REJECTED") < text.index("UNVERIFIED") < text.index("APPROVED")


def test_finding_shows_ghost_severity_location_message_and_fix():
    text = squeeze(build(runs_source()))
    assert "The Possession error at reservation_status" in text
    assert "reservation_status alone predicts the target with AUC 1.0000" in text
    assert "Fix: Check whether this column is recorded after the outcome." in text


def test_finding_without_ghost_shows_check_name():
    finding = make_finding(
        check="split_order",
        ghost=None,
        severity=Severity.WARNING,
        location="train.py:3",
    )
    source = make_source(runs=[make_verdict(findings=[finding])])
    html = build(source)
    assert "split_order warning at" in squeeze(html)
    assert 'class="finding finding-warning"' in html


def test_waived_finding_shows_reason():
    finding = make_finding(severity=Severity.WAIVED, waiver_reason="known at booking")
    html = build(make_source(runs=[make_verdict(findings=[finding])]))
    text = squeeze(html)
    assert "The Possession waived at reservation_status" in text
    assert "Waiver reason: known at booking" in text
    assert 'class="finding finding-waived"' in html


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
    assert "No findings." in build()


def test_a_run_without_score():
    source = make_source(runs=[make_verdict(Status.REJECTED, value=None)])
    assert "REJECTED full no score" in squeeze(build(source))


def test_time_that_is_not_utc_iso_is_shown_as_written():
    source = make_source(runs=[make_verdict(created_at="yesterday")])
    html = build(source)
    assert "<time>yesterday</time>" in html


def test_fact_line_and_heading():
    source = make_source(
        "hotel", description="Will it be cancelled?", target="is_canceled"
    )
    html = build(source, slug="hotel")
    assert "<h1>hotel</h1>" in html
    assert "Will it be cancelled?" in html
    assert (
        "target is_canceled, metric roc_auc, threshold 0.8500, 1 recorded run"
        in squeeze(html)
    )
    assert '<a href="../index.html">All cases</a>' in html


def test_empty_description_is_left_out():
    html = build(make_source(description=""))
    assert "<p></p>" not in html


# fix, card, raw files


def test_no_fix_section_without_haunted_script():
    assert "The fix" not in build()


def test_fix_section_with_diff():
    source = make_source(haunted="a\nscale()\nb\n", train="a\nb\n")
    html = build(source)
    text = squeeze(html)
    assert "<h2>The fix</h2>" in html
    assert "haunted/train.py against train.py: 1 line removed, 0 lines added." in text
    assert 'class="diff"' in html
    assert 'class="line line-removed"' in html


def test_identical_scripts_sentence():
    source = make_source(haunted="a\n", train="a\n")
    html = build(source)
    assert "The two scripts are identical." in html
    assert 'class="diff"' not in html


def test_no_approved_run_yet():
    source = make_source(runs=[rejected(1.0)])
    html = build(source)
    assert "No approved run yet." in html
    assert "model_card.md" not in html
    assert '<a href="verdict.json">verdict.json</a>' in html


def test_model_card_of_the_last_approved_run_and_raw_links():
    source = make_source(runs=[approved(0.91), rejected(0.99), approved(0.9096)])
    html = build(source)
    assert "Model card of the last approved run" in html
    assert "0.9096" in html
    assert '<a href="model_card.md">model_card.md</a>' in html
    assert '<a href="verdict.json">verdict.json</a>' in html


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
        assert tag in TAGS
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


def canary_finding(location="train.py", fix=CANARY_FIX):
    return make_finding(
        check="canary",
        ghost="The Peeking Spirit",
        location=location,
        message="Scrambling the test rows changed training.",
        fix=fix,
    )


def hotel_source():
    first = make_verdict(
        Status.REJECTED,
        value=1.0,
        findings=[make_finding(location="reservation_status"), canary_finding()],
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


def paragraph(html: str, cls: str) -> str:
    found = re.search(rf'<p class="{cls}"[^>]*>(.*?)</p>', html, re.S)
    assert found, cls
    return found.group(1)


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


def test_landing_is_the_whole_document():
    html = landing()
    assert html.startswith('<html lang="en">')
    assert f"<title>{LANDING_TITLE}</title>" in html
    assert 'href="assets/gallery.css"' in html
    assert Page(html).tags("main") == [{"id": "main", "tabindex": "-1"}]


def test_hero_shows_the_haunted_figure_and_the_honest_score():
    html = landing()
    assert '<p class="haunted" data-value="1.0000">1.0000</p>' in html
    assert paragraph(html, "honest") == "0.9096"
    lede = squeeze(f"<p>{paragraph(html, 'lede')}</p>")
    assert lede.startswith(
        "On its first run, hotel-cancellations scored 1.0000 on ROC AUC, "
        "where 0.5 is guessing and 1.0 is perfect."
    )
    assert lede.endswith("With the leaks removed, its honest score is")


def test_hero_sentence_lists_error_findings_in_order_with_suspect_lines():
    lede = squeeze(f"<p>{paragraph(landing(), 'lede')}</p>")
    assert (
        "The audit found The Possession, a column that gives the answer away, "
        "at reservation_status and The Peeking Spirit, training that saw the "
        "test rows, at train.py (suspect lines train.py:100)." in lede
    )


def test_hero_sentence_joins_three_phrases_and_several_suspect_lines():
    first = make_verdict(
        Status.REJECTED,
        value=0.99,
        findings=[
            make_finding(location="a"),
            make_finding(ghost=None, check="other", location=""),
            canary_finding(fix="Suspect lines: train.py:3, train.py:9."),
        ],
    )
    html = landing(make_source("x", runs=[first, approved(0.8)]))
    lede = squeeze(f"<p>{paragraph(html, 'lede')}</p>")
    assert (
        "The audit found The Possession, a column that gives the answer away, "
        "at a, other and The Peeking Spirit, training that saw the test rows, at "
        "train.py (suspect lines train.py:3, train.py:9)." in lede
    )


def test_warnings_and_waived_findings_are_not_in_the_sentence():
    first = make_verdict(
        Status.REJECTED,
        value=0.99,
        findings=[
            make_finding(location="e-loc"),
            make_finding(severity=Severity.WARNING, location="w-loc"),
            make_finding(severity=Severity.WAIVED, location="v-loc", waiver_reason="r"),
        ],
    )
    lede = paragraph(landing(make_source("x", runs=[first, approved(0.8)])), "lede")
    assert "e-loc" in lede
    assert "w-loc" not in lede
    assert "v-loc" not in lede


def test_hero_without_error_findings():
    first = make_verdict(Status.REJECTED, value=0.99)
    html = landing(make_source("x", runs=[first, approved(0.8)]))
    lede = squeeze(f"<p>{paragraph(html, 'lede')}</p>")
    assert "The audit rejected it. With the leaks removed, its honest score is" in lede
    assert "The audit found" not in lede


def test_hero_without_an_approved_run():
    html = landing(make_source("x", runs=[rejected(0.99)]))
    lede = squeeze(f"<p>{paragraph(html, 'lede')}</p>")
    assert lede.endswith("No approved run yet.")
    assert "honest score" not in lede
    assert 'class="honest"' not in html
    assert 'class="haunted"' in html


def test_hero_without_a_rejected_run():
    html = landing(make_source("x", runs=[approved(0.9)]))
    assert "No run has been rejected yet." in html
    assert 'class="haunted"' not in html
    assert 'class="honest"' not in html
    assert f'<h1 id="motto">{MOTTO}</h1>'.replace("'", "&#x27;") in html


def test_hero_skips_a_rejected_run_without_a_score():
    html = landing(make_source("x", runs=[make_verdict(Status.REJECTED, value=None)]))
    assert "No run has been rejected yet." in html


def test_hero_uses_the_app_with_the_highest_first_rejected_score():
    html = landing(flight_source(), hotel_source())
    assert 'data-value="1.0000"' in html
    assert paragraph(html, "honest") == "0.9096"


def test_metric_label_other_than_roc_auc_is_the_metric_name():
    first = make_verdict(Status.REJECTED, value=0.99)
    first = dataclasses.replace(
        first, metric=dataclasses.replace(first.metric, name="f1")
    )
    summary = summarize_app("x", make_source("x", runs=[first]))
    sentence = squeeze(hero_sentence(choose_hero([summary])).html)
    assert "scored 0.9900 on f1." in sentence
    assert "where" not in sentence.split("The audit")[0]


def test_the_motto_is_the_only_h1_and_the_figure_is_not_a_heading():
    html = landing()
    page = Page(html)
    assert page.tags("h1") == [{"id": "motto"}]
    assert re.findall(r"<h1[^>]*>(.*?)</h1>", html) == [
        "If your model looks too good to be true, it&#x27;s haunted."
    ]
    assert re.search(r"<h[1-6][^>]*>\s*1\.0000", html) is None


def test_the_motto_comes_after_the_figure_and_its_sentence():
    html = landing()
    assert html.index('haunted"') < html.index("lede") < html.index("honest")
    assert html.index("honest") < html.index("<h1")


def test_the_cases_are_an_unnumbered_list_and_the_page_has_one_ol():
    html = landing()
    assert '<ul class="cases">' in html
    assert html.count("<ol") == 1
    assert len(Page(html).tags("ol")) == 1
    cases = [a for a in Page(html).tags("li") if a.get("class") == "case"]
    assert len(cases) == 2


def test_every_summary_field():
    text = squeeze(landing())
    assert "Two cases" in text
    assert (
        "hotel-cancellations Predict whether a hotel booking will be cancelled." in text
    )
    assert "REJECTED on its first run" in text
    assert "The Possession, The Peeking Spirit" in text
    assert "from 1.0000 to 0.9096" in text
    assert "Last run: APPROVED" in text
    assert "4 recorded runs" in text
    assert "flight-delays Predict flight delays." in text
    assert "The Prophet" in text
    assert "from 0.7971 to 0.6193" in text
    assert "2 recorded runs" in text


def test_summary_links_to_the_case_page_by_file():
    hrefs = [a["href"] for a in Page(landing()).tags("a") if "href" in a]
    assert "hotel-cancellations/index.html" in hrefs
    assert "flight-delays/index.html" in hrefs
    assert squeeze(landing()).count("Read the case") == 2


def test_an_app_without_a_ghost_says_so_and_one_case_is_singular():
    html = landing(make_source("solo", runs=[approved(0.9)]))
    assert "One case" in html
    assert "No ghost found" in html
    assert "1 recorded run" in squeeze(html)
    assert "1 recorded runs" not in squeeze(html)


def test_ten_cases_are_counted_in_digits():
    sources = [make_source(f"app{n}", runs=[approved(0.9)]) for n in range(10)]
    assert "<h2>10 cases</h2>" in landing(*sources)


def test_the_three_steps_in_at_most_80_words():
    html = landing()
    assert len(CHECK_STEPS) == 3
    for step in CHECK_STEPS:
        assert f"<li>{step}</li>" in html
    words = ("How the check works " + " ".join(CHECK_STEPS)).split()
    assert len(words) <= 80
    assert "<h2>How the check works</h2>" in html


def test_every_catalogue_entry_with_a_ghost_has_its_three_texts():
    html = landing()
    entries = [entry for entry in CATALOG if entry.ghost is not None]
    assert len(entries) == 5
    assert html.count("<dt>") == 5
    text = squeeze(html)
    for entry in entries:
        assert f'<dt><em class="ghost">{entry.ghost}</em></dt>' in html
        for field in (entry.meaning, entry.why_it_matters, entry.how_to_fix):
            assert field.replace("`", "") in text
    assert text.count("Why it matters:") == 5
    assert text.count("How to fix it:") == 5


def test_catalogue_code_spans_become_code():
    html = landing()
    assert "<code>hexperiment.split</code>" in html
    assert "<code>checks.proxy_auc_max</code>" in html
    assert "`" not in html.split("<h2>The ghosts</h2>")[1]


def test_checks_without_a_ghost_are_not_in_the_catalogue():
    html = landing()
    assert "split_order" not in html
    assert "test_set_uses" not in html


def test_how_it_was_built():
    html = landing()
    assert "<h2>How it was built</h2>" in html
    assert "Kiro specs" in html
    assert "Released under the MIT licence." in html


def test_no_benchmark_section_without_a_benchmark():
    html = landing()
    assert 'id="benchmark"' not in html
    assert "Benchmark" not in html


def test_benchmark_section_with_a_benchmark_between_steps_and_ghosts():
    html = landing(benchmark=sample_benchmark())
    assert '<h2 id="benchmark">Benchmark</h2>' in html
    assert "rejected 9 of 12 (75.0%)" in html
    assert (
        html.index("How the check works")
        < html.index('id="benchmark"')
        < html.index("<h2>The ghosts</h2>")
    )


def test_repo_link_on_the_landing_page_only_when_given():
    assert "Source on GitHub" not in landing()
    html = landing(repo_url=REPO)
    assert html.count("Source on GitHub") == 2


def test_hostile_strings_on_the_landing_page():
    first = make_verdict(
        Status.REJECTED,
        value=0.99,
        findings=[
            make_finding(
                check=text,
                ghost=None,
                location=text,
                fix=f"Suspect lines: {text}, {text}.",
            )
            for text in HOSTILE
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
        assert tag in TAGS
        assert tag not in {"script", "img", "style"}
        assert not any(name.startswith("on") or name == "style" for name in attrs)
    assert "<script>" not in html
    assert "<script>alert(1)</script>" in page.text
    assert "\ufffd" in page.text
    assert "\x00" not in page.text
    html.encode("utf-8")


# plain language (task 11a)


def sentence_of(source):
    summary = summarize_app("x", source)
    return squeeze(hero_sentence(choose_hero([summary])).html)


def test_sentence_names_the_run_number_when_not_the_first():
    source = make_source("x", runs=[approved(0.8), rejected(0.99), approved(0.9)])
    assert sentence_of(source).startswith("On run 2, x scored 0.9900 on ROC AUC, ")


def test_sentence_without_a_gloss_for_a_ghost_or_a_check():
    first = make_verdict(
        Status.REJECTED,
        value=0.99,
        findings=[
            make_finding(ghost="The Unknown", location="col"),
            make_finding(ghost=None, check="other", location="loc"),
        ],
    )
    text = sentence_of(make_source("x", runs=[first, approved(0.8)]))
    assert "The audit found The Unknown at col and other at loc." in text


def test_a_gloss_without_a_location_has_no_trailing_comma():
    first = make_verdict(
        Status.REJECTED, value=0.99, findings=[make_finding(location="")]
    )
    text = sentence_of(make_source("x", runs=[first, approved(0.8)]))
    assert (
        "The audit found The Possession, a column that gives the answer away. " in text
    )


def test_metric_range_has_roc_auc_only():
    assert dict(METRIC_RANGE) == {"roc_auc": "0.5 is guessing and 1.0 is perfect"}


def test_every_catalogue_ghost_has_a_short_gloss():
    for entry in CATALOG:
        if entry.ghost is not None:
            assert 0 < len(GHOST_GLOSS[entry.ghost].split()) <= 8


def test_explainer_is_short_and_follows_the_motto():
    assert len(EXPLAINER.split()) <= 70
    html = landing()
    assert paragraph(html, "explain") == EXPLAINER
    assert html.index("<h1") < html.index('class="explain"')


def test_explainer_is_in_the_hero_without_a_hero():
    html = landing(make_source("x", runs=[approved(0.9)]))
    assert 'class="haunted"' not in html
    assert paragraph(html, "explain") == EXPLAINER
    assert html.index("<h1") < html.index('class="explain"')


def test_hostile_ghost_name_and_location_are_escaped_in_the_sentence():
    first = make_verdict(
        Status.REJECTED,
        value=0.99,
        findings=[make_finding(ghost="<b>x</b>", location="<script>1</script>")],
    )
    html = landing(make_source("x", runs=[first, approved(0.8)]))
    lede = paragraph(html, "lede")
    assert "<script>" not in lede
    assert "&lt;script&gt;" in lede
    assert "<b>" not in lede
