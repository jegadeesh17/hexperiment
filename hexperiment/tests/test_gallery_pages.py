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
    FONTS_URL,
    case_page,
    page_shell,
    relative_href,
    repo_link_label,
    status_mark,
)
from hexperiment.gallery.summary import summarize_app
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
    assert "<ol class=\"record\">" in html
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
