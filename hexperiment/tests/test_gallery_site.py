import dataclasses
import posixpath
import re
from datetime import date
from html.parser import HTMLParser

import pytest

from gallery_builder import (
    approved,
    make_finding,
    make_source,
    make_verdict,
    rejected,
)
from hexperiment.errors import HexInputError
from hexperiment.gallery.links import broken_links, page_links
from hexperiment.gallery.markup import ART_TAGS, TAGS
from hexperiment.gallery.pages import FONTS_URL
from hexperiment.gallery.records import (
    Benchmark,
    BenchmarkMechanism,
    GalleryInputs,
)
from hexperiment.gallery.site import MARKER_TEXT, build_site
from hexperiment.models import Severity, Status

REPO = "https://github.com/example/example"
STYLESHEET = b"body { color: #F1F6F9; }\n"
OLD_TIME = "2020-03-04T05:06:07Z"


def inputs(*sources, repo_url=None, benchmark=None):
    return GalleryInputs(
        apps=tuple(sources),
        benchmark=benchmark,
        repo_url=repo_url,
        stylesheet=STYLESHEET,
    )


def old(verdict):
    run_id = verdict.run_id.replace("2026", "2020")
    return dataclasses.replace(verdict, created_at=OLD_TIME, run_id=run_id)


def one_app():
    return make_source("demo", runs=[old(rejected(0.99)), old(approved(0.9))])


def two_apps():
    return (
        make_source(
            "hotel cancellations",
            runs=[old(rejected(1.0)), old(approved(0.9096))],
            haunted="a\nscale()\nb\n",
            train="a\nb\n",
        ),
        make_source(
            "flight delays",
            runs=[old(rejected(0.7971, ghost="The Prophet")), old(approved(0.6193))],
            haunted="x = 1\n",
            train="x = 2\n",
        ),
    )


def no_approved_app():
    return make_source("pending", runs=[old(rejected(0.99))])


def benchmark():
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
        created_at=OLD_TIME,
        day="2020-03-04",
        hexperiment_version="0.1.0",
        python_version="3.13.1",
        scikit_learn_version="1.6.0",
        source="a taxonomy",
        mechanisms=(mechanism,),
        clean_scripts=12,
        false_alarms=0,
        not_measured=(),
    )


def sites():
    return {
        "one app": build_site(inputs(one_app(), repo_url=REPO)),
        "two apps": build_site(
            inputs(*two_apps(), repo_url=REPO, benchmark=benchmark())
        ),
        "no approved run": build_site(inputs(no_approved_app())),
    }


# the file set


def test_files_of_one_app():
    files = build_site(inputs(one_app()))
    assert set(files) == {
        ".hexperiment-gallery",
        "assets/gallery.css",
        "index.html",
        "demo/index.html",
        "demo/verdict.json",
        "demo/model_card.md",
    }


def test_files_of_two_apps():
    files = build_site(inputs(*two_apps()))
    assert set(files) == {
        ".hexperiment-gallery",
        "assets/gallery.css",
        "index.html",
        "hotel-cancellations/index.html",
        "hotel-cancellations/verdict.json",
        "hotel-cancellations/model_card.md",
        "flight-delays/index.html",
        "flight-delays/verdict.json",
        "flight-delays/model_card.md",
    }


def test_an_app_without_an_approved_run_has_no_model_card():
    files = build_site(inputs(no_approved_app()))
    assert "pending/model_card.md" not in files
    assert "pending/verdict.json" in files


def test_insertion_order_follows_the_design():
    files = build_site(inputs(*two_apps()))
    assert list(files) == [
        ".hexperiment-gallery",
        "assets/gallery.css",
        "index.html",
        "hotel-cancellations/index.html",
        "hotel-cancellations/verdict.json",
        "hotel-cancellations/model_card.md",
        "flight-delays/index.html",
        "flight-delays/verdict.json",
        "flight-delays/model_card.md",
    ]


def test_marker_stylesheet_and_evidence_copies_are_stored_as_given():
    source = one_app()
    files = build_site(inputs(source))
    assert files[".hexperiment-gallery"] == MARKER_TEXT.encode("utf-8")
    assert files["assets/gallery.css"] == STYLESHEET
    assert files["demo/verdict.json"] == source.last_verdict_json
    assert files["demo/model_card.md"] == source.approved_card


FONTS = (
    ("LICENSE-SpecialElite.txt", b"licence text\n"),
    ("special-elite-latin.woff2", b"wOF2\x00\x01\r\n\xff"),
)


def test_the_fonts_are_written_under_assets_fonts_byte_for_byte():
    files = build_site(dataclasses.replace(inputs(one_app()), fonts=FONTS))
    for name, data in FONTS:
        assert files[f"assets/fonts/{name}"] == data
    fonts = [p for p in files if p.startswith("assets/fonts/")]
    assert fonts == [f"assets/fonts/{name}" for name, _ in FONTS]


def test_without_fonts_no_font_file_is_written():
    files = build_site(inputs(one_app()))
    assert not any(p.startswith("assets/fonts/") for p in files)


def test_a_repeated_name_is_refused_by_the_pure_function_too():
    with pytest.raises(HexInputError, match="both give the case folder"):
        build_site(inputs(one_app(), one_app()))


# determinism


def test_building_twice_gives_equal_mappings():
    assert build_site(inputs(*two_apps(), repo_url=REPO)) == build_site(
        inputs(*two_apps(), repo_url=REPO)
    )


def test_the_folder_of_an_app_never_reaches_a_page(tmp_path):
    def moved(source, folder):
        config = dataclasses.replace(source.config, app_dir=folder)
        return dataclasses.replace(source, config=config)

    first = tmp_path / "one" / "demo"
    second = tmp_path / "another place" / "named differently"
    a = build_site(inputs(moved(one_app(), first)))
    b = build_site(inputs(moved(one_app(), second)))
    assert a == b
    for content in a.values():
        assert str(tmp_path).encode() not in content


@pytest.mark.parametrize("name", ["one app", "two apps", "no approved run"])
def test_no_page_contains_the_current_year(name):
    year = str(date.today().year)
    for path, content in sites()[name].items():
        if path.endswith(".html"):
            assert year not in content.decode("utf-8"), path


@pytest.mark.parametrize("name", ["one app", "two apps", "no approved run"])
def test_pages_are_utf8_without_bom_and_without_carriage_returns(name):
    for path, content in sites()[name].items():
        if path.endswith(".html"):
            assert not content.startswith(b"\xef\xbb\xbf"), path
            assert b"\r" not in content, path
            content.decode("utf-8")
            assert content.startswith(b"<!doctype html>\n")


# links, checked here with a parser of the test's own


class LinkParser(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.targets: list[str] = []
        self.ids: set[str] = set()
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("href", "src"):
                self.targets.append(value)
            if name == "id":
                self.ids.add(value)


def check_links_independently(files, repo_url):
    pages = {
        p: LinkParser(c.decode("utf-8"))
        for p, c in files.items()
        if p.endswith(".html")
    }
    for page, parsed in pages.items():
        assert parsed.targets, page
        for target in parsed.targets:
            if target.startswith("#"):
                assert target[1:] in parsed.ids, (page, target)
            elif target.startswith("http"):
                assert target == FONTS_URL or (repo_url and target.startswith(repo_url))
            else:
                assert not target.startswith(("/", "..\\")), (page, target)
                resolved = posixpath.normpath(
                    posixpath.join(posixpath.dirname(page), target)
                )
                assert not resolved.startswith(".."), (page, target)
                assert resolved in files, (page, target)


@pytest.mark.parametrize(
    ("name", "repo"),
    [("one app", REPO), ("two apps", REPO), ("no approved run", None)],
)
def test_every_link_resolves(name, repo):
    check_links_independently(sites()[name], repo)


def test_the_independent_check_sees_the_expected_links():
    files = sites()["two apps"]
    landing = LinkParser(files["index.html"].decode("utf-8"))
    assert "hotel-cancellations/index.html" in landing.targets
    assert "assets/gallery.css" in landing.targets
    case = LinkParser(files["flight-delays/index.html"].decode("utf-8"))
    assert {
        "../index.html",
        "../assets/gallery.css",
        "verdict.json",
        "model_card.md",
    } <= set(case.targets)


def test_page_links_lists_href_and_src_in_order():
    html = '<a href="a.html">x</a><link href="b.css"><a id="q" href="#q">y</a>'
    assert page_links(html) == (("href", "a.html"), ("href", "b.css"), ("href", "#q"))


def test_broken_links_reports_a_missing_target_and_an_unknown_external_address():
    files = {
        "index.html": b'<a href="missing.html">a</a><a href="https://evil.example/">b</a>'
        b'<a href="ok.html">c</a><a href="#nowhere">d</a>',
        "ok.html": b"<p>fine</p>",
    }
    problems = broken_links(files, None)
    assert len(problems) == 3
    assert "missing.html" in problems[0]
    assert "https://evil.example/" in problems[1]
    assert "#nowhere" in problems[2]


def test_broken_links_accepts_the_fonts_the_repo_address_and_fragments():
    files = {
        "a/index.html": (
            f'<link href="{FONTS_URL}"><a href="{REPO}/tree/main">r</a>'
            '<a href="../b.html#top">b</a><a href="#here" id="here">h</a>'
        ).encode(),
        "b.html": b'<h1 id="top">b</h1>',
    }
    assert broken_links(files, REPO) == ()
    assert len(broken_links(files, "https://other.example")) == 1


def test_broken_links_refuses_to_leave_the_site_and_a_missing_fragment():
    files = {
        "a/index.html": b'<a href="../../x.html">x</a><a href="../b.html#gone">b</a>',
        "b.html": b"<p>b</p>",
        "x.html": b"<p>x</p>",
    }
    problems = broken_links(files, None)
    assert len(problems) == 2


def test_build_site_raises_runtime_error_for_a_broken_link():
    source = dataclasses.replace(one_app(), approved_card=None)
    with pytest.raises(RuntimeError, match=r"model_card\.md"):
        build_site(inputs(source))


# well-formed pages


VOID = {"meta", "link", "br"}


class Structure(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.starts: list[tuple[str, dict]] = []
        self.ids: list[str] = []
        self.title = ""
        self.headings: list[int] = []
        self._in_title = False
        self.feed(html)
        self.close()
        assert self.stack == []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.starts.append((tag, attributes))
        if "id" in attributes:
            self.ids.append(attributes["id"])
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append(int(tag[1]))
        self._in_title = tag == "title"
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        assert tag not in VOID
        assert self.stack and self.stack[-1] == tag, (tag, self.stack)
        self.stack.pop()
        self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data

    def count(self, tag):
        return sum(1 for name, _ in self.starts if name == tag)


def structures(files):
    return {
        path: Structure(content.decode("utf-8"))
        for path, content in files.items()
        if path.endswith(".html")
    }


@pytest.mark.parametrize("name", ["one app", "two apps", "no approved run"])
def test_pages_are_well_formed_and_accessible(name):
    pages = structures(sites()[name])
    for path, page in pages.items():
        assert len(page.ids) == len(set(page.ids)), path
        for tag in ("main", "header", "footer", "h1", "title"):
            assert page.count(tag) == 1, (path, tag)
        assert page.starts[0] == ("html", {"lang": "en"})
        body = [i for i, (tag, _) in enumerate(page.starts) if tag == "body"][0]
        first = page.starts[body + 1]
        assert first[0] == "a"
        assert first[1]["href"] == "#main"
        assert "main" in page.ids
        assert page.headings[0] == 1
        for before, after in zip(page.headings, page.headings[1:], strict=False):
            assert after <= before + 1, path
    titles = [page.title for page in pages.values()]
    assert len(titles) == len(set(titles))


def test_the_landing_page_has_one_ordered_list_and_the_cases_are_unordered():
    page = structures(sites()["two apps"])["index.html"]
    assert page.count("ol") == 1


# the hostile site

HOSTILE = [
    "<script>alert(1)</script>",
    '"><img src=x onerror=alert(1)>',
    "`<b>`",
    "&amp;",
    "\x00",
    "\ud800",
]


def hostile_site():
    findings = [
        make_finding(
            check=text,
            ghost=None,
            severity=severity,
            location=text,
            message=text,
            fix=f"Suspect lines: {text}.",
            waiver_reason=text,
        )
        for text in HOSTILE
        for severity in (Severity.ERROR, Severity.WAIVED, Severity.WARNING)
    ]
    first = make_verdict(Status.REJECTED, value=0.99, findings=findings)
    source = make_source(
        "<script>alert(1)</script> app",
        runs=[first, approved(0.8)],
        description=HOSTILE[1],
        target="&amp;",
        haunted="\n".join(HOSTILE) + "\nx = 1\n",
        train="\n".join(HOSTILE[::-1]) + "\nx = 2\n",
    )
    return build_site(inputs(source, repo_url=REPO, benchmark=benchmark()))


def test_the_hostile_site_has_only_allowlisted_tags_and_attributes():
    files = hostile_site()
    check_links_independently(files, REPO)
    for path, page in structures(files).items():
        for tag, attrs in page.starts:
            assert tag in TAGS | ART_TAGS, (path, tag)
            for name in attrs:
                assert not name.startswith("on"), (path, name)
                assert name != "style", path
        assert "<script" not in files[path].decode("utf-8")


def test_the_hostile_text_appears_as_text_on_both_page_kinds():
    files = hostile_site()
    for path in ("index.html", "script-alert-1-script-app/index.html"):
        text = files[path].decode("utf-8")
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text, path
        assert "\x00" not in text
        assert "\ufffd" in text
