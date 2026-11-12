import re
import tomllib
from pathlib import Path, PurePath

import pytest

from gallery_builder import approved, make_source, rejected
from hexperiment.gallery import sources
from hexperiment.gallery.records import GalleryInputs
from hexperiment.gallery.site import build_site
from hexperiment.gallery.sources import read_fonts, read_stylesheet

COLOURS = {
    "sky": "#23153A",
    "night": "#0B0612",
    "well": "#1A0F2A",
    "bone": "#EDE6F5",
    "soft": "#D2C8E1",
    "muted": "#BDB2CF",
    "dash": "#7A6893",
    "rows": "#8A5CCB",
    "orange": "#FF8A1F",
    "orange-soft": "#FFC08A",
    "lime": "#A6FF4D",
    "lime-soft": "#C9FF94",
}
RULE_COLOUR = "rgba(237, 230, 245, 0.16)"
TEXT_COLOURS = ("bone", "soft", "muted", "orange", "orange-soft", "lime", "lime-soft")
REDUCED_MOTION = "@media (prefers-reduced-motion: no-preference)"
SUPPORTS_TIMELINE = "@supports (animation-timeline: view())"
# Classes the markup carries that need no rule of their own.
STRUCTURAL = {"context", "finding-error"}
# Classes of the first page design, removed when the pages are rebuilt.
OLD_PAGE_CLASSES = frozenset({"model-card", "no-findings", "site", "source"})


@pytest.fixture(scope="module")
def css() -> str:
    text = read_stylesheet().decode("utf-8")
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def rule_body(css: str, selector: str) -> str:
    found = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert found, selector
    return found.group(1)


def enclosing_at_rules(css: str, position: int) -> list[str]:
    """The preludes of the at-rules that hold the text at `position`."""
    stack: list[str] = []
    start = 0
    for index, char in enumerate(css[:position]):
        if char == "{":
            stack.append(css[start:index].strip())
            start = index + 1
        elif char == "}":
            stack.pop()
            start = index + 1
        elif char == ";":
            start = index + 1
    return [p for p in stack if p.startswith("@")]


def declarations(css: str, pattern: str) -> list[int]:
    return [m.start() for m in re.finditer(pattern, css)]


def luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def package_data() -> list[str]:
    pyproject = Path(sources.__file__).resolve().parents[3] / "pyproject.toml"
    table = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return table["tool"]["setuptools"]["package-data"]["hexperiment.gallery"]


def test_the_stylesheet_and_fonts_ship_as_package_data():
    patterns = package_data()
    assert "assets/*.css" in patterns
    assert "assets/fonts/*" in patterns
    assert any(PurePath("assets/gallery.css").full_match(p) for p in patterns)
    for name in (
        "special-elite-latin.woff2",
        "archivo-latin.woff2",
        "LICENSE-SpecialElite.txt",
        "OFL-Archivo.txt",
    ):
        assert any(PurePath(f"assets/fonts/{name}").full_match(p) for p in patterns)
    assert {name for name, _ in read_fonts()} >= {
        "special-elite-latin.woff2",
        "archivo-latin.woff2",
    }


def test_the_colours_are_exactly_the_tokens_of_the_steering_file(css):
    found = {c.upper() for c in re.findall(r"#[0-9A-Fa-f]{3,8}\b", css)}
    assert found == set(COLOURS.values())
    for name, value in COLOURS.items():
        assert f"--{name}: {value};" in css
    assert f"--rule: {RULE_COLOUR};" in css
    assert re.findall(r"rgba?\([^)]*\)", css) == [RULE_COLOUR]


@pytest.mark.parametrize(
    "word",
    [
        "gradient",
        "box-shadow",
        "text-shadow",
        "transition",
        "@import",
        "text-transform",
    ],
)
def test_forbidden_features_are_absent(css, word):
    assert word not in css


def test_urls_name_only_the_shipped_fonts(css):
    urls = re.findall(r"url\(([^)]*)\)", css)
    shipped = {name for name, _ in read_fonts()}
    assert len(urls) == 2
    for url in urls:
        found = re.fullmatch(r'"fonts/([^"/]+)"', url)
        assert found and found.group(1) in shipped


def test_font_faces_and_stacks(css):
    assert css.count("@font-face") == 2
    elite = css[css.index('font-family: "Special Elite"') :].split("}")[0]
    assert 'src: url("fonts/special-elite-latin.woff2") format("woff2")' in elite
    assert "font-weight: 400" in elite
    assert "font-display: swap" in elite
    archivo = css[css.index('font-family: "Archivo"') :].split("}")[0]
    assert 'src: url("fonts/archivo-latin.woff2") format("woff2")' in archivo
    assert "font-weight: 100 900" in archivo
    assert "font-display: swap" in archivo
    assert '--type: "Special Elite", "Courier New", serif;' in css
    assert '--text: "Archivo", system-ui, sans-serif;' in css
    assert '--code: ui-monospace, "Cascadia Mono", Consolas, monospace;' in css


def test_figures_use_tabular_lining_numbers(css):
    assert "font-variant-numeric: tabular-nums lining-nums" in css


def test_motion_is_for_users_who_accept_it(css):
    positions = declarations(css, r"(?<![\w-])animation(?:-name)?\s*:")
    positions += declarations(css, r"scroll-behavior:\s*smooth")
    assert positions
    for position in positions:
        assert REDUCED_MOTION in enclosing_at_rules(css, position)


def test_scroll_timelines_need_support(css):
    positions = declarations(
        css, r"(?<![\w(-])animation-timeline\s*:|view-timeline\s*:"
    )
    positions += declarations(css, r"animation-range\s*:")
    assert positions
    for position in positions:
        assert SUPPORTS_TIMELINE in enclosing_at_rules(css, position)


def test_the_load_animations_end_within_five_seconds(css):
    assert "animation: rise 1.4s cubic-bezier(0.16, 1, 0.3, 1) 0.9s both;" in css
    assert "animation: bob 1.3s ease-in-out 2.3s 2 alternate;" in css
    assert 0.9 + 1.4 <= 5
    assert 2.3 + 2 * 1.3 <= 5
    outside = [
        m.start()
        for m in re.finditer(r"animation[^;}]*infinite", css)
        if SUPPORTS_TIMELINE not in enclosing_at_rules(css, m.start())
    ]
    assert not outside


def test_scroll_snap_needs_a_big_screen(css):
    positions = declarations(css, r"scroll-snap-type\s*:")
    assert positions
    for position in positions:
        assert "@media (min-width: 861px) and (min-height: 600px)" in (
            enclosing_at_rules(css, position)
        )


def test_focus_rules(css):
    body = rule_body(css, ":focus-visible")
    assert "outline: 3px solid var(--lime)" in body
    assert "outline-offset: 3px" in body
    assert "outline-color: var(--bone)" in rule_body(css, ".cta:focus-visible")
    assert "outline: none" in rule_body(css, '[tabindex="-1"]:focus')


def test_skip_link_is_off_screen_until_focused(css):
    assert "left: -10000px" in rule_body(css, ".skip")
    shown = rule_body(css, ".skip:focus")
    assert "left: var(--gutter)" in shown
    assert "var(--night)" in rule_body(css, ".skip")
    assert "var(--bone)" in rule_body(css, ".skip")


def test_scroll_boxes_scroll_inside_themselves(css):
    assert "overflow-x: auto" in rule_body(css, ".diff")
    assert "overflow-x: auto" in rule_body(css, ".scroll")
    assert "var(--well)" in rule_body(css, ".diff")


def test_status_mark_rules(css):
    rejected_mark = rule_body(css, ".status-rejected .mark")
    assert "var(--orange)" in rejected_mark
    assert "border-radius: 2px" in rejected_mark
    approved_mark = rule_body(css, ".status-approved .mark")
    assert "var(--lime)" in approved_mark
    assert "border-radius: 50%" in approved_mark
    assert "var(--muted)" in rule_body(css, ".status-unverified")
    unverified = rule_body(css, ".status-unverified .mark")
    assert "1px dashed var(--dash)" in unverified
    assert "border-radius: 50%" in unverified


def test_scores_and_findings_by_state(css):
    assert "var(--muted)" in rule_body(css, ".v.u, .n.u")
    assert "var(--muted)" in rule_body(css, ".finding-waived .sev")
    assert "var(--muted)" in rule_body(css, ".finding p.waiver")


def test_diff_lines_are_coloured_and_unified(css):
    assert "var(--orange-soft)" in rule_body(css, ".line-removed code")
    assert "var(--lime-soft)" in rule_body(css, ".line-added code")
    plain = rule_body(css, ".context, .change, .removed, .added")
    assert "display: block" in plain
    assert "@media (min-width: 1024px)" not in css


def test_model_card_groups_span_when_wide(css):
    assert "minmax(14rem, 1fr)" in rule_body(css, ".card")
    assert "grid-column: 1 / -1" in rule_body(css, ".card-group.wide")


def test_page_rules_that_differ_are_scoped(css):
    for selector in (
        ".hero h1",
        ".case-head h1",
        ".hero .site-head",
        ".case-head .site-head",
    ):
        assert re.search(re.escape(selector) + r"\s*\{", css), selector
    assert "clamp(40px, min(5.6vw, 9vh), 76px)" in rule_body(css, ".hero h1")
    assert "clamp(38px, min(5vw, 8.5vh), 68px)" in rule_body(css, ".case-head h1")


@pytest.mark.parametrize("background", ["sky", "well"])
@pytest.mark.parametrize("foreground", TEXT_COLOURS)
def test_text_colours_meet_wcag_aa(foreground, background):
    assert contrast(COLOURS[foreground], COLOURS[background]) >= 4.5


def test_every_class_used_by_the_pages_has_a_rule_or_is_listed(css):
    one, two = (
        make_source("one", runs=[rejected(1.0), approved(0.9)]),
        make_source(
            "two", runs=[rejected(0.8), approved(0.7)], haunted="a\n", train="b\n"
        ),
    )
    files = build_site(
        GalleryInputs(
            apps=(one, two),
            benchmark=None,
            repo_url="https://github.com/owner/repo",
            stylesheet=b"",
        )
    )
    used: set[str] = set()
    for name, data in files.items():
        if name.endswith(".html"):
            for value in re.findall(r'class="([^"]*)"', data.decode("utf-8")):
                used.update(value.split())
    assert {"diff", "record", "status", "finding"} <= used
    # The list must not hold classes the pages no longer use.
    assert OLD_PAGE_CLASSES <= used

    def has_rule(name: str) -> bool:
        return re.search(r"\." + re.escape(name) + r"(?![\w-])", css) is not None

    missing = {c for c in used - STRUCTURAL - OLD_PAGE_CLASSES if not has_rule(c)}
    assert not missing
    # Old classes get no rule until the pages are rebuilt.
    assert not {c for c in OLD_PAGE_CLASSES if has_rule(c)}
