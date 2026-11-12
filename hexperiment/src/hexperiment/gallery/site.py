"""Turn the inputs of a gallery build into the files of the site (pure)."""

from __future__ import annotations

from hexperiment.gallery.links import broken_links
from hexperiment.gallery.markup import document
from hexperiment.gallery.pages import (
    LANDING_PATH,
    STYLESHEET_PATH,
    case_page,
    landing_page,
)
from hexperiment.gallery.records import GalleryInputs
from hexperiment.gallery.summary import (
    MARKER_FILENAME,
    assign_slugs,
    choose_hero,
    summarize_app,
)

MARKER_TEXT = "Built by hexperiment gallery. The next build replaces this folder.\n"
FONTS_DIR = "assets/fonts"


def build_site(inputs: GalleryInputs) -> dict[str, bytes]:
    """Relative path to file content for the whole site.

    Raises ``RuntimeError`` when a page links to something the build did not write:
    that is a bug in the gallery, not bad input.
    """
    names = [source.config.app.name for source in inputs.apps]
    slugs = assign_slugs(list(zip(names, names, strict=True)))
    summaries = [
        summarize_app(slug, source)
        for slug, source in zip(slugs, inputs.apps, strict=True)
    ]
    hero = choose_hero(summaries)

    files: dict[str, bytes] = {
        MARKER_FILENAME: MARKER_TEXT.encode("utf-8"),
        STYLESHEET_PATH: inputs.stylesheet,
    }

    for name, data in inputs.fonts:
        files[f"{FONTS_DIR}/{name}"] = data
    files[LANDING_PATH] = document(
        landing_page(summaries, hero, inputs.benchmark, inputs.repo_url)
    )
    for summary, source in zip(summaries, inputs.apps, strict=True):
        files[f"{summary.slug}/index.html"] = document(
            case_page(summary, source, inputs.repo_url)
        )
        files[f"{summary.slug}/verdict.json"] = source.last_verdict_json
        if source.approved_card is not None:
            files[f"{summary.slug}/model_card.md"] = source.approved_card

    problems = broken_links(files, inputs.repo_url)
    if problems:
        raise RuntimeError(f"the site has a broken link: {problems[0]}")
    return files
