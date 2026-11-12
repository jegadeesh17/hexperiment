import hashlib
import json
import os
from pathlib import Path

import pytest

from gallery_builder import make_finding, make_verdict, write_app
from hexperiment import cli
from hexperiment.errors import HexInputError
from hexperiment.gallery import command
from hexperiment.gallery.command import (
    build_gallery,
    check_out_dir,
    validate_repo_url,
)
from hexperiment.gallery.writer import write_site
from hexperiment.models import Severity, Status
from test_gallery_benchmark import valid

REPO = "https://github.com/example/example"


def runs():
    return [
        make_verdict(
            Status.REJECTED,
            value=1.0,
            findings=[make_finding(severity=Severity.ERROR)],
            run_id="20261111-195051-000001",
            created_at="2026-11-11T19:50:51Z",
        ),
        make_verdict(
            Status.APPROVED,
            value=0.9096,
            run_id="20261111-195052-000002",
            created_at="2026-11-11T19:50:52Z",
        ),
    ]


def tree_hash(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def tree_paths(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def rename_in_toml(app: Path, name: str) -> None:
    text = (app / "hex.toml").read_text(encoding="utf-8")
    text = text.replace(f'name = "{app.name}"', f'name = "{name}"')
    (app / "hex.toml").write_text(text, encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    """Two apps, and a marked gallery at ``site`` that every refusal must keep."""
    apps = tmp_path / "apps"
    apps.mkdir()
    demo = write_app(apps, "demo", runs(), haunted="a = 1\n", train="a = 2\n")
    other = write_app(apps, "other", runs())
    site = tmp_path / "site"
    write_site({".hexperiment-gallery": b"m", "index.html": b"old"}, site)
    return {"root": tmp_path, "demo": demo, "other": other, "site": site}


def call(world, **overrides):
    arguments = {
        "app_dirs": [world["demo"], world["other"]],
        "out_dir": world["site"],
        "repo_url": None,
        "benchmark": None,
    }
    arguments.update(overrides)
    return build_gallery(
        arguments["app_dirs"],
        arguments["out_dir"],
        repo_url=arguments["repo_url"],
        benchmark=arguments["benchmark"],
    )


# success


def test_a_build_writes_the_site_and_prints_the_path(world, tmp_path):
    out = tmp_path / "fresh"
    outcome = call(world, out_dir=out)
    assert outcome.exit_code == 0
    assert outcome.stderr == ""
    assert outcome.stdout == str(out.resolve() / "index.html")
    assert tree_paths(out) == sorted(
        [
            ".hexperiment-gallery",
            "assets",
            "assets/gallery.css",
            "assets/fonts",
            "assets/fonts/LICENSE-SpecialElite.txt",
            "assets/fonts/OFL-Archivo.txt",
            "assets/fonts/archivo-latin.woff2",
            "assets/fonts/special-elite-latin.woff2",
            "demo",
            "demo/index.html",
            "demo/model_card.md",
            "demo/verdict.json",
            "index.html",
            "other",
            "other/index.html",
            "other/model_card.md",
            "other/verdict.json",
        ]
    )


def test_a_build_writes_the_fonts(world, tmp_path):
    out = tmp_path / "fresh"
    assert call(world, out_dir=out).exit_code == 0
    for name in ("special-elite-latin.woff2", "archivo-latin.woff2"):
        assert (out / "assets" / "fonts" / name).read_bytes().startswith(b"wOF2")


def test_a_marked_gallery_is_replaced(world):
    outcome = call(world)
    assert outcome.exit_code == 0
    assert (world["site"] / "demo" / "index.html").is_file()
    assert (world["site"] / "index.html").read_bytes() != b"old"


def test_the_repository_link_is_on_both_pages_and_absent_without_it(world, tmp_path):
    with_link = tmp_path / "with"
    without = tmp_path / "without"
    assert call(world, out_dir=with_link, repo_url=REPO).exit_code == 0
    assert call(world, out_dir=without).exit_code == 0
    for page in ("index.html", "demo/index.html"):
        assert f'href="{REPO}"' in (with_link / page).read_text(encoding="utf-8")
        assert "example/example" not in (without / page).read_text(encoding="utf-8")


def test_a_benchmark_file_adds_its_section(world, tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps(valid()), encoding="utf-8")
    plain = tmp_path / "plain"
    rich = tmp_path / "rich"
    call(world, out_dir=plain)
    assert call(world, out_dir=rich, benchmark=results).exit_code == 0
    assert 'id="benchmark"' not in (plain / "index.html").read_text(encoding="utf-8")
    assert 'id="benchmark"' in (rich / "index.html").read_text(encoding="utf-8")


def test_an_empty_out_dir_is_accepted(world, tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    assert call(world, out_dir=out).exit_code == 0


def test_a_build_does_not_touch_the_apps(world):
    before = tree_hash(world["root"] / "apps")
    assert call(world).exit_code == 0
    assert tree_hash(world["root"] / "apps") == before
    assert not (world["demo"] / ".hex").exists()
    for run in (world["demo"] / "runs").iterdir():
        assert run.name.startswith("2026")
    assert not (world["demo"] / "verdict.json").exists()


# refusals


def prepare_not_an_app(world):
    empty = world["root"] / "empty"
    empty.mkdir()
    return {"app_dirs": [empty]}


def prepare_no_run(world):
    return {"app_dirs": [write_app(world["root"], "norun", [])]}


def prepare_same_name(world):
    elsewhere = world["root"] / "elsewhere"
    elsewhere.mkdir()
    twin = write_app(elsewhere, "other", runs())
    rename_in_toml(twin, "demo")
    return {"app_dirs": [world["demo"], twin]}


def prepare_same_folder(world):
    return {"app_dirs": [world["demo"], world["demo"]]}


def prepare_name(name):
    def prepare(world):
        rename_in_toml(world["demo"], name)
        return {}

    return prepare


def prepare_out(make):
    def prepare(world):
        return {"out_dir": make(world)}

    return prepare


def make_unmarked(world):
    out = world["root"] / "unmarked"
    out.mkdir()
    (out / "mine.txt").write_text("keep", encoding="utf-8")
    return out


def make_file(world):
    path = world["root"] / "afile"
    path.write_text("x", encoding="utf-8")
    return path


def make_link(world):
    link = world["root"] / "link"
    try:
        os.symlink(world["site"], link, target_is_directory=True)
    except OSError:
        pytest.skip("this platform does not allow links here")
    return link


def prepare_url(url):
    def prepare(world):
        return {"repo_url": url}

    return prepare


def prepare_benchmark(mutate):
    def prepare(world):
        data = valid()
        mutate(data)
        path = world["root"] / "results.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return {"benchmark": path}

    return prepare


def prepare_missing_benchmark(world):
    return {"benchmark": world["root"] / "nope.json"}


REFUSALS = {
    "not an app folder": prepare_not_an_app,
    "no recorded run": prepare_no_run,
    "same name twice": prepare_same_name,
    "same folder twice": prepare_same_folder,
    "empty slug": prepare_name("!!!"),
    "assets": prepare_name("assets"),
    "device name": prepare_name("con"),
    "out non-empty without marker": prepare_out(make_unmarked),
    "out inside an app": prepare_out(lambda w: w["demo"] / "site"),
    "out is an app": prepare_out(lambda w: w["demo"]),
    "out holds an app": prepare_out(lambda w: w["root"] / "apps"),
    "out is a file": prepare_out(make_file),
    "out is a link": prepare_out(make_link),
    "out has no parent": prepare_out(lambda w: w["root"] / "no" / "site"),
    "url javascript": prepare_url("javascript:alert(1)"),
    "url without host": prepare_url("https://"),
    "url with space": prepare_url("https://a b.org"),
    "url with quote": prepare_url('https://a.org/"x'),
    "url with angle bracket": prepare_url("https://a.org/<x>"),
    "url ftp": prepare_url("ftp://a.org"),
    "benchmark missing": prepare_missing_benchmark,
    "benchmark wrong": prepare_benchmark(lambda d: d.pop("mechanisms")),
    "benchmark incomplete": prepare_benchmark(lambda d: d.update(complete=False)),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_a_refusal_exits_2_with_one_line_and_changes_nothing(world, name):
    overrides = REFUSALS[name](world)
    before = tree_hash(world["root"])
    paths_before = tree_paths(world["root"])
    outcome = call(world, **overrides)
    assert outcome.exit_code == 2
    assert outcome.stdout == ""
    assert outcome.stderr.startswith("hexperiment: ")
    assert "\n" not in outcome.stderr.strip()
    assert tree_hash(world["root"]) == before
    assert tree_paths(world["root"]) == paths_before


def test_internal_errors_exit_2_with_a_traceback(world, monkeypatch):
    def broken(inputs):
        raise RuntimeError("the site has a broken link")

    monkeypatch.setattr(command, "build_site", broken)
    before = tree_hash(world["root"])
    outcome = call(world)
    assert outcome.exit_code == 2
    assert outcome.stderr.startswith("hexperiment: internal error: RuntimeError")
    assert "Traceback" in outcome.stderr
    assert tree_hash(world["root"]) == before


def test_validate_repo_url_returns_a_good_url_unchanged():
    assert validate_repo_url(REPO) == REPO
    assert validate_repo_url("http://example.org:8080/a?b=1#c") == (
        "http://example.org:8080/a?b=1#c"
    )


def test_check_out_dir_accepts_a_new_folder(world):
    check_out_dir(world["root"] / "brand-new", [world["demo"]])


def test_check_out_dir_refuses_text_that_names_the_problem(world):
    with pytest.raises(HexInputError, match="inside the app folder"):
        check_out_dir(world["demo"] / "site", [world["demo"]])


# command line


def test_main_prints_the_path_and_returns_0(world, tmp_path, capsys):
    out = tmp_path / "cli-out"
    code = cli.main(
        ["gallery", str(world["demo"]), str(world["other"]), "--out", str(out)]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == str(out.resolve() / "index.html")
    assert captured.err == ""


def test_main_prints_a_refusal_on_standard_error(world, tmp_path, capsys):
    code = cli.main(["gallery", str(tmp_path / "nope"), "--out", str(tmp_path / "o")])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.startswith("hexperiment: ")


def test_main_passes_the_options_through(world, tmp_path, capsys):
    results = tmp_path / "results.json"
    results.write_text(json.dumps(valid()), encoding="utf-8")
    out = tmp_path / "opts"
    code = cli.main(
        [
            "gallery",
            str(world["demo"]),
            "--out",
            str(out),
            "--repo-url",
            REPO,
            "--benchmark",
            str(results),
        ]
    )
    assert code == 0
    page = (out / "index.html").read_text(encoding="utf-8")
    assert REPO in page
    assert 'id="benchmark"' in page


def test_main_prints_exactly_the_text_of_build_gallery(world, tmp_path, capsys):
    out = tmp_path / "same"
    outcome = build_gallery([world["demo"]], out, repo_url=None, benchmark=None)
    code = cli.main(["gallery", str(world["demo"]), "--out", str(out)])
    assert code == outcome.exit_code
    assert capsys.readouterr().out == outcome.stdout + "\n"


def test_a_failure_to_print_keeps_the_exit_code(world, tmp_path, monkeypatch):
    def broken_print(*args, **kwargs):
        raise OSError("closed")

    monkeypatch.setattr(cli, "print_text", broken_print)
    code = cli.main(["gallery", str(world["demo"]), "--out", str(tmp_path / "quiet")])
    assert code == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["gallery", "app"],
        ["gallery", "--out", "site"],
        ["gallery", "app", "--out", "site", "--no-such-option"],
        ["gallery"],
    ],
)
def test_argparse_errors_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert caught.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_the_parser_reads_the_options():
    args = cli.build_parser().parse_args(
        ["gallery", "a", "b", "--out", "o", "--repo-url", "u", "--benchmark", "f"]
    )
    assert (args.app_dirs, args.out, args.repo_url, args.benchmark) == (
        ["a", "b"],
        "o",
        "u",
        "f",
    )


# determinism


def test_lf_and_crlf_apps_give_the_same_site_bytes(tmp_path):
    outputs = []
    for label, newline in (("lf", "\n"), ("crlf", "\r\n")):
        root = tmp_path / label
        root.mkdir()
        app = write_app(
            root,
            "demo",
            runs(),
            haunted="a = 1\nb = 2\n",
            train="a = 1\nb = 3\n",
            newline=newline,
        )
        out = root / "out"
        outcome = build_gallery([app], out, repo_url=REPO, benchmark=None)
        assert outcome.exit_code == 0, outcome.stderr
        outputs.append(out)
    lf, crlf = outputs
    assert tree_paths(lf) == tree_paths(crlf)
    for relative in tree_paths(lf):
        path_lf, path_crlf = lf / relative, crlf / relative
        if path_lf.is_file():
            assert path_lf.read_bytes() == path_crlf.read_bytes(), relative
            # Font files are binary: a 0x0D byte in them is data, not a line end.
            if not relative.endswith(".woff2"):
                assert b"\r" not in path_lf.read_bytes(), relative


def test_the_apps_and_their_test_set_uses_are_unchanged_for_crlf(tmp_path):
    app = write_app(tmp_path, "demo", runs(), newline="\r\n")
    before = tree_hash(app)
    assert (
        build_gallery([app], tmp_path / "o", repo_url=None, benchmark=None).exit_code
        == 0
    )
    assert tree_hash(app) == before
    for verdict in (app / "runs").glob("*/verdict.json"):
        assert json.loads(verdict.read_text(encoding="utf-8"))["test_set_uses"] == 1
