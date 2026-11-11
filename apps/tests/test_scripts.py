"""Static rules for the training scripts (Requirements 2.2, 2.4, 3.2, 3.4, 5.1).

Nothing is executed: the scripts are parsed with ``ast`` and ``tokenize``.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path
from types import ModuleType

import pytest
from conftest import APPS_DIR, FLIGHTS, HOTELS, script_module

HONEST_HOTELS = APPS_DIR / HOTELS / "train.py"
HAUNTED_HOTELS = APPS_DIR / HOTELS / "haunted" / "train.py"
HONEST_FLIGHTS = APPS_DIR / FLIGHTS / "train.py"
HAUNTED_FLIGHTS = APPS_DIR / FLIGHTS / "haunted" / "train.py"

ALL_SCRIPTS = [
    pytest.param(HONEST_HOTELS, id="hotels-honest"),
    pytest.param(HAUNTED_HOTELS, id="hotels-haunted"),
    pytest.param(HONEST_FLIGHTS, id="flights-honest"),
    pytest.param(HAUNTED_FLIGHTS, id="flights-haunted"),
]
HONEST_SCRIPTS = [
    pytest.param(HONEST_HOTELS, id="hotels-honest"),
    pytest.param(HONEST_FLIGHTS, id="flights-honest"),
]
HAUNTED_SCRIPTS = [
    pytest.param(HAUNTED_HOTELS, id="hotels-haunted"),
    pytest.param(HAUNTED_FLIGHTS, id="flights-haunted"),
]
FLIGHT_SCRIPTS = [
    pytest.param(HONEST_FLIGHTS, id="flights-honest"),
    pytest.param(HAUNTED_FLIGHTS, id="flights-haunted"),
]
LEAK_COMMENT = "# LEAK:"
RANDOM_SPLIT_MARKER = "# LEAK: random split"
POST_OUTCOME_HOTEL_COLUMNS = (
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
)


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def dotted(node: ast.expr) -> str:
    """The dotted name of a call target, e.g. ``hexperiment.audit``; '' if unnamed."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def call_nodes(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def call_names(tree: ast.AST) -> list[str]:
    return [dotted(call.func) for call in call_nodes(tree)]


def string_constants(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def comments(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [tok.string for tok in tokens if tok.type == tokenize.COMMENT]


def main_function(tree: ast.Module) -> ast.FunctionDef:
    found = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    assert len(found) == 1, "the script needs exactly one main()"
    return found[0]


@pytest.mark.parametrize("path", ALL_SCRIPTS)
def test_script_parses_and_loads_data_with_the_kit(path: Path) -> None:
    names = call_names(parse(path))
    assert "hexperiment.load" in names


@pytest.mark.parametrize("path", ALL_SCRIPTS)
def test_audit_is_the_last_statement_of_main(path: Path) -> None:
    last = main_function(parse(path)).body[-1]
    assert isinstance(last, ast.Expr)
    assert isinstance(last.value, ast.Call)
    assert dotted(last.value.func) == "hexperiment.audit"
    keywords = {kw.arg: kw.value for kw in last.value.keywords}
    assert "y_score_train" in keywords
    script = keywords.get("script")
    assert isinstance(script, ast.Name)
    assert script.id == "__file__"


@pytest.mark.parametrize("path", ALL_SCRIPTS)
def test_audit_is_called_once(path: Path) -> None:
    assert call_names(parse(path)).count("hexperiment.audit") == 1


@pytest.mark.parametrize("path", ALL_SCRIPTS)
def test_every_random_component_has_a_seed_and_no_early_stopping(path: Path) -> None:
    """Early stopping draws a random validation split, which the canary would see."""
    seeded = ("HistGradientBoostingClassifier", "train_test_split")
    boosting = 0
    for call in call_nodes(parse(path)):
        name = dotted(call.func).split(".")[-1]
        if name not in seeded:
            continue
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert "random_state" in keywords, ast.dump(call)
        if name == "HistGradientBoostingClassifier":
            boosting += 1
            stopping = keywords.get("early_stopping")
            assert isinstance(stopping, ast.Constant), ast.dump(call)
            assert stopping.value is False, ast.dump(call)
    assert boosting >= 1, "the script must build a HistGradientBoostingClassifier"


@pytest.mark.parametrize("path", ALL_SCRIPTS)
def test_nothing_resets_the_index(path: Path) -> None:
    for call in call_nodes(parse(path)):
        assert not (
            isinstance(call.func, ast.Attribute) and call.func.attr == "reset_index"
        )


@pytest.mark.parametrize("path", ALL_SCRIPTS)
def test_overlap_is_removed_after_the_split_and_nothing_is_deduplicated_before(
    path: Path,
) -> None:
    names = call_names(parse(path))
    assert "drop_seen_test_rows" in names
    for call in call_nodes(parse(path)):
        assert not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "drop_duplicates"
        )


@pytest.mark.parametrize("path", HONEST_SCRIPTS)
def test_honest_script_splits_with_the_kit(path: Path) -> None:
    names = call_names(parse(path))
    assert "hexperiment.split" in names
    assert not [n for n in names if n.split(".")[-1] == "train_test_split"]


def test_honest_hotels_does_not_use_post_outcome_columns() -> None:
    constants = string_constants(parse(HONEST_HOTELS))
    for column in POST_OUTCOME_HOTEL_COLUMNS:
        assert column not in constants


def test_haunted_hotels_has_the_planted_leaks() -> None:
    constants = string_constants(parse(HAUNTED_HOTELS))
    assert "reservation_status" in constants
    assert "country_cancel_rate" in constants


def test_haunted_hotels_is_the_honest_script_plus_exactly_two_leaks() -> None:
    honest = string_constants(parse(HONEST_HOTELS))
    haunted = string_constants(parse(HAUNTED_HOTELS))
    assert honest <= haunted
    # "mean" is the aggregation of the country_cancel_rate leak, not a column name.
    assert haunted - honest == {"reservation_status", "country_cancel_rate", "mean"}
    ghosts = {
        ghost
        for comment in comments(HAUNTED_HOTELS)
        if comment.startswith(LEAK_COMMENT)
        for ghost in ("The Possession", "The Peeking Spirit")
        if ghost in comment
    }
    assert ghosts == {"The Possession", "The Peeking Spirit"}
    for comment in comments(HAUNTED_HOTELS):
        if comment.startswith(LEAK_COMMENT):
            assert "The Possession" in comment or "The Peeking Spirit" in comment


# --- the four copies of drop_seen_test_rows --------------------------------------


def drop_seen_test_rows_node(path: Path) -> ast.FunctionDef:
    found = [
        node
        for node in parse(path).body
        if isinstance(node, ast.FunctionDef) and node.name == "drop_seen_test_rows"
    ]
    assert len(found) == 1, f"{path} needs exactly one drop_seen_test_rows"
    return found[0]


def test_the_four_copies_of_drop_seen_test_rows_are_identical() -> None:
    dumps = {
        script.id: ast.dump(drop_seen_test_rows_node(script.values[0]))
        for script in ALL_SCRIPTS
    }
    assert len(set(dumps.values())) == 1, "the copies differ: " + ", ".join(dumps)


# --- the feature lists fit the prepared data -------------------------------------

DERIVED_BY_HAUNTED_SCRIPT = {
    HAUNTED_HOTELS: {"country_cancel_rate"},
    HAUNTED_FLIGHTS: {"carrier_day_late_rate"},
}


def prepared_columns(path: Path, hotels_fetch, flights_fetch, raw_hotels, raw_flights):
    if HOTELS in path.parts:
        return set(hotels_fetch.prepare(raw_hotels(40)).columns)
    delays = [float(10 * (i % 4)) for i in range(40)]
    return set(flights_fetch.prepare(raw_flights(delays)).columns)


def script_features(path: Path) -> set[str]:
    app = HOTELS if HOTELS in path.parts else FLIGHTS
    return set(script_module(app, haunted="haunted" in path.parts).FEATURES)


@pytest.mark.parametrize("path", HONEST_SCRIPTS)
def test_every_honest_feature_is_a_prepared_column(
    path: Path, hotels_fetch, flights_fetch, raw_hotels, raw_flights
) -> None:
    columns = prepared_columns(
        path, hotels_fetch, flights_fetch, raw_hotels, raw_flights
    )
    missing = script_features(path) - columns
    assert not missing, f"features that prepare() does not produce: {sorted(missing)}"


@pytest.mark.parametrize("path", HAUNTED_SCRIPTS)
def test_every_haunted_feature_is_prepared_or_derived_by_the_script(
    path: Path, hotels_fetch, flights_fetch, raw_hotels, raw_flights
) -> None:
    columns = prepared_columns(
        path, hotels_fetch, flights_fetch, raw_hotels, raw_flights
    )
    derived = DERIVED_BY_HAUNTED_SCRIPT[path]
    features = script_features(path)
    assert derived <= features
    assert not derived & columns, "a derived column must not already be prepared"
    missing = features - derived - columns
    assert not missing, f"features that prepare() does not produce: {sorted(missing)}"


def test_honest_flights_has_no_post_departure_column() -> None:
    constants = string_constants(parse(HONEST_FLIGHTS))
    for column in ("dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"):
        assert column not in constants


HOTEL_DATE_PARTS = (
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
)
DATE_CALLS = {
    "to_datetime",
    "Timestamp",
    "datetime",
    "date",
    "strptime",
    "make_datetime",
}


@pytest.mark.parametrize(
    "path", [HONEST_HOTELS, HAUNTED_HOTELS], ids=["honest", "haunted"]
)
def test_hotels_scripts_never_combine_year_month_and_day(path: Path) -> None:
    """The canary scrambles each column alone, so a combined date can be impossible."""
    tree = parse(path)
    assert not DATE_CALLS & {name.split(".")[-1] for name in call_names(tree)}
    # The three columns may be listed as features; main() must not touch them.
    in_main = string_constants(main_function(tree))
    assert not set(HOTEL_DATE_PARTS) & in_main


@pytest.mark.parametrize("path", HAUNTED_SCRIPTS)
def test_haunted_script_marks_its_leaks(path: Path) -> None:
    marked = [c for c in comments(path) if c.startswith(LEAK_COMMENT)]
    assert len(marked) >= 2


@pytest.mark.parametrize("path", HONEST_SCRIPTS)
def test_honest_script_has_no_leak_marker(path: Path) -> None:
    assert not [c for c in comments(path) if LEAK_COMMENT in c]


@pytest.mark.parametrize("path", HAUNTED_SCRIPTS)
def test_haunted_script_says_how_to_run_it(path: Path) -> None:
    header = path.read_text(encoding="utf-8").split("from __future__", 1)[0]
    assert "../train.py" in header


# --- flights ---------------------------------------------------------------------


def test_haunted_flights_splits_at_random_not_with_the_kit() -> None:
    names = call_names(parse(HAUNTED_FLIGHTS))
    assert [n for n in names if n.split(".")[-1] == "train_test_split"]
    assert "hexperiment.split" not in names


def test_haunted_flights_has_exactly_one_random_split_marker() -> None:
    lines = HAUNTED_FLIGHTS.read_text(encoding="utf-8").splitlines()
    assert sum(line.rstrip().endswith(RANDOM_SPLIT_MARKER) for line in lines) == 1


def test_haunted_flights_has_the_same_day_carrier_rate() -> None:
    assert "carrier_day_late_rate" in string_constants(parse(HAUNTED_FLIGHTS))


@pytest.mark.parametrize("path", FLIGHT_SCRIPTS)
def test_flights_scripts_never_combine_year_month_and_day(path: Path) -> None:
    tree = parse(path)
    assert "year" not in string_constants(tree)
    assert "to_datetime" not in {n.split(".")[-1] for n in call_names(tree)}


def test_split_fixed_source_of_the_haunted_flights_script(split_fixed_source) -> None:
    fixed = split_fixed_source(HAUNTED_FLIGHTS.read_text(encoding="utf-8"))
    names = call_names(ast.parse(fixed))
    assert "hexperiment.split" in names
    assert not [n for n in names if n.split(".")[-1] == "train_test_split"]
    assert RANDOM_SPLIT_MARKER not in fixed


# --- apps/verify.py: split_fixed_source (pure) ------------------------------------

MARKER = RANDOM_SPLIT_MARKER


@pytest.fixture
def split_fixed_source(verify_module: ModuleType):
    return verify_module.split_fixed_source


def test_split_fixed_source_replaces_only_the_marked_line(split_fixed_source) -> None:
    source = f"a = 1\n    train, test = other(df)   {MARKER}\nb = 2\n"
    fixed = split_fixed_source(source)
    assert fixed == "a = 1\n    train, test = hexperiment.split(df)\nb = 2\n"


@pytest.mark.parametrize("count", [0, 2])
def test_split_fixed_source_needs_exactly_one_marker(
    split_fixed_source, count: int
) -> None:
    source = f"x = 1  {MARKER}\n" * count + "y = 2\n"
    with pytest.raises(ValueError, match="exactly one"):
        split_fixed_source(source)
