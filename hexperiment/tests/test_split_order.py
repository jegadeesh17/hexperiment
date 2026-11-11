import textwrap

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment.canary import hint_locations
from hexperiment.checks.split_order import FIX, check_split_order
from hexperiment.models import CheckResult, DataAuditResult, Outcome, Severity


def script(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def check(text: str, name: str = "train.py") -> CheckResult:
    return check_split_order(script(text), name)


def lines_of(result: CheckResult) -> list[int]:
    return [int(f.location.rsplit(":", 1)[1]) for f in result.findings]


# --- leaking and clean cases (Requirements 8.1, 8.2, 17.1) ---------------------------

LEAKING = """
    import hexperiment
    from sklearn.preprocessing import StandardScaler

    df = hexperiment.load()
    df["x"] = df["x"].fillna(df["x"].mean())
    X = StandardScaler().fit_transform(df[["x"]])
    train, test = hexperiment.split(df)
"""


def test_a_mean_and_a_fit_before_the_split_are_warned_about() -> None:
    result = check(LEAKING)

    assert result.name == "split_order"
    assert result.ghost is None
    assert result.outcome is Outcome.PASSED  # warnings do not change the outcome
    mean, fit = result.findings
    assert (mean.location, fit.location) == ("train.py:5", "train.py:6")
    assert (mean.evidence, fit.evidence) == (
        {"call": "mean", "first_split_line": 7},
        {"call": "fit_transform", "first_split_line": 7},
    )
    for finding in result.findings:
        assert finding.check == "split_order"
        assert finding.ghost is None
        assert finding.severity is Severity.WARNING
        assert finding.fix == FIX
        assert finding.fix == (
            "this runs before the split; make sure it does not use test rows"
        )
    assert mean.message == "mean() is called before the first split"


def test_fits_after_the_split_are_clean() -> None:
    result = check(
        """
        import hexperiment
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = hexperiment.load()
        train, test = hexperiment.split(df)
        scaler = StandardScaler().fit(train[["x"]])
        fill = train["x"].mean()
        model = make_pipeline(StandardScaler()).fit(train[["x"]])
        """
    )
    assert result == CheckResult("split_order", None, Outcome.PASSED, ())


@pytest.mark.parametrize(
    "method",
    ["fit", "fit_transform", "fit_resample", "mean", "median", "std", "quantile"]
    + ["min", "max", "transform"],
)
def test_every_listed_method_is_found_before_the_split(method: str) -> None:
    result = check(
        f"""
        thing = df.x.{method}(1)
        train, test = split(df)
        """
    )
    (finding,) = result.findings
    assert finding.evidence == {"call": method, "first_split_line": 2}
    assert finding.location == "train.py:1"


def test_a_target_encoding_with_groupby_transform_is_found() -> None:
    result = check(
        """
        df["enc"] = df.groupby("category")["target"].transform("mean")
        train, test = hexperiment.split(df)
        """
    )
    # transform("mean") is one call, and the string argument is not a call.
    (finding,) = result.findings
    assert finding.evidence["call"] == "transform"
    assert finding.location == "train.py:1"


def test_statistics_on_series_expressions_of_any_shape_count() -> None:
    result = check(
        """
        a = df["x"].mean()
        b = df.x.std()
        c = df.groupby("g")["x"].median()
        d = pd.Series([1, 2]).max()
        e = (df["x"] + 1).min()
        train, test = split(df)
        """
    )
    assert lines_of(result) == [1, 2, 3, 4, 5]


def test_the_location_uses_the_script_name_it_is_given() -> None:
    result = check_split_order("m = df.x.mean()\ntrain, test = split(df)\n", "app.py")
    assert result.findings[0].location == "app.py:1"


# --- the split call is recognized (Requirement 8.1) ----------------------------------


@pytest.mark.parametrize(
    "call",
    [
        "split(df)",
        "hexperiment.split(df)",
        "train_test_split(X, y)",
        "model_selection.train_test_split(X, y)",
        "sklearn.model_selection.train_test_split(X, y)",
    ],
)
def test_each_split_spelling_is_recognized(call: str) -> None:
    result = check(f"m = df.x.mean()\ntrain, test = {call}\n")
    assert result.outcome is Outcome.PASSED
    assert lines_of(result) == [1]
    assert result.findings[0].evidence["first_split_line"] == 2


def test_a_split_method_of_another_object_is_not_a_split_call() -> None:
    result = check(
        """
        words = "a b".split()
        parts = path.split("/")
        """
    )
    assert result.outcome is Outcome.INCONCLUSIVE


def test_the_first_split_call_by_line_is_used() -> None:
    result = check(
        """
        a = df.x.mean()
        train, test = split(df)
        b = df.x.mean()
        more, rest = train_test_split(train)
        """
    )
    assert lines_of(result) == [1]
    assert result.findings[0].evidence["first_split_line"] == 2


def test_the_first_split_call_does_not_depend_on_the_order_of_the_tree() -> None:
    # A call inside a later function body is found by line number, not by depth.
    result = check(
        """
        def make():
            return train_test_split(a)

        m = df.x.mean()
        s = split(df)
        """
    )
    assert result.findings == ()  # the first split call is at line 2, before the mean


def test_an_earlier_statement_on_the_same_line_as_the_split_is_not_earlier() -> None:
    result = check("m = df.x.mean(); train, test = split(df)\n")
    assert result.findings == ()
    assert result.outcome is Outcome.PASSED


def test_a_multi_line_split_call_is_dated_by_its_first_line() -> None:
    result = check(
        """
        m = df.x.mean()
        train, test = hexperiment.split(
            df.assign(z=df.x.mean())
        )
        """
    )
    # Line 1 is before the split; line 3 is nested in its arguments.
    assert lines_of(result) == [1, 3]
    assert [f.evidence["first_split_line"] for f in result.findings] == [2, 2]


# --- import aliases (Requirement 8.1) ------------------------------------------------

ALIASED_SPLITS = {
    "import hexperiment as hx": ("import hexperiment as hx", "hx.split(df)"),
    "from hexperiment import split as hsplit": (
        "from hexperiment import split as hsplit",
        "hsplit(df)",
    ),
    "from sklearn.model_selection import train_test_split as tts": (
        "from sklearn.model_selection import train_test_split as tts",
        "tts(X, y)",
    ),
    "import sklearn.model_selection as ms": (
        "import sklearn.model_selection as ms",
        "ms.train_test_split(X, y)",
    ),
    "from sklearn import model_selection as ms": (
        "from sklearn import model_selection as ms",
        "ms.train_test_split(X, y)",
    ),
    "from hexperiment import split": ("from hexperiment import split", "split(df)"),
    "import hexperiment": ("import hexperiment", "hexperiment.split(df)"),
}


@pytest.mark.parametrize(
    ("statement", "call"), ALIASED_SPLITS.values(), ids=ALIASED_SPLITS.keys()
)
def test_an_aliased_split_call_is_the_first_split(statement: str, call: str) -> None:
    result = check(f"{statement}\nm = df.x.mean()\ntrain, test = {call}\n")
    assert result.outcome is Outcome.PASSED
    assert lines_of(result) == [2]
    assert result.findings[0].evidence == {"call": "mean", "first_split_line": 3}


@pytest.mark.parametrize(
    ("statement", "call"),
    [
        ("import hexperiment as hx", "hx.split(df)"),
        ("from hexperiment import split as hsplit", "hsplit(df)"),
        ("from sklearn.model_selection import train_test_split as tts", "tts(X)"),
    ],
)
def test_an_aliased_split_hides_nothing_after_it(statement: str, call: str) -> None:
    result = check(f"{statement}\ntrain, test = {call}\nm = train.x.mean()\n")
    assert result.findings == ()


@pytest.mark.parametrize(
    ("statement", "call"),
    [
        ("import hexperiment as hx", "hx.load()"),
        ("from hexperiment import load as hload", "hload()"),
        ("from re import split as rsplit", "rsplit(',', text)"),
        ("import numpy as np", "np.split(values, 3)"),
    ],
)
def test_an_alias_of_something_other_than_a_split_is_not_a_split(
    statement: str, call: str
) -> None:
    result = check(f"{statement}\nm = df.x.mean()\ndata = {call}\n")
    assert result.outcome is Outcome.INCONCLUSIVE


# --- calls nested in the arguments of the split call (Requirement 8.2) ---------------


def test_a_call_nested_in_the_split_arguments_is_reported_like_an_earlier_call() -> (
    None
):
    result = check("train, test = split(df.fillna(df.mean()))\n")
    (finding,) = result.findings
    assert finding.location == "train.py:1"
    assert finding.evidence == {"call": "mean", "first_split_line": 1}
    assert finding.severity is Severity.WARNING
    assert finding.fix == FIX
    assert result.outcome is Outcome.PASSED


@pytest.mark.parametrize(
    "call",
    [
        "split(df.fillna(df.mean()))",
        "hx.split(df.fillna(df.mean()))",
        "hexperiment.split(df.fillna(df.mean()))",
        "tts(df.fillna(df.mean()))",
        "split(df, weights=df.x.std())",
        "split(prep(Scaler().fit_transform(df)))",
        "split(df.assign(z=(df.x - df.x.mean()) / df.x.std()))",
    ],
)
def test_nested_suspects_are_found_for_every_split_spelling(call: str) -> None:
    result = check(
        "import hexperiment as hx\n"
        "from sklearn.model_selection import train_test_split as tts\n"
        f"train, test = {call}\n"
    )
    assert result.findings
    assert set(lines_of(result)) == {3}
    assert all(f.evidence["first_split_line"] == 3 for f in result.findings)


def test_nested_and_earlier_calls_are_reported_in_line_order() -> None:
    result = check(
        """
        m = df.x.mean()
        train, test = split(
            df.fillna(df.mean()),
            weights=df.x.std(),
        )
        """
    )
    assert lines_of(result) == [1, 3, 4]
    assert [f.evidence["call"] for f in result.findings] == ["mean", "mean", "std"]


def test_harmless_nested_calls_are_not_reported() -> None:
    result = check("train, test = split(df.dropna().reset_index(drop=True))\n")
    assert result.findings == ()


def test_calls_in_the_arguments_of_a_later_split_are_not_reported() -> None:
    result = check(
        """
        train, test = split(df)
        more, rest = split(train.fillna(train.mean()))
        """
    )
    assert result.findings == ()


# --- calls that are ignored (Requirement 8.2) ----------------------------------------


def test_numpy_module_functions_are_ignored() -> None:
    result = check(
        """
        import numpy as np
        import scipy.stats
        a = np.mean(values)
        b = np.max(values)
        c = np.random.std(5)
        d = scipy.stats.median(values)
        train, test = split(df)
        """
    )
    assert result.findings == ()


def test_a_module_alias_does_not_hide_a_method_of_a_frame() -> None:
    result = check(
        """
        import numpy as np
        a = np.mean(values)
        b = df["x"].mean()
        c = df.x.mean()
        train, test = split(df)
        """
    )
    assert lines_of(result) == [3, 4]


def test_a_name_that_is_not_an_imported_module_is_a_receiver() -> None:
    result = check("m = np.mean(values)\ntrain, test = split(df)\n")
    assert lines_of(result) == [1]  # np is not imported here, so it is a name


def test_builtin_min_and_max_are_ignored() -> None:
    result = check(
        """
        a = min(a, b)
        c = max(1, 2, 3)
        train, test = split(df)
        """
    )
    assert result.findings == ()


def test_bare_fit_style_names_are_not_method_calls() -> None:
    result = check("fit(x)\nmean(x)\ntrain, test = split(df)\n")
    assert result.findings == ()


# --- inconclusive: no split call, bad source (Requirement 8.3) -----------------------


def test_no_split_call_gives_one_inconclusive_warning() -> None:
    result = check("m = df.x.mean()\nscaler.fit(df)\n")
    assert result.outcome is Outcome.INCONCLUSIVE
    assert result.ghost is None
    (finding,) = result.findings
    assert finding.check == "split_order"
    assert finding.severity is Severity.WARNING
    assert finding.location == "train.py"
    assert "could not be applied" in finding.message
    assert "no call to a split function" in finding.message
    assert finding.evidence == {}
    assert "hexperiment.split" in finding.fix


def test_a_syntax_error_gives_one_inconclusive_warning() -> None:
    result = check_split_order("def broken(:\n    pass\n", "train.py")
    assert result.outcome is Outcome.INCONCLUSIVE
    (finding,) = result.findings
    assert finding.severity is Severity.WARNING
    assert "could not be parsed" in finding.message


def test_a_source_with_null_bytes_gives_one_inconclusive_warning() -> None:
    result = check_split_order("x = 1\x00\n", "train.py")
    assert result.outcome is Outcome.INCONCLUSIVE
    assert len(result.findings) == 1


def test_unavailable_source_gives_one_inconclusive_warning() -> None:
    result = check_split_order(None, "train.py")
    assert result.outcome is Outcome.INCONCLUSIVE
    (finding,) = result.findings
    assert "could not be read" in finding.message
    assert finding.location == "train.py"


def test_an_empty_script_gives_one_inconclusive_warning() -> None:
    assert check_split_order("", "train.py").outcome is Outcome.INCONCLUSIVE


def test_the_script_is_parsed_and_never_executed() -> None:
    marker = "raise SystemExit('executed')\nimport nonexistent_module_xyz\n"
    result = check_split_order(marker + "train, test = split(df)\n", "train.py")
    assert result.outcome is Outcome.PASSED
    assert result.findings == ()


# --- the canary reads the hint (Requirement 10.5) ------------------------------------


def test_the_canary_lists_the_hint_lines_but_not_the_inconclusive_warning() -> None:
    hint = check(LEAKING)
    result = DataAuditResult(None, None, (hint,), (), None, None)
    assert hint_locations(result) == ("train.py:5", "train.py:6")
    inconclusive = DataAuditResult(None, None, (check("x = 1\n"),), (), None, None)
    assert hint_locations(inconclusive) == ()


# --- properties ----------------------------------------------------------------------

SUSPECTS = [
    "m = df.x.mean()",
    "s = scaler.fit_transform(df)",
    "d = df['x'].std()",
    "enc.fit(df)",
]
HARMLESS = ["a = 1", "b = np.mean(v)", "c = min(a, b)", "# note", "", "d = fit(x)"]


@given(
    before=st.lists(st.sampled_from(SUSPECTS + HARMLESS), max_size=15),
    after=st.lists(st.sampled_from(SUSPECTS + HARMLESS), max_size=15),
)
def test_exactly_the_suspect_lines_before_the_split_are_reported(
    before: list[str], after: list[str]
) -> None:
    source = "\n".join(
        ["import numpy as np", *before, "train, test = split(df)", *after, ""]
    )
    result = check_split_order(source, "train.py")

    expected = [i + 2 for i, line in enumerate(before) if line in SUSPECTS]
    assert result.outcome is Outcome.PASSED
    assert lines_of(result) == expected
    assert all(f.severity is Severity.WARNING for f in result.findings)
    split_line = len(before) + 2
    assert all(f.evidence["first_split_line"] == split_line for f in result.findings)


@given(text=st.text(max_size=200))
def test_any_text_gives_a_result_with_warnings_only(text: str) -> None:
    result = check_split_order(text, "train.py")
    assert result.outcome in (Outcome.PASSED, Outcome.INCONCLUSIVE)
    assert all(f.severity is Severity.WARNING for f in result.findings)
    if result.outcome is Outcome.INCONCLUSIVE:
        assert len(result.findings) == 1


# --- direct data reads (Requirement 8.4) ---------------------------------------------

READER_MESSAGE = (
    "this reads data without hexperiment.load(); the canary test cannot scramble "
    "this copy"
)
READER_FIX = (
    "load the data with hexperiment.load() and derive everything from that frame"
)
READERS = [
    "read_csv",
    "read_parquet",
    "read_excel",
    "read_json",
    "read_feather",
    "read_table",
    "read_fwf",
    "read_pickle",
    "read_sql",
    "read_orc",
    "loadtxt",
    "genfromtxt",
]


def test_pandas_read_csv_in_the_script_warns_with_its_line() -> None:
    result = check(
        """
        import pandas as pd
        import hexperiment

        df = pd.read_csv("data.csv")
        train, test = hexperiment.split(df)
        """
    )
    assert result.outcome is Outcome.PASSED
    (finding,) = result.findings
    assert finding.location == "train.py:4"
    assert finding.check == "split_order"
    assert finding.ghost is None
    assert finding.severity is Severity.WARNING
    assert finding.message == READER_MESSAGE
    assert finding.fix == READER_FIX
    assert finding.evidence == {"call": "read_csv"}


@pytest.mark.parametrize("reader", READERS)
def test_every_listed_reader_warns(reader: str) -> None:
    result = check(
        f"""
        data = lib.{reader}("x")
        train, test = split(data)
        """
    )
    (finding,) = result.findings
    assert (finding.location, finding.evidence) == ("train.py:1", {"call": reader})


def test_a_bare_reader_name_and_an_import_alias_warn() -> None:
    result = check(
        """
        from pandas import read_csv, read_json as rj

        a = read_csv("a.csv")
        b = rj("b.json")
        train, test = split(a)
        """
    )
    assert lines_of(result) == [3, 4]
    assert [f.evidence["call"] for f in result.findings] == ["read_csv", "read_json"]


def test_a_reader_after_the_split_still_warns() -> None:
    result = check(
        """
        import hexperiment

        train, test = hexperiment.split(hexperiment.load())
        extra = open_data().read_csv("x.csv")
        """
    )
    assert lines_of(result) == [4]
    assert result.outcome is Outcome.PASSED


def test_hexperiment_load_is_not_a_reader() -> None:
    result = check(
        """
        import hexperiment
        from hexperiment import load

        df = hexperiment.load()
        other = load()
        train, test = hexperiment.split(df)
        """
    )
    assert result == CheckResult("split_order", None, Outcome.PASSED, ())


def test_a_script_with_no_split_call_still_gets_the_reader_warning() -> None:
    result = check(
        """
        import pandas as pd

        df = pd.read_csv("data.csv")
        """
    )
    assert result.outcome is Outcome.INCONCLUSIVE
    unapplied, reader = result.findings
    assert "could not be applied" in unapplied.message
    assert unapplied.location == "train.py"
    assert (reader.location, reader.message) == ("train.py:3", READER_MESSAGE)


def test_reader_and_hint_warnings_are_listed_in_line_order() -> None:
    result = check(
        """
        import pandas as pd

        df = pd.read_csv("data.csv")
        df["x"] = df["x"].fillna(df["x"].mean())
        extra = pd.read_json("more.json")
        train, test = split(df)
        """
    )
    assert lines_of(result) == [3, 4, 5]
    assert [f.message for f in result.findings][0] == READER_MESSAGE


def test_an_unparseable_script_has_no_reader_warnings() -> None:
    result = check("df = pd.read_csv(\n")
    (finding,) = result.findings
    assert "could not be applied" in finding.message


@given(
    reader=st.sampled_from(READERS),
    padding=st.integers(0, 5),
    has_split=st.booleans(),
)
def test_a_reader_is_reported_at_its_line_wherever_the_split_is(
    reader: str, padding: int, has_split: bool
) -> None:
    lines = ["import pandas as pd"] * padding + [f"df = pd.{reader}('x')"]
    if has_split:
        lines.append("train, test = split(df)")
    result = check_split_order("\n".join(lines) + "\n", "train.py")
    readers = [f for f in result.findings if f.message == READER_MESSAGE]
    assert [f.location for f in readers] == [f"train.py:{padding + 1}"]
    assert (result.outcome is Outcome.PASSED) == has_split
