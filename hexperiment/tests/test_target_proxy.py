import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sklearn.metrics import roc_auc_score

from hexperiment.checks.target_proxy import check_target_proxy, standalone_score
from hexperiment.config import SplitConfig
from hexperiment.data import split_frame
from hexperiment.errors import HexInputError
from hexperiment.models import CheckResult, Outcome, Severity


def labels(frame: pd.DataFrame) -> np.ndarray:
    return (frame["y"] == 1).to_numpy(dtype=bool)


def score(train: pd.DataFrame, test: pd.DataFrame, feature: str = "f"):
    return standalone_score(train, test, feature, "y", labels(train), labels(test))


def check(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] | None = None,
    auc_max: float = 0.95,
) -> CheckResult:
    return check_target_proxy(
        train,
        test,
        features if features is not None else ["f"],
        "y",
        labels(train),
        labels(test),
        auc_max,
    )


def noisy_frames(n: int = 300, seed: int = 0, noise: float = 0.0):
    """Frames where ``leak`` follows the target and ``clean`` is independent noise."""
    gen = np.random.default_rng(seed)
    y = gen.integers(0, 2, size=2 * n)
    frame = pd.DataFrame(
        {
            "leak": y + gen.normal(scale=noise, size=2 * n),
            "word": np.where(y == 1, "yes", "no"),
            "clean": gen.normal(size=2 * n),
            "kind": gen.choice(["p", "q", "r"], size=2 * n),
            "y": y,
        }
    )
    return frame.iloc[:n].copy(), frame.iloc[n:].copy()


# --- leaking and clean cases (Requirements 6.1, 6.2, 17.1) ---------------------------


def test_a_numeric_feature_that_follows_the_target_is_the_possession() -> None:
    train, test = noisy_frames(noise=0.01)
    result = check(train, test, ["leak", "clean"])

    assert result.name == "target_proxy"
    assert result.ghost == "The Possession"
    assert result.outcome is Outcome.FAILED
    (finding,) = result.findings
    assert finding.check == "target_proxy"
    assert finding.ghost == "The Possession"
    assert finding.severity is Severity.ERROR
    assert finding.location == "leak"
    assert finding.evidence["score"] == 1.0
    assert finding.evidence["threshold"] == 0.95
    assert finding.fix == (
        "check whether this column is recorded after the outcome; if so, drop it"
    )
    assert "leak" in finding.message
    assert "1.0000" in finding.message


def test_a_category_derived_from_the_target_is_the_possession() -> None:
    train, test = noisy_frames()
    assert score(train, test, "word") == 1.0
    (finding,) = check(train, test, ["word"]).findings
    assert finding.location == "word"


def test_independent_features_pass() -> None:
    train, test = noisy_frames()
    result = check(train, test, ["clean", "kind"])
    assert result == CheckResult("target_proxy", "The Possession", Outcome.PASSED, ())


def test_each_leaking_feature_gets_its_own_finding_in_feature_order() -> None:
    train, test = noisy_frames()
    result = check(train, test, ["word", "clean", "leak"])
    assert [f.location for f in result.findings] == ["word", "leak"]


def test_a_feature_equal_to_the_target_is_flagged_even_when_it_is_the_target() -> None:
    train, test = noisy_frames()
    assert [f.location for f in check(train, test, ["y"]).findings] == ["y"]


# --- the standalone score ------------------------------------------------------------


def test_an_inverted_numeric_feature_scores_like_the_direct_one() -> None:
    train, test = noisy_frames()
    test = test.assign(f=1 - test["y"])
    train = train.assign(f=1 - train["y"])
    assert score(train, test) == 1.0


def test_a_numeric_score_is_the_larger_of_auc_and_one_minus_auc() -> None:
    gen = np.random.default_rng(3)
    train, test = noisy_frames(n=200)
    test = test.assign(f=gen.normal(size=len(test)) - 0.8 * test["y"])
    auc = roc_auc_score(test["y"], test["f"])
    assert auc < 0.5
    assert score(train.assign(f=0.0), test) == pytest.approx(1 - auc)


def test_a_numeric_score_counts_rows_with_a_missing_value_as_one_tied_group() -> None:
    # Present values 0.1 (class 0) and 0.9 (class 1); the two missing rows, one of
    # each class, tie at the bottom or at the top. Both rankings give 2.5 / 4.
    train = pd.DataFrame({"f": [0.0, 1.0], "y": [0, 1]})
    test = pd.DataFrame(
        {"f": [0.1, 0.9, np.nan, np.nan], "y": [0, 1, 1, 0]},
    )
    assert score(train, test) == 0.625


def test_infinite_values_are_ordinary_extremes() -> None:
    train = pd.DataFrame({"f": [0.0, 1.0], "y": [0, 1]})
    test = pd.DataFrame({"f": [-np.inf, np.inf, -np.inf, np.inf], "y": [0, 1, 0, 1]})
    assert score(train, test) == 1.0


def test_boolean_features_are_scored_as_numbers() -> None:
    train = pd.DataFrame({"f": [True, False, True, False], "y": [1, 0, 1, 0]})
    test = pd.DataFrame({"f": [True, False, True, False], "y": [1, 0, 1, 0]})
    assert score(train, test) == 1.0


def test_nullable_integer_features_with_missing_values_are_numeric() -> None:
    train = pd.DataFrame({"f": pd.array([0, 1], dtype="Int64"), "y": [0, 1]})
    test = pd.DataFrame(
        {"f": pd.array([0, 1, None, 1], dtype="Int64"), "y": [0, 1, 0, 1]}
    )
    assert score(train, test) == 1.0


def test_unseen_categories_get_the_training_positive_rate() -> None:
    # Training rates: a = 1.0, b = 0.0, overall 0.5. The test scores are
    # a -> 1, b -> 0, unseen z -> 0.5 twice, so AUC = 3.5 / 4.
    train = pd.DataFrame({"f": list("aaaabbbb"), "y": [1] * 4 + [0] * 4})
    test = pd.DataFrame({"f": list("abzz"), "y": [1, 0, 1, 0]})
    assert score(train, test) == 0.875


def test_missing_categories_get_the_training_positive_rate_too() -> None:
    train = pd.DataFrame({"f": list("aaaabbbb"), "y": [1] * 4 + [0] * 4})
    test = pd.DataFrame({"f": ["a", "b", None, np.nan], "y": [1, 0, 1, 0]})
    assert score(train, test) == 0.875


def test_missing_training_categories_are_left_out_of_the_rates() -> None:
    train = pd.DataFrame({"f": ["a", "a", "b", None, None], "y": [1, 1, 0, 0, 0]})
    test = pd.DataFrame({"f": ["a", "b", "a", "b"], "y": [1, 0, 1, 0]})
    assert score(train, test) == 1.0


def test_all_unseen_categories_score_one_half() -> None:
    train = pd.DataFrame({"f": list("abab"), "y": [1, 0, 1, 0]})
    test = pd.DataFrame({"f": list("xyxy"), "y": [1, 0, 1, 0]})
    assert score(train, test) == 0.5


def test_a_categorical_dtype_is_scored_by_category() -> None:
    train = pd.DataFrame(
        {"f": pd.Categorical(list("aabb"), categories=list("abc")), "y": [1, 1, 0, 0]}
    )
    test = pd.DataFrame(
        {"f": pd.Categorical(list("abab"), categories=list("abc")), "y": [1, 0, 1, 0]}
    )
    assert score(train, test) == 1.0


def test_a_category_score_can_fall_below_one_half() -> None:
    # The training relation is reversed in the test rows: the AUC is not folded.
    train = pd.DataFrame({"f": list("aabb"), "y": [1, 1, 0, 0]})
    test = pd.DataFrame({"f": list("abab"), "y": [0, 1, 0, 1]})
    assert score(train, test) == 0.0


# --- boundaries ----------------------------------------------------------------------


def test_a_score_equal_to_auc_max_is_flagged() -> None:
    train, test = noisy_frames(n=400, noise=1.5)
    train, test = train.assign(f=train["leak"]), test.assign(f=test["leak"])
    exact = score(train, test)
    assert exact is not None
    assert 0.5 < exact < 0.95

    flagged = check(train, test, auc_max=exact)
    assert flagged.outcome is Outcome.FAILED
    assert flagged.findings[0].location == "f"

    just_above = float(np.nextafter(exact, 2.0))
    assert check(train, test, auc_max=just_above).outcome is Outcome.PASSED


def test_a_constant_test_feature_is_skipped() -> None:
    train, test = noisy_frames()
    train = train.assign(f_num=train["clean"], f_cat=train["kind"])
    test = test.assign(f_num=1.5, f_cat="p")
    assert score(train, test, "f_num") is None
    assert score(train, test, "f_cat") is None
    # Without the skip a constant feature would score 0.5 and trip auc_max = 0.5.
    result = check(train, test, ["f_num", "f_cat"], auc_max=0.5)
    assert result.outcome is Outcome.PASSED
    assert result.findings == ()


def test_a_test_feature_that_is_always_missing_is_skipped() -> None:
    train = pd.DataFrame({"f": [0.0, 1.0], "y": [0, 1]})
    test = pd.DataFrame({"f": [np.nan] * 4, "y": [0, 1, 0, 1]})
    assert score(train, test) is None


def test_a_numeric_feature_with_a_missing_value_for_each_class_is_not_skipped() -> None:
    # Requirement 6.3: the missing rows are the whole class 0, so the feature
    # separates the classes and scores 1.0 instead of being skipped.
    train = pd.DataFrame({"f": [0.0, 1.0], "y": [0, 1]})
    test = pd.DataFrame({"f": [0.1, 0.2, np.nan, np.nan], "y": [1, 1, 0, 0]})
    assert score(train, test) == 1.0
    (finding,) = check(train, test).findings
    assert finding.location == "f"
    assert finding.evidence["score"] == 1.0


@pytest.mark.parametrize("filled_class", [1, 0], ids=["lowest", "highest"])
def test_a_feature_that_is_missing_for_one_class_only_is_flagged_either_way(
    filled_class: int,
) -> None:
    # Missing rows are the other class, ranked lowest (class 1 filled in) or
    # highest (class 0 filled in); both rankings are tried (Requirement 6.1).
    train, test = noisy_frames(seed=4)

    def fill(frame: pd.DataFrame) -> pd.Series:
        return frame["clean"].where(frame["y"] == filled_class)

    result = check(train.assign(f=fill(train)), test.assign(f=fill(test)))
    assert [f.location for f in result.findings] == ["f"]
    assert result.findings[0].evidence["score"] == 1.0


def test_partly_missing_independent_noise_is_not_flagged() -> None:
    train, test = noisy_frames(n=400, seed=5)
    gen = np.random.default_rng(50)

    def holes(frame: pd.DataFrame) -> pd.Series:
        return frame["clean"].mask(gen.random(len(frame)) < 0.3)

    train, test = train.assign(f=holes(train)), test.assign(f=holes(test))
    assert test["f"].isna().any()
    assert score(train, test) < 0.65
    assert check(train, test).findings == ()


def test_a_single_class_test_target_is_an_input_error_not_a_skip() -> None:
    train = pd.DataFrame({"f": [0.0, 1.0, 2.0], "y": [0, 1, 0]})
    test = pd.DataFrame({"f": [0.0, 1.0, 2.0], "y": [1, 1, 1]})
    with pytest.raises(HexInputError, match="single class"):
        score(train, test)


# --- datetime and timedelta features (Requirement 6.1) --------------------------------


def day_offsets(frame: pd.DataFrame, seed: int) -> pd.Series:
    """Days 0 to 19 for class 0 and days 30 to 49 for class 1."""
    gen = np.random.default_rng(seed)
    days = frame["y"].to_numpy() * 30 + gen.integers(0, 20, size=len(frame))
    return pd.Series(days, index=frame.index, dtype="float64")


@pytest.mark.parametrize("tz", [None, "UTC", "Europe/Paris", "Asia/Kolkata"])
def test_a_datetime_that_separates_the_classes_is_flagged(tz: str | None) -> None:
    train, test = noisy_frames(seed=6)

    def stamps(frame: pd.DataFrame) -> pd.Series:
        when = pd.Timestamp("2024-01-01") + pd.to_timedelta(day_offsets(frame, 1), "D")
        return when if tz is None else when.dt.tz_localize(tz)

    train, test = train.assign(f=stamps(train)), test.assign(f=stamps(test))
    assert (test["f"].dt.tz is None) == (tz is None)
    assert score(train, test) == 1.0
    assert [f.location for f in check(train, test).findings] == ["f"]


def test_a_datetime_with_a_missing_value_for_one_class_is_flagged() -> None:
    train, test = noisy_frames(seed=6)

    def stamps(frame: pd.DataFrame) -> pd.Series:
        when = pd.Timestamp("2024-01-01") + pd.to_timedelta(day_offsets(frame, 2), "D")
        return when.where(frame["y"] == 1)  # NaT for class 0

    train, test = train.assign(f=stamps(train)), test.assign(f=stamps(test))
    assert test["f"].isna().any()
    assert score(train, test) == 1.0


def test_datetimes_one_nanosecond_apart_are_still_ordered() -> None:
    # Float64 cannot tell these apart (they are about 1.7e18 ns from the epoch).
    base = pd.Timestamp("2024-01-01").as_unit("ns")
    stamps = [base + pd.Timedelta(nanoseconds=i % 2) for i in range(8)]
    frame = pd.DataFrame({"f": stamps, "y": [0, 1] * 4})
    assert score(frame, frame) == 1.0


def test_a_timedelta_that_separates_the_classes_is_flagged() -> None:
    train, test = noisy_frames(seed=6)

    def gaps(frame: pd.DataFrame) -> pd.Series:
        return pd.to_timedelta(day_offsets(frame, 3), "D")

    train, test = train.assign(f=gaps(train)), test.assign(f=gaps(test))
    assert test["f"].dtype.kind == "m"
    assert score(train, test) == 1.0
    assert [f.location for f in check(train, test).findings] == ["f"]


def test_an_independent_datetime_is_not_flagged() -> None:
    train, test = noisy_frames(n=400, seed=8)
    gen = np.random.default_rng(80)

    def stamps(frame: pd.DataFrame) -> pd.Series:
        days = pd.Series(gen.integers(0, 365, size=len(frame)), index=frame.index)
        return pd.Timestamp("2024-01-01") + pd.to_timedelta(days, "D")

    train, test = train.assign(f=stamps(train)), test.assign(f=stamps(test))
    assert check(train, test).findings == ()


# --- numeric codes (Requirement 6.1) --------------------------------------------------


def coded_frames(
    positive: list[int], negative: list[int], n: int = 200, seed: int = 9
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gen = np.random.default_rng(seed)

    def build() -> pd.DataFrame:
        y = gen.integers(0, 2, size=n)
        f = np.where(y == 1, gen.choice(positive, size=n), gen.choice(negative, size=n))
        return pd.DataFrame({"f": f, "y": y})

    return build(), build()


def test_integer_codes_1_and_3_positive_and_2_and_4_negative_are_flagged() -> None:
    train, test = coded_frames([1, 3], [2, 4])
    # As ordered values the codes only reach AUC 0.75; the category score is 1.0.
    assert _value_only_score(test) < 0.95
    assert score(train, test) == 1.0
    assert [f.location for f in check(train, test).findings] == ["f"]


def _value_only_score(test: pd.DataFrame) -> float:
    auc = roc_auc_score(test["y"], test["f"])
    return max(auc, 1 - auc)


@pytest.mark.parametrize(("levels", "flagged"), [(50, True), (51, False)])
def test_the_category_score_applies_up_to_50_distinct_numeric_values(
    levels: int, flagged: bool
) -> None:
    # The class is the parity of the code, so the order of the codes says little.
    codes = np.tile(np.arange(levels), 6)
    frame = pd.DataFrame({"f": codes, "y": codes % 2})
    assert _value_only_score(frame) < 0.6
    assert (check(frame, frame).outcome is Outcome.FAILED) is flagged


@pytest.mark.parametrize("numeric", [True, False], ids=["numeric", "string"])
def test_a_high_cardinality_id_column_is_not_flagged(numeric: bool) -> None:
    train, test = noisy_frames(n=300, seed=7)
    ids = np.random.default_rng(70).permutation(2 * len(train))

    def column(positions: slice) -> pd.Series | list[str]:
        chosen = ids[positions]
        return chosen if numeric else [f"user-{i}" for i in chosen]

    train = train.assign(f=column(slice(0, len(train))))
    test = test.assign(f=column(slice(len(train), None)))
    assert test["f"].nunique() == len(test)
    assert score(train, test) < 0.6
    assert check(train, test).findings == ()


def test_a_feature_with_a_low_score_is_not_flagged() -> None:
    train, test = noisy_frames(noise=3.0)
    train, test = train.assign(f=train["leak"]), test.assign(f=test["leak"])
    assert 0.5 < score(train, test) < 0.95
    assert check(train, test).findings == ()


def test_the_arguments_are_not_modified() -> None:
    train, test = noisy_frames()
    y_train, y_test = labels(train), labels(test)
    before = (
        train.copy(deep=True),
        test.copy(deep=True),
        y_train.copy(),
        y_test.copy(),
    )
    features = ["leak", "word", "kind"]
    check_target_proxy(train, test, features, "y", y_train, y_test, 0.95)
    pd.testing.assert_frame_equal(train, before[0])
    pd.testing.assert_frame_equal(test, before[1])
    assert np.array_equal(y_train, before[2])
    assert np.array_equal(y_test, before[3])


# --- a clean synthetic dataset (Requirement 17.4) ------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_independent_random_features_have_no_error(seed: int) -> None:
    gen = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "n1": gen.normal(size=2000),
            "n2": gen.integers(0, 50, size=2000),
            "c1": gen.choice(list("abcde"), size=2000),
            "c2": gen.choice(["u", "v"], size=2000),
            "b": gen.random(2000) < 0.3,
            "y": gen.integers(0, 2, size=2000),
        }
    )
    train, test = split_frame(df, SplitConfig(seed=seed))
    result = check(train, test, ["n1", "n2", "c1", "c2", "b"])
    assert not [f for f in result.findings if f.severity is Severity.ERROR]
    assert result.outcome is Outcome.PASSED


# --- properties (Requirements 17.2, 6.4) ---------------------------------------------


@given(
    test_y=st.lists(st.integers(0, 1), min_size=2, max_size=30),
    train_y=st.lists(st.integers(0, 1), min_size=2, max_size=30),
    as_text=st.booleans(),
)
def test_a_feature_equal_to_the_target_is_always_detected(
    test_y: list[int], train_y: list[int], as_text: bool
) -> None:
    assume(0 in test_y and 1 in test_y)  # both classes, so two unique values
    assume(0 in train_y and 1 in train_y)  # a text label needs both seen in train

    def build(values: list[int]) -> pd.DataFrame:
        feature = ["pos" if v else "neg" for v in values] if as_text else values
        return pd.DataFrame({"f": feature, "y": values})

    train, test = build(train_y), build(test_y)
    result = check(train, test, auc_max=0.95)
    assert result.outcome is Outcome.FAILED
    (finding,) = result.findings
    assert finding.location == "f"
    assert finding.evidence["score"] == 1.0


feature_values = st.tuples(
    st.one_of(st.none(), st.integers(-3, 3).map(float)),  # num, with missing
    st.sampled_from(["p", "q", "r", None]),  # cat, with missing
    st.one_of(st.none(), st.integers(1, 4)),  # code, nullable integer
    st.one_of(st.none(), st.integers(0, 4)),  # day, for the datetime columns
    st.one_of(st.none(), st.integers(0, 4)),  # gap, for the timedelta column
    st.integers(0, 1),
)
REORDER_FEATURES = ["num", "cat", "code", "when", "when_tz", "gap"]


def reorder_frame(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["num", "cat", "code", "day", "gap_days", "y"])
    days = df.pop("day").astype("float64")
    when = pd.Timestamp("2024-01-01") + pd.to_timedelta(days, "D")
    df["num"] = df["num"].astype(float)
    df["code"] = df["code"].astype("Int64")
    df["when"] = when
    df["when_tz"] = when.dt.tz_localize("Asia/Kolkata")
    df["gap"] = pd.to_timedelta(df.pop("gap_days").astype("float64"), "D")
    return df


@settings(max_examples=30, deadline=None)
@given(
    train=st.lists(feature_values, min_size=2, max_size=25),
    test=st.lists(feature_values, min_size=2, max_size=25),
    order=st.randoms(use_true_random=False),
)
def test_reordering_rows_never_changes_the_result(
    train: list[tuple], test: list[tuple], order
) -> None:
    assume(0 in {row[-1] for row in test} and 1 in {row[-1] for row in test})

    def run(train_rows: list[tuple], test_rows: list[tuple]):
        tr, te = reorder_frame(train_rows), reorder_frame(test_rows)
        # With auc_max = 0 every scored feature is reported, so the findings carry
        # the score of each feature and a skipped feature is the one that is absent.
        return check(tr, te, REORDER_FEATURES, auc_max=0.0)

    expected = run(train, test)
    shuffled_train, shuffled_test = train[:], test[:]
    order.shuffle(shuffled_train)
    order.shuffle(shuffled_test)
    assert run(shuffled_train, shuffled_test) == expected
