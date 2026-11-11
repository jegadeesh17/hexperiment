import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from sklearn.metrics import roc_auc_score

from hexperiment.checks.target_proxy import check_target_proxy, standalone_score
from hexperiment.config import SplitConfig
from hexperiment.data import split_frame
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


def test_a_numeric_score_ignores_rows_with_a_missing_value() -> None:
    train = pd.DataFrame({"f": [0.0, 1.0], "y": [0, 1]})
    test = pd.DataFrame(
        {"f": [0.1, 0.9, np.nan, np.nan], "y": [0, 1, 1, 0]},
    )
    assert score(train, test) == 1.0


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


def test_a_numeric_feature_with_one_class_left_after_dropping_missing_is_skipped() -> (
    None
):
    train = pd.DataFrame({"f": [0.0, 1.0], "y": [0, 1]})
    test = pd.DataFrame({"f": [0.1, 0.2, np.nan, np.nan], "y": [1, 1, 0, 0]})
    assert score(train, test) is None
    assert check(train, test, auc_max=0.5).outcome is Outcome.PASSED


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
    st.one_of(st.none(), st.integers(-3, 3).map(float)),
    st.sampled_from(["p", "q", "r", None]),
    st.integers(0, 1),
)


@given(
    train=st.lists(feature_values, min_size=2, max_size=25),
    test=st.lists(feature_values, min_size=2, max_size=25),
    auc_max=st.sampled_from([0.5, 0.75, 0.95]),
    order=st.randoms(use_true_random=False),
)
def test_reordering_rows_never_changes_the_result(
    train: list[tuple], test: list[tuple], auc_max: float, order
) -> None:
    def build(rows: list[tuple]) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["num", "cat", "y"])
        df["num"] = df["num"].astype(float)
        return df

    def run(train_rows: list[tuple], test_rows: list[tuple]):
        tr, te = build(train_rows), build(test_rows)
        scores = {f: score(tr, te, f) for f in ("num", "cat")}
        return scores, check(tr, te, ["num", "cat"], auc_max)

    expected = run(train, test)
    shuffled_train, shuffled_test = train[:], test[:]
    order.shuffle(shuffled_train)
    order.shuffle(shuffled_test)
    assert run(shuffled_train, shuffled_test) == expected
