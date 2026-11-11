import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment import HexInputError


def test_message_contains_problem_and_fix() -> None:
    error = HexInputError("data.path does not exist.", "Run fetch_data.py.")
    assert "data.path does not exist." in str(error)
    assert "Run fetch_data.py." in str(error)
    assert str(error) == "data.path does not exist. Run fetch_data.py."


def test_keeps_problem_and_fix_attributes() -> None:
    error = HexInputError("bad", "fix it")
    assert error.problem == "bad"
    assert error.fix == "fix it"


def test_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="bad fix it"):
        raise HexInputError("bad", "fix it")


def test_problem_and_fix_are_required() -> None:
    with pytest.raises(TypeError):
        HexInputError("only a problem")  # type: ignore[call-arg]


@given(st.text(), st.text())
def test_message_is_problem_then_fix(problem: str, fix: str) -> None:
    assert str(HexInputError(problem, fix)) == f"{problem} {fix}"
