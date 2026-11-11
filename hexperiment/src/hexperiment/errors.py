"""The single exception type for invalid input."""

from __future__ import annotations


class HexInputError(ValueError):
    """Invalid input. States what is wrong and how to fix it."""

    problem: str
    fix: str

    def __init__(self, problem: str, fix: str) -> None:
        super().__init__(f"{problem} {fix}")
        self.problem = problem
        self.fix = fix
