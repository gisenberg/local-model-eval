"""Expression evaluator using a recursive‑descent parser.

Supports +, -, *, /, parentheses, unary minus and floating‑point numbers.
Raises ValueError for malformed input, division by zero or empty strings.

Typical usage
-------------
>>> evaluator = ExpressionEvaluator()
>>> evaluator.evaluate("3 + 4 * (2 - 1)")
7.0
"""

from __future__ import annotations

import re
from typing import List


class ExpressionEvaluator:
    """Recursive‑descent parser for simple arithmetic expressions."""

    _NUMBER_RE = re.compile(r"""^
        (?:\d+\.\d*|\.\d+|\d+)   # 123, 123.45, .45
        (?=[^\d\.]|$)            # stop before next digit or dot
    """, re.VERBOSE)

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(self, expr: str) -> float:
        """Evaluate *expr* and return the result as a float.

        Parameters
        ----------
        expr: str
            Arithmetic expression containing numbers, + - * /, parentheses
            and optional whitespace.

        Returns
        -------
        float
            Result of the expression.

        Raises
        ------
        ValueError
            If the expression is empty, contains invalid tokens,
            has mismatched parentheses, or attempts division by zero.
        """
        if not expr or expr.strip() == "":
            raise ValueError("empty expression")

        self._tokenize(expr)
        self._pos = 0
        result = self._parse_expression()
        if self._pos != len(self._tokens):
            # leftover tokens -> syntax error
            raise ValueError("invalid token or mismatched parentheses")
        return result

    # ------------------------------------------------------------------ #
    # Tokeniser
    # ------------------------------------------------------------------ #
    def _tokenize(self, expr: str) -> None:
        """Convert *expr* into a list of tokens (numbers, operators, parentheses)."""
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch in "+-*/()":
                self._tokens.append(ch)
                i += 1
                continue
            # Number token – consume the longest prefix that matches the number regex
            match = self._NUMBER_RE.match(expr, i)
            if not match:
                raise ValueError(f"invalid token '{ch}' at position {i}")
            self._tokens.append(match.group(0))
            i = match.end()
        # No further validation here; the parser will detect syntax errors.

    # ------------------------------------------------------------------ #
    # Recursive‑descent parsing
    # ------------------------------------------------------------------ #
    def _parse_expression(self) -> float:
        """expr := term ((+|-) term)*"""
        value = self._parse_term()
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ("+", "-"):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_term()
            if op == "+":
                value += right
            else:  # op == "-"
                value -= right
        return value

    def _parse_term(self) -> float:
        """term := factor ((*|/) factor)*"""
        value = self._parse_factor()
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ("*", "/"):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_factor()
            if op == "*":
                value *= right
            else:  # op == "/"
                if right == 0.0:
                    raise ValueError("division by zero")
                value /= right
        return value

    def _parse_factor(self) -> float:
        """factor := (+|-) factor | number | '(' expr ')'"""
        if self._pos >= len(self._tokens):
            raise ValueError("unexpected end of expression")

        token = self._tokens[self._pos]

        # Unary plus/minus
        if token in ("+", "-"):
            self._pos += 1
            operand = self._parse_factor()
            return operand if token == "+" else -operand

        # Parenthesised sub‑expression
        if token == "(":
            self._pos += 1
            value = self._parse_expression()
            if self._pos >= len(self._tokens) or self._tokens[self._pos] != ")":
                raise ValueError("mismatched parentheses")
            self._pos += 1  # consume ')'
            return value

        # Number
        if self._NUMBER_RE.fullmatch(token):
            self._pos += 1
            return float(token)

        # Anything else is invalid
        raise ValueError(f"invalid token '{token}' at position {self._pos}")


# ---------------------------------------------------------------------- #
# Pytest test suite (save as test_expression_evaluator.py)
# ---------------------------------------------------------------------- #

import pytest


@pytest.fixture
def evaluator():
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / 3") == pytest.approx(10 / 3)


def test_unary_minus(evaluator):
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-(2+3)") == -5.0
    assert evaluator.evaluate("+7 * -2") == -14.0
    assert evaluator.evaluate("-(-3)") == 3.0


def test_floats_and_whitespace(evaluator):
    assert evaluator.evaluate("  3.14 + 2.86 ") == 6.0
    assert evaluator.evaluate(".5 * .5") == 0.25
    assert evaluator.evaluate("1.0 / .2") == 5.0


def test_division_by_zero(evaluator):
    with pytest.raises(ValueError, match="division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="division by zero"):
        evaluator.evaluate("(2+3) / (5-5)")


def test_invalid_syntax(evaluator):
    # mismatched parentheses
    with pytest.raises(ValueError, match="mismatched parentheses"):
        evaluator.evaluate("(1+2")
    with pytest.raises(ValueError, match="mismatched parentheses"):
        evaluator.evaluate("1+2)")
    # empty expression
    with pytest.raises(ValueError, match="empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="empty expression"):
        evaluator.evaluate("   ")
    # invalid token
    with pytest.raises(ValueError, match="invalid token"):
        evaluator.evaluate("2 & 3")
    with pytest.raises(ValueError, match="invalid token"):
        evaluator.evaluate("2++3")