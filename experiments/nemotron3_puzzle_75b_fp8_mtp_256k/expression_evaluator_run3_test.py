"""ExpressionEvaluator – a tiny recursive‑descent math expression evaluator.

Supports +, -, *, /, parentheses, unary minus and floating point numbers.
Raises ValueError for mismatched parentheses, division by zero,
invalid tokens or empty input.
"""

from __future__ import annotations

from typing import List, Union


class ExpressionEvaluator:
    """Evaluates arithmetic expressions using a recursive‑descent parser."""

    def __init__(self) -> None:
        self._tokens: List[Union[float, str]] = []
        self._pos: int = 0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def evaluate(self, expr: str) -> float:
        """Parse and evaluate *expr*.

        Parameters
        ----------
        expr: str
            The arithmetic expression to evaluate.

        Returns
        -------
        float
            The result of the expression.

        Raises
        ------
        ValueError
            If the expression is empty, contains invalid tokens,
            has mismatched parentheses or attempts division by zero.
        """
        if not expr or expr.strip() == "":
            raise ValueError("Empty expression")

        self._tokenize(expr)
        self._pos = 0
        if not self._tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()
        if self._pos != len(self._tokens):
            # Leftover tokens -> mismatched parentheses or stray symbols
            raise ValueError("Mismatched parentheses or extra tokens")
        return result

    # --------------------------------------------------------------------- #
    # Tokeniser
    # --------------------------------------------------------------------- #
    def _tokenize(self, s: str) -> None:
        """Convert *s* into a list of tokens (numbers, operators, parentheses)."""
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch.isspace():
                i += 1
                continue
            if ch in "+-*/()":
                self._tokens.append(ch)
                i += 1
                continue
            if ch.isdigit() or ch == ".":
                start = i
                dot_seen = False
                while i < n and (s[i].isdigit() or s[i] == "."):
                    if s[i] == ".":
                        if dot_seen:
                            raise ValueError("Invalid number format")
                        dot_seen = True
                    i += 1
                num_str = s[start:i]
                # Reject "." or multiple dots
                if num_str == "." or num_str.count(".") > 1:
                    raise ValueError("Invalid number format")
                try:
                    value = float(num_str)
                except ValueError as exc:
                    raise ValueError(f"Invalid number: {num_str}") from exc
                self._tokens.append(value)
                continue
            raise ValueError(f"Invalid character: {ch!r}")

    # --------------------------------------------------------------------- #
    # Recursive‑descent parsing
    # --------------------------------------------------------------------- #
    def _parse_expression(self) -> float:
        """expression ::= term (('+' | '-') term)*"""
        left = self._parse_term()
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ("+", "-"):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_term()
            left = left + right if op == "+" else left - right
        return left

    def _parse_term(self) -> float:
        """term ::= factor (('*' | '/') factor)*"""
        left = self._parse_factor()
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ("*", "/"):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_factor()
            if op == "*":
                left = left * right
            else:  # division
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
        return left

    def _parse_factor(self) -> float:
        """factor ::= ('+' | '-') factor | NUMBER | '(' expression ')'"""
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression")

        token = self._tokens[self._pos]

        # Unary plus
        if token == "+":
            self._pos += 1
            return self._parse_factor()

        # Unary minus
        if token == "-":
            self._pos += 1
            return -self._parse_factor()

        # Parenthesised sub‑expression
        if token == "(":
            self._pos += 1
            val = self._parse_expression()
            if self._pos >= len(self._tokens) or self._tokens[self._pos] != ")":
                raise ValueError("Mismatched parentheses")
            self._pos += 1  # consume ')'
            return val

        # Number
        if isinstance(token, (int, float)):
            self._pos += 1
            return float(token)

        raise ValueError(f"Invalid token: {token!r}")

import pytest


@pytest.fixture
def evaluator():
    return ExpressionEvaluator()


def test_basic_precedence(evaluator):
    """2 + 3 * 4 = 14 (multiplication before addition)."""
    assert evaluator.evaluate("2 + 3 * 4") == pytest.approx(14.0)


def test_parentheses(evaluator):
    """(2 + 3) * 4 = 20."""
    assert evaluator.evaluate("(2 + 3) * 4") == pytest.approx(20.0)


def test_unary_minus(evaluator):
    """-3 + 2 * (-1) = -5."""
    assert evaluator.evaluate("-3 + 2 * (-1)") == pytest.approx(-5.0)


def test_floating_point(evaluator):
    """3.5 + 2.1 * 2 = 7.7."""
    assert evaluator.evaluate("3.5 + 2.1 * 2") == pytest.approx(7.7)


def test_division_by_zero(evaluator):
    """Division by zero must raise ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")


def test_mismatched_parentheses(evaluator):
    """Missing closing parenthesis -> ValueError."""
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")


def test_invalid_token(evaluator):
    """Any unsupported character triggers ValueError."""
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 @ 3")


def test_empty_expression(evaluator):
    """Empty or whitespace‑only strings raise ValueError."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")