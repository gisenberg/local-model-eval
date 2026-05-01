"""Expression evaluator implementing a recursive‑descent parser.

Supported features
------------------
* Binary operators: ``+``, ``-``, ``*``, ``/`` (correct precedence)
* Parentheses for grouping
* Unary minus (and unary plus) – e.g. ``-3`` or ``-(2+1)``
* Floating‑point numbers
* Explicit error handling:
    - mismatched parentheses
    - division by zero
    - invalid tokens
    - empty expression
"""

from __future__ import annotations

import re
from typing import List


class ExpressionEvaluator:
    """
    Evaluate a mathematical expression without using ``eval`` or ``ast``.

    Parameters
    ----------
    expr: str
        The expression to evaluate. May contain whitespace.

    Returns
    -------
    float
        The numeric result of the expression.

    Raises
    ------
    ValueError
        * Empty expression
        * Mismatched parentheses
        * Division by zero
        * Invalid token
    ZeroDivisionError
        *Never* – division by zero is wrapped in a ``ValueError`` as required.
    """

    # --------------------------------------------------------------------- #
    #  Tokenisation
    # --------------------------------------------------------------------- #
    _TOKEN_RE = re.compile(r"""
        \s*                           # optional leading whitespace
        (                               # start group for a token
            (?P<NUM>[0-9]*\.?[0-9]+)    # integer or decimal number
            |                           # or
            (?P<OP>[\+\-\*\/\(\)])      # any operator or parenthesis
        )
        \s*                           # optional trailing whitespace
    """, re.VERBOSE)

    def _tokenize(self, expr: str) -> List[str]:
        """Convert *expr* into a list of token strings."""
        tokens = self._TOKEN_RE.findall(expr)
        # Verify that the whole string was consumed (no illegal characters)
        consumed = ''.join(tokens)
        if len(consumed.replace(' ', '')) != len(expr.replace(' ', '')):
            raise ValueError(f"Invalid token(s) in expression: {expr!r}")
        if not tokens:
            raise ValueError("Empty expression")
        return tokens

    # --------------------------------------------------------------------- #
    #  Recursive‑descent parser
    # --------------------------------------------------------------------- #
    class _Parser:
        """Inner parser that works on a list of tokens."""

        def __init__(self, tokens: List[str]) -> None:
            self._tokens = tokens
            self._pos = 0

        # ----- utility ---------------------------------------------------- #
        def _peek(self) -> str | None:
            """Return current token without consuming it, or ``None`` if at end."""
            return self._tokens[self._pos] if self._pos < len(self._tokens) else None

        def _next(self) -> str:
            """Consume and return the current token."""
            token = self._peek()
            if token is None:
                raise ValueError("Unexpected end of tokens")
            self._pos += 1
            return token

        def has_more(self) -> bool:
            """True if there are remaining tokens."""
            return self._pos < len(self._tokens)

        # ----- grammar --------------------------------------------------- #
        def parse_expr(self) -> float:
            """Parse an expression: term (('+' | '-') term)*"""
            value = self.parse_term()
            while self._peek() in ('+', '-'):
                op = self._next()
                right = self.parse_term()
                if op == '+':
                    value += right
                else:
                    value -= right
            return value

        def parse_term(self) -> float:
            """Parse a term: factor (('*' | '/') factor)*"""
            value = self.parse_factor()
            while self._peek() in ('*', '/'):
                op = self._next()
                right = self.parse_factor()
                if op == '*':
                    value *= right
                else:
                    if right == 0:
                        raise ValueError("division by zero")
                    value /= right
            return value

        def parse_factor(self) -> float:
            """
            Parse a factor: (('+' | '-') factor)? primary

            Handles unary plus/minus.
            """
            if self._peek() in ('+', '-'):
                op = self._next()
                if op == '-':
                    # Unary minus applies to the entire factor
                    value = self.parse_factor()
                    return -value
                # Unary plus – just skip it
                value = self.parse_factor()
                return value
            return self.parse_primary()

        def parse_primary(self) -> float:
            """
            Parse a primary: number | '(' expr ')'
            """
            token = self._peek()
            if token == '(':
                self._next()                # consume '('
                value = self.parse_expr()
                if self._peek() != ')':
                    raise ValueError("Mismatched parentheses")
                self._next()                # consume ')'
                return value
            # number (integer or decimal)
            if re.fullmatch(r'\d+\.\d+|\d+\.?\d*', token):
                self._next()
                return float(token)
            raise ValueError(f"Invalid token: {token!r}")

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def evaluate(self, expr: str) -> float:
        """
        Evaluate *expr* and return the result as ``float``.

        The method tokenises the expression, parses it with a recursive‑descent
        algorithm, and returns the computed value.

        Parameters
        ----------
        expr: str
            The arithmetic expression to evaluate.

        Returns
        -------
        float
            Result of the evaluation.

        Raises
        ------
        ValueError
            If the expression is empty, contains mismatched parentheses,
            division by zero, or any invalid token.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        parser = self._Parser(tokens)
        result = parser.parse_expr()

        if parser.has_more():
            raise ValueError(f"Invalid token(s) at position {parser._pos}")

        return result

import pytest


def test_basic_addition():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0


def test_precedence_and_parentheses():
    evaluator = ExpressionEvaluator()
    # multiplication has higher precedence than addition
    assert evaluator.evaluate("2 * 3 + 4") == 10.0
    # parentheses override precedence
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0


def test_unary_minus_and_complex():
    evaluator = ExpressionEvaluator()
    # unary minus, parentheses, and mixed operators
    expr = "-3 + -(2 * (5 - 3))"
    assert evaluator.evaluate(expr) == -7.0


def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="division by zero"):
        evaluator.evaluate("5 / 0")


def test_mismatched_parentheses():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 * (3 + 4")   # missing closing ')'

    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1+2")          # missing closing ')'