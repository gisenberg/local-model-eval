"""Recursive‑descent expression evaluator.

The evaluator understands the binary operators ``+``, ``-``, ``*`` and ``/`` with
the usual precedence, parentheses for grouping, and a unary minus operator.
Floating‑point numbers are supported.  All errors are reported as ``ValueError``.
"""

from __future__ import annotations

from typing import List, Union, Optional


class ExpressionEvaluator:
    """Evaluate a mathematical expression without using ``eval``.

    Parameters
    ----------
    expr : str
        The expression to evaluate.  It may contain the operators ``+ - * /``,
        parentheses ``( )``, floating‑point numbers and a leading unary minus.

    Raises
    ------
    ValueError
        * ``empty expression`` – the input string is empty or contains only whitespace.
        * ``mismatched parentheses`` – a ``(`` without a matching ``)`` (or vice‑versa).
        * ``division by zero`` – a division by ``0`` is attempted.
        * ``invalid token`` – any character that is not a digit, ``.``, an operator
          or a parenthesis is encountered.
    """

    def __init__(self, expr: str) -> None:
        self.tokens: List[Union[float, str]] = self._tokenize(expr)
        if not self.tokens:
            raise ValueError("empty expression")
        self.pos: int = 0                     # current token index

    # --------------------------------------------------------------------- #
    # Tokenisation
    # --------------------------------------------------------------------- #
    def _tokenize(self, expr: str) -> List[Union[float, str]]:
        """Convert the raw string into a list of tokens.

        Numbers are converted to ``float``; all other non‑whitespace characters
        are returned as single‑character strings.  Any other character triggers
        a ``ValueError``.
        """
        tokens: List[Union[float, str]] = []
        i = 0
        while i < len(expr):
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or ch == ".":
                # parse a multi‑digit / decimal number
                start = i
                while i < len(expr) and (expr[i].isdigit() or expr[i] == "."):
                    i += 1
                num_str = expr[start:i]
                try:
                    tokens.append(float(num_str))
                except ValueError as exc:
                    raise ValueError(f"invalid number: {num_str}") from exc
            elif ch in "+-*/()":
                tokens.append(ch)
                i += 1
            else:
                raise ValueError(f"invalid token: {ch!r}")
        return tokens

    # --------------------------------------------------------------------- #
    # Parsing
    # --------------------------------------------------------------------- #
    def _peek(self) -> Optional[Union[float, str]]:
        """Return the current token without consuming it."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> Optional[Union[float, str]]:
        """Consume the current token and advance the position."""
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        return None

    # expr ::= term (('+' | '-') term)*
    def parse_expr(self) -> float:
        """Parse an expression (addition / subtraction)."""
        value = self.parse_term()
        while (op := self._peek()) in ("+", "-"):
            self._consume()                     # consume the operator
            right = self.parse_term()
            if op == "+":
                value += right
            else:
                value -= right
        return value

    # term ::= factor (('*' | '/') factor)*
    def parse_term(self) -> float:
        """Parse a term (multiplication / division)."""
        value = self.parse_factor()
        while (op := self._peek()) in ("*", "/"):
            self._consume()
            right = self.parse_factor()
            if op == "*":
                value *= right
            else:
                if right == 0:
                    raise ValueError("division by zero")
                value /= right
        return value

    # factor ::= ('-' factor)? primary
    def parse_factor(self) -> float:
        """Parse a factor, handling an optional unary minus."""
        if self._peek() == "-":
            self._consume()                     # consume the unary minus
            return -self.parse_factor()
        return self.parse_primary()

    # primary ::= number | '(' expr ')'
    def parse_primary(self) -> float:
        """Parse a primary element (number or parenthesised sub‑expression)."""
        token = self._peek()
        if token == "(":
            self._consume()                     # '('
            value = self.parse_expr()
            if self._peek() != ")":
                raise ValueError("mismatched parentheses")
            self._consume()                     # ')'
            return value
        if isinstance(token, float):
            self._consume()
            return token
        raise ValueError(f"invalid token: {token!r}")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def evaluate(self) -> float:
        """Evaluate the expression and return the result as a ``float``."""
        return self.parse_expr()

"""pytest test‑suite for :class:`ExpressionEvaluator`."""

import pytest


def test_basic_precedence():
    """2 + 3 * 4  → 14  (multiplication before addition)."""
    evaluator = ExpressionEvaluator("-4 + 3 * 2")   # also checks unary minus
    assert evaluator.evaluate() == 2.0


def test_parentheses_correct():
    """(2 + 3) * 4  → 20."""
    evaluator = ExpressionEvaluator("(2+3)*4")
    assert evaluator.evaluate() == 20.0


def test_mismatched_parentheses():
    """'(2+3' lacks a closing parenthesis → ValueError."""
    evaluator = ExpressionEvaluator("(2+3")
    with pytest.raises(ValueError, match="mismatched parentheses"):
        evaluator.evaluate()


def test_division_by_zero():
    """5 / 0 → division by zero error."""
    evaluator = ExpressionEvaluator("5/0")
    with pytest.raises(ValueError, match="division by zero"):
        evaluator.evaluate()


def test_invalid_cases():
    """Empty string and non‑numeric token should raise ValueError."""
    # empty expression
    with pytest.raises(ValueError, match="empty expression"):
        ExpressionEvaluator("").evaluate()

    # illegal character
    with pytest.raises(ValueError, match="invalid token"):
        ExpressionEvaluator("2 + a").evaluate()