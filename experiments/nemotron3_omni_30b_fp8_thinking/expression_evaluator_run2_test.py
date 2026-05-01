"""Expression evaluator implementing a recursive‑descent parser.

The evaluator supports the binary operators +, -, *, / with the correct
precedence, parentheses for grouping, and unary minus.  It works with
floating‑point numbers and raises ``ValueError`` for empty expressions,
mismatched parentheses, division by zero, or any invalid token.
"""

from __future__ import annotations

import re
from typing import List


class ExpressionEvaluator:
    """Evaluate a mathematical expression provided as a string.

    Parameters
    ----------
    expr : str
        The arithmetic expression to evaluate.  It may contain
        integers, floats, the operators ``+ - * /``, parentheses and
        unary minus.

    Returns
    -------
    float
        The result of the evaluated expression.

    Raises
    ------
    ValueError
        If the expression is empty, contains mismatched parentheses,
        division by zero, or any invalid token.
    """

    def evaluate(self, expr: str) -> float:
        """Parse and evaluate *expr*.

        Parameters
        ----------
        expr : str
            The arithmetic expression.

        Returns
        -------
        float
            The computed value.

        Raises
        ------
        ValueError
            For empty input, syntax errors, or runtime errors such as
            division by zero.
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        parser = self._Parser(tokens)
        return parser.parse()

    # -----------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------
    def _tokenize(self, expr: str) -> List[str]:
        """Convert *expr* into a list of tokens.

        Tokens are numbers (as strings) and operators/parentheses.
        Whitespace is ignored.

        Raises
        ------
        ValueError
            If an illegal character is encountered.
        """
        tokens: List[str] = []
        i = 0
        while i < len(expr):
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue

            if ch.isdigit() or ch == ".":
                start = i
                while i < len(expr) and (expr[i].isdigit() or expr[i] == "."):
                    i += 1
                num_str = expr[start:i]
                # Validate numeric literal
                try:
                    float(num_str)  # noqa: F821
                except ValueError as exc:
                    raise ValueError(f"Invalid number token: {num_str}") from exc
                tokens.append(num_str)
                continue

            if ch in "+-*/()":
                tokens.append(ch)
                i += 1
                continue

            raise ValueError(f"Invalid token: {ch}")

        return tokens

    # -----------------------------------------------------------------
    # Recursive‑descent parser
    # -----------------------------------------------------------------
    class _Parser:
        """Recursive‑descent parser for the token list."""

        def __init__(self, tokens: List[str]) -> None:
            self.tokens = tokens
            self.pos = 0
            self.has_error = False

        # -----------------------------------------------------------------
        # Helper methods
        # -----------------------------------------------------------------
        def peek(self) -> str | None:
            """Return the current token without consuming it."""
            return self.tokens[self.pos] if self.pos < len(self.tokens) else None

        def consume(self) -> str:
            """Consume and return the current token, advancing the pointer."""
            if self.pos >= len(self.tokens):
                self.has_error = True
                return ""
            token = self.tokens[self.pos]
            self.pos += 1
            return token

        def has_error(self) -> bool:
            return self.has_error

        # -----------------------------------------------------------------
        # Grammar entry point
        # -----------------------------------------------------------------
        def parse(self) -> float:
            """Parse the entire token stream and return the result."""
            result = self.parse_expression()
            if self.pos < len(self.tokens):
                self.has_error = True
            return result

        # -----------------------------------------------------------------
        # Grammar rules
        # -----------------------------------------------------------------
        def parse_expression(self) -> float:
            """Parse an expression consisting of terms separated by '+' or '-'."""
            value = self.parse_term()
            while (tok := self.peek()) in ("+", "-"):
                op = self.consume()
                rhs = self.parse_term()
                if op == "+":
                    value += rhs
                else:
                    value -= rhs
            return value

        def parse_term(self) -> float:
            """Parse a term consisting of factors separated by '*' or '/'."""
            value = self.parse_factor()
            while (tok := self.peek()) in ("*", "/"):
                op = self.consume()
                rhs = self.parse_factor()
                if op == "*":
                    value *= rhs
                else:
                    if rhs == 0:
                        raise ValueError("Division by zero")
                    value /= rhs
            return value

        def parse_factor(self) -> float:
            """Parse a factor, handling parentheses and unary minus."""
            token = self.peek()
            if token == "(":
                self.consume()  # '('
                value = self.parse_expression()
                if self.peek() != ")":
                    raise ValueError("Mismatched parentheses")
                self.consume()  # ')'
                return value

            # Unary minus: present when '-' appears at start or after '('
            if token == "-" and (
                self.pos == 0
                or self.tokens[self.pos - 1] in ("+", "-", "*", "/", "(")
            ):
                self.consume()  # consume '-'
                return -self.parse_factor()

            if token is None:
                raise ValueError("Unexpected end of input")

            # Must be a number
            if token not in ("+", "-", "*", "/", "(", ")"):
                try:
                    value = float(token)
                except ValueError as exc:
                    raise ValueError(f"Invalid token: {token}") from exc
                self.consume()
                return value

            # Any other operator encountered here is a syntax error
            raise ValueError(f"Unexpected token: {token}")

    # -----------------------------------------------------------------
    # End of class
    # -----------------------------------------------------------------

# test_expression_evaluator.py
import pytest



def test_addition():
    evalutor = ExpressionEvaluator()
    assert evalutor.evaluate("2+3") == 5.0


def test_precedence_mul_over_add():
    evalutor = ExpressionEvaluator()
    assert evalutor.evaluate("2+3*4") == 14.0  # 3*4 first


def test_parentheses():
    evalutor = ExpressionEvaluator()
    assert evalutor.evaluate("(2+3)*4") == 20.0


def test_unary_minus():
    evalutor = ExpressionEvaluator()
    assert evalutor.evaluate("-3+4") == 1.0
    assert evalutor.evaluate("-(2+1)") == -3.0


def test_division_by_zero():
    evalutor = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evalutor.evaluate("5/0")