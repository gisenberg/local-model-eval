"""Expression evaluator implementing a recursive‑descent parser.

The evaluator understands the binary operators ``+``, ``-``, ``*``, ``/``,
parentheses for grouping and the unary minus operator.  It works with
floating‑point numbers only.

Typical usage
-------------
>>> >>> evaluator = ExpressionEvaluator()
>>> evaluator.evaluate("3 + 4 * (2 - 1)")
11.0
"""

from __future__ import annotations

from typing import List, Union


class Parser:
    """Recursive‑descent parser for the token list produced by :meth:`tokenize`.

    The grammar recognised is::

        expr   ::= term (( '+' | '-' ) term)*
        term   ::= factor (( '*' | '/' ) factor)*
        factor ::= ( '-' )? primary
        primary::= number | '(' expr ')'

    All operations are performed with floating‑point precision.
    """

    def __init__(self, tokens: List[Union[float, str]]) -> None:
        self.tokens: List[Union[float, str]] = tokens
        self.pos: int = 0

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def peek(self) -> Union[float, str, None]:
        """Return the current token without consuming it, or ``None`` if at EOF."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: Union[float, str] | None = None) -> Union[float, str]:
        """Consume the current token and advance the cursor.

        If *expected* is supplied the token must equal it, otherwise a
        ``ValueError`` is raised.
        """
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of input")
        if expected is not None and token != expected:
            raise ValueError(f"Expected {expected!r}, got {token!r}")
        self.pos += 1
        return token

    # --------------------------------------------------------------------- #
    # Grammar implementation
    # --------------------------------------------------------------------- #
    def parse(self) -> float:
        """Parse the whole token list and return the resulting float."""
        result = self.parse_expr()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token {self.peek()!r}")
        return result

    def parse_expr(self) -> float:
        """Parse an expression: ``term`` possibly followed by ``+`` or ``-`` terms."""
        left = self.parse_term()
        while self.peek() in ("+", "-"):
            op = self.consume()
            right = self.parse_term()
            if op == "+":
                left = left + right
            else:
                left = left - right
        return left

    def parse_term(self) -> float:
        """Parse a term: ``factor`` possibly followed by ``*`` or ``/`` factors."""
        left = self.parse_factor()
        while self.peek() in ("*", "/"):
            op = self.consume()
            right = self.parse_factor()
            if op == "*":
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
        return left

    def parse_factor(self) -> float:
        """Parse a factor, which may be a unary minus followed by a primary."""
        if self.peek() == "-":
            self.consume("-")
            return -self.parse_primary()
        return self.parse_primary()

    def parse_primary(self) -> float:
        """Parse a primary expression: a number or a parenthesised sub‑expression."""
        token = self.peek()
        if token == "(":
            self.consume("(")
            val = self.parse_expr()
            self.consume(")")          # will raise if ')' is missing
            return val
        elif isinstance(token, float):
            self.consume()
            return token
        else:
            raise ValueError(f"Unexpected token {token!r}")


class ExpressionEvaluator:
    """Public façade that tokenises a string and evaluates it.

    The class raises :class:`ValueError` for any of the following conditions:

    * empty or whitespace‑only input,
    * mismatched parentheses,
    * division by zero,
    * illegal characters (invalid tokens).
    """

    def evaluate(self, expr: str) -> float:
        """Evaluate *expr* and return the result as a ``float``.

        Parameters
        ----------
        expr: str
            The arithmetic expression to evaluate.

        Returns
        -------
        float
            The numeric result.

        Raises
        ------
        ValueError
            If the expression is empty, contains illegal characters,
            has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokens = self.tokenize(expr)
        parser = Parser(tokens)
        return parser.parse()

    # --------------------------------------------------------------------- #
    # Tokeniser
    # --------------------------------------------------------------------- #
    @staticmethod
    def tokenize(expr: str) -> List[Union[float, str]]:
        """Convert *expr* into a list of numbers and operator/parenthesis tokens.

        The tokenizer recognises:
            - floating‑point numbers (e.g. ``3.14``, ``-0.5``),
            - the operators ``+``, ``-``, ``*``, ``/``,
            - parentheses ``(`` and ``)``,
            - whitespace is ignored.

        Any other character causes a ``ValueError`` with the message
        ``Invalid token: <char>``.
        """
        tokens: List[Union[float, str]] = []
        i = 0
        n = len(expr)

        while i < n:
            ch = expr[i]

            # Skip whitespace
            if ch.isspace():
                i += 1
                continue

            # Number (integer or floating point)
            if ch.isdigit() or ch == ".":
                start = i
                dot_seen = ch == "."
                i += 1
                while i < n and (expr[i].isdigit() or expr[i] == "."):
                    if expr[i] == "." and dot_seen:
                        raise ValueError(f"Invalid number: {expr[start:i]}")
                    i += 1
                num_str = expr[start:i]
                try:
                    tokens.append(float(num_str))
                except ValueError as exc:
                    raise ValueError(f"Invalid number: {num_str}") from exc
                continue

            # Operators and parentheses
            if ch in "+-*/()":
                tokens.append(ch)
                i += 1
                continue

            # Anything else is illegal
            raise ValueError(f"Invalid token: {ch!r}")

        return tokens

import pytest



def test_basic_addition():
    """Simple addition."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1+2") == 3.0


def test_precedence():
    """Operator precedence: multiplication before addition."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2+3*4") == 14.0   # 3*4 = 12, 2+12 = 14


def test_unary_minus():
    """Unary minus and parentheses grouping."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5+3") == -2.0
    assert evaluator.evaluate("-(2+1)") == -3.0


def test_parentheses():
    """Parentheses change evaluation order."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(1+2)*(3-4/2)") == 3.0   # (3)*(1) = 3


def test_errors():
    """Various error conditions must raise ValueError."""
    evaluator = ExpressionEvaluator()

    # empty (or whitespace‑only) expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

    # division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1/0")

    # mismatched parentheses
    with pytest.raises(ValueError, match="mismatched parentheses"):
        evaluator.evaluate("(1+2")

    # invalid token
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1+2a")