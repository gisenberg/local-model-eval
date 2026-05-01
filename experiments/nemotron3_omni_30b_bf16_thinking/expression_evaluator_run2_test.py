from __future__ import annotations

from typing import List, Tuple


class ExpressionEvaluator:
    """
    A tiny recursive‑descent evaluator for arithmetic expressions.

    Supported features
    ------------------
    * Binary operators: ``+``, ``-``, ``*``, ``/`` (with the usual precedence).
    * Parentheses for grouping.
    * Unary minus (e.g. ``-3`` or ``-(2+1)``).  Unary plus is also accepted.
    * Floating‑point numbers (e.g. ``3.14``).

    Errors
    ------
    * ``ValueError`` is raised for:
        - mismatched parentheses,
        - division by zero,
        - any token that cannot be parsed,
        - an empty expression.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate an arithmetic expression.

        Parameters
        ----------
        expr: str
            The expression to evaluate.  Whitespace is ignored.

        Returns
        -------
        float
            The numeric result.

        Raises
        ------
        ValueError
            If the expression is empty, contains an invalid token,
            has mismatched parentheses, or attempts division by zero.
        """
        if not expr or expr.strip() == "":
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Invalid expression")

        parser = _Parser(tokens)
        result = parser.parse_expr()

        # Ensure the whole input was consumed (e.g. “2+” is illegal)
        if parser.pos != len(parser.tokens):
            raise ValueError("Invalid expression")

        return result

    # --------------------------------------------------------------------- #
    # Lexer
    # --------------------------------------------------------------------- #
    def _tokenize(self, s: str) -> List[Tuple[str, str]]:
        """
        Convert the input string into a list of tokens.

        Each token is a tuple ``(type, value)`` where *type* is one of:
            - ``'num'``   : a numeric literal (integer or float)
            - ``'plus'``  : ``+``
            - ``'minus'`` : ``-``
            - ``'mul'``   : ``*``
            - ``'div'``   : ``/``
            - ``'lparen'``: ``(``
            - ``'rparen'``: ``)``

        Whitespace is ignored.  Any character that is not one of the above
        triggers a ``ValueError``.
        """
        tokens: List[Tuple[str, str]] = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch.isspace():
                i += 1
                continue

            if ch.isdigit() or ch == '.':
                # Parse a number (integer or decimal)
                start = i
                dot_seen = ch == '.'
                i += 1
                while i < len(s) and (s[i].isdigit() or (s[i] == '.' and not dot_seen)):
                    if s[i] == '.':
                        dot_seen = True
                    i += 1
                num_str = s[start:i]
                tokens.append(('num', num_str))
                continue

            if ch in '+-*/':
                tokens.append((ch, ch))
                i += 1
                continue

            if ch == '(':
                tokens.append(('lparen', '('))
                i += 1
                continue

            if ch == ')':
                tokens.append(('rparen', ')'))
                i += 1
                continue

            raise ValueError(f"Invalid token: {ch}")

        return tokens


# -------------------------------------------------------------------------
# Recursive‑descent parser (inner to keep the public API clean)
# -------------------------------------------------------------------------
class _Parser:
    """
    Recursive‑descent parser that works on the token list produced by
    :meth:`ExpressionEvaluator._tokenize`.
    """

    def __init__(self, tokens: List[Tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0

    # -----------------------------------------------------------------
    # Helper utilities
    # -----------------------------------------------------------------
    def _peek(self) -> str | None:
        """Return the type of the current token without consuming it."""
        return self.tokens[self.pos][0] if self.pos < len(self.tokens) else None

    def _consume(self) -> Tuple[str, str]:
        """Consume the current token and return it.  Raises if out of range."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of input")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    # -----------------------------------------------------------------
    # Grammar
    # -----------------------------------------------------------------
    def parse_expr(self) -> float:
        """Parse addition and subtraction."""
        value = self.parse_term()
        while (tok := self._peek()) == 'plus':
            self._consume()
            value += self.parse_term()
        while (tok := self._peek()) == 'minus':
            self._consume()
            value -= self.parse_term()
        return value

    def parse_term(self) -> float:
        """Parse multiplication and division."""
        value = self.parse_factor()
        while (tok := self._peek()) == 'mul':
            self._consume()
            value *= self.parse_factor()
        while (tok := self._peek()) == 'div':
            self._consume()
            divisor = self.parse_factor()
            if divisor == 0.0:
                raise ValueError("division by zero")
            value /= divisor
        return value

    def parse_factor(self) -> float:
        """
        Parse an optional unary ``+``/``-`` followed by a primary expression.
        This allows ``-3`` and ``-(2+1)`` while also accepting ``+5``.
        """
        if (tok := self._peek()) == 'minus':
            self._consume()
            return -self.parse_factor()
        if (tok := self._peek()) == 'plus':
            self._consume()
            return self.parse_factor()

        return self.parse_primary()

    def parse_primary(self) -> float:
        """
        Parse a number or a parenthesised sub‑expression.
        """
        if (tok := self._peek()) == 'num':
            value_str = self._consume()[1]
            return float(value_str)

        if (tok := self._peek()) == 'lparen':
            self._consume()  # '('
            value = self.parse_expr()
            if self._peek() != 'rparen':
                raise ValueError("Mismatched parentheses")
            self._consume()  # ')'
            return value

        raise ValueError(f"Unexpected token: {self._peek()}")

import pytest


def test_basic_precedence():
    """2 + 3 * 4 should respect operator precedence (2 + 12 = 14)."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2+3*4") == 14


def test_parentheses():
    """Parentheses must force grouping before applying surrounding operators."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2+3)*4") == 20
    assert evaluator.evaluate("2*(3+4)") == 14


def test_unary_minus():
    """Unary minus should apply to the following number or parenthesised expression."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5+3") == -2          # -5 + 3 = -2
    assert evaluator.evaluate("-(2+1)") == -3       # -(2+1) = -3
    assert evaluator.evaluate("+7") == 7            # unary plus is a no‑op


def test_division_by_zero():
    """Division by zero must raise a ValueError."""
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="division by zero"):
        evaluator.evaluate("5/0")


def test_invalid_expression():
    """Malformed input (mismatched parentheses, illegal tokens, empty string) raises ValueError."""
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate("(1+2")          # missing closing ')'
    with pytest.raises(ValueError):
        evaluator.evaluate("2+*3")          # '*' cannot follow another operator
    with pytest.raises(ValueError):
        evaluator.evaluate("")              # empty expression