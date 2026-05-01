"""
ExpressionEvaluator – a tiny recursive‑descent calculator.

Supported features
------------------
* Binary operators: ``+``, ``-``, ``*``, ``/`` (correct precedence)
* Parentheses for grouping
* Unary minus, e.g. ``-3`` or ``-(2+1)``
* Floating‑point numbers (e.g. ``3.14``)

The public API is the ``ExpressionEvaluator.evaluate`` method which
receives a string and returns the computed ``float`` value.

The implementation deliberately avoids ``eval`` / ``ast.literal_eval`` and
uses a classic recursive‑descent parser.
"""

from __future__ import annotations

from typing import List, Optional


class ExpressionEvaluator:
    """
    Evaluate a mathematical expression.

    Parameters
    ----------
    expr : str
        The expression to evaluate.  It is parsed each time ``evaluate`` is
        called; the class does not store the expression between calls.

    Returns
    -------
    float
        The numeric result of the expression.

    Raises
    ------
    ValueError
        * ``empty expression`` – the input string is empty or contains only
          whitespace.
        * ``mismatched parentheses`` – a ``(`` without a matching ``)`` (or
          vice‑versa).
        * ``division by zero`` – a division operation where the right‑hand
          operand is zero.
        * ``invalid token`` – any character that is not a digit, a decimal
          point, or one of the allowed operators / parentheses.
    """

    def __init__(self, expr: str) -> None:
        """
        Initialise the parser with the expression to be evaluated.

        The tokenizer runs once; the parser works on the resulting token list.
        """
        self.tokens: List[str] = self._tokenize(expr)
        self.pos: int = 0  # current token index

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def evaluate(self, expr: str) -> float:
        """
        Parse and evaluate *expr*.

        The method tokenises the string, then runs the recursive‑descent
        parser.  Any syntax problem raises ``ValueError`` with an explanatory
        message.

        Parameters
        ----------
        expr : str
            The expression to evaluate.

        Returns
        -------
        float
            The computed value.
        """
        self.tokens = self._tokenize(expr)
        self.pos = 0
        if not self.tokens:
            raise ValueError("empty expression")
        result = self._parse_expression()
        if self.pos < len(self.tokens):
            raise ValueError(f"invalid token(s) at position {self.pos}")
        return result

    # --------------------------------------------------------------------- #
    # Private helpers
    # --------------------------------------------------------------------- #
    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert the raw string into a list of tokens.

        Tokens are:
            - numbers (as strings, e.g. ``'3.14'``)
            - operators: ``+ - * / ( )``
            - whitespace is ignored.

        The tokenizer does **not** treat a leading ``-`` as part of a number;
        unary minus is recognised by the parser.
        """
        tokens: List[str] = []
        i = 0
        n = len(expr)

        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue

            if ch.isdigit() or ch == '.':
                # parse a (decimal) number
                start = i
                dot_seen = (ch == '.')
                i += 1
                while i < n and (expr[i].isdigit() or (expr[i] == '.' and not dot_seen)):
                    if expr[i] == '.':
                        dot_seen = True
                    i += 1
                num_str = expr[start:i]
                if num_str.count('.') > 1:
                    raise ValueError(f"invalid number format: '{num_str}'")
                tokens.append(num_str)
                continue

            if ch in '+-*/()':
                tokens.append(ch)
                i += 1
                continue

            raise ValueError(f"invalid character '{ch}' in expression")

        return tokens

    def _peek(self) -> Optional[str]:
        """Return the current token without consuming it."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _advance(self) -> None:
        """Move to the next token."""
        self.pos += 1

    # --------------------------------------------------------------------- #
    # Recursive‑descent parsing (precedence climbing)
    # --------------------------------------------------------------------- #
    def _parse_expression(self) -> float:
        """
        Parse an expression consisting of terms separated by ``+`` or ``-``.
        """
        value = self._parse_term()
        while (op := self._peek()) in ('+', '-'):
            self._advance()
            rhs = self._parse_term()
            if op == '+':
                value += rhs
            else:
                value -= rhs
        return value

    def _parse_term(self) -> float:
        """
        Parse a term consisting of factors separated by ``*`` or ``/``.
        """
        value = self._parse_factor()
        while (op := self._peek()) in ('*', '/'):
            self._advance()
            rhs = self._parse_factor()
            if op == '*':
                value *= rhs
            else:
                if rhs == 0:
                    raise ValueError("division by zero")
                value /= rhs
        return value

    def _parse_factor(self) -> float:
        """
        Parse a factor, which may be:
            * a parenthesised sub‑expression,
            * a unary minus followed by a factor,
            * a numeric literal.
        """
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end of input")

        if token == '(':
            self._advance()
            value = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("mismatched parentheses")
            self._advance()          # consume ')'
            return value

        if token == '-':
            # Unary minus
            self._advance()
            return -self._parse_factor()

        # Must be a number
        self._advance()
        try:
            return float(token)
        except ValueError as exc:
            raise ValueError(f"invalid number token '{token}'") from exc

import pytest



def test_simple_addition():
    evaluator = ExpressionEvaluator("2+3")
    assert evaluator.evaluate("2+3") == 5.0


def test_precedence_and_operators():
    evaluator = ExpressionEvaluator("2*3+4")
    # 2*3 = 6, 6+4 = 10
    assert evaluator.evaluate("2*3+4") == 10.0


def test_unary_minus_and_parentheses():
    evaluator = ExpressionEvaluator("-(2+1)*5")
    # (2+1) = 3, unary minus -> -3, *5 -> -15
    assert evaluator.evaluate("-(2+1)*5") == -15.0


def test_division_by_zero():
    evaluator = ExpressionEvaluator("5/0")
    with pytest.raises(ValueError, match="division by zero"):
        evaluator.evaluate("5/0")


def test_mismatched_parentheses():
    evaluator = ExpressionEvaluator("((1+2")
    with pytest.raises(ValueError, match="mismatched parentheses"):
        evaluator.evaluate("((1+2")