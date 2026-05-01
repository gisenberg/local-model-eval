"""
ExpressionEvaluator – a tiny recursive‑descent calculator.

Supported syntax
----------------
* Binary operators: ``+``, ``-``, ``*``, ``/`` (with the usual precedence)
* Parentheses for grouping
* Unary minus (e.g. ``-3`` or ``-(2+1)``)
* Floating‑point numbers (e.g. ``3.14``)

The public API is the ``ExpressionEvaluator`` class with a single
``evaluate(expr: str) -> float`` method.  All errors are reported as
``ValueError`` with an explanatory message.

The implementation does **not** use ``eval`` or ``ast.literal_eval``.
"""

from __future__ import annotations

from typing import List, Union


class ExpressionEvaluator:
    """
    Evaluate a simple arithmetic expression.

    Parameters
    ----------
    expr : str
        The arithmetic expression to evaluate.

    Returns
    -------
    float
        The numeric result of the expression.

    Raises
    ------
    ValueError
        If the expression is empty, contains mismatched parentheses,
        division by zero, an invalid token, or any other syntactic error.
    """

    def evaluate(self, expr: str) -> float:
        """
        Parse and evaluate *expr*.

        Parameters
        ----------
        expr : str
            The expression to evaluate.

        Returns
        -------
        float
            The computed value.

        Raises
        ------
        ValueError
            For any syntax or arithmetic error.
        """
        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Empty expression")
        parser = _Parser(tokens)
        return parser.parse()

    # --------------------------------------------------------------------- #
    # Tokenisation
    # --------------------------------------------------------------------- #
    def _tokenize(self, expr: str) -> List[Union[float, str]]:
        """
        Convert the input string into a list of tokens.

        Tokens are either a ``float`` (numeric literal) or a single‑character
        operator/parenthesis.

        Raises
        ------
        ValueError
            If an illegal character is encountered.
        """
        tokens: List[Union[float, str]] = []
        i = 0
        length = len(expr)

        while i < length:
            ch = expr[i]

            if ch.isspace():
                i += 1
                continue

            if ch.isdigit() or ch == '.':
                # parse a floating‑point number
                start = i
                while i < length and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                num_str = expr[start:i]
                try:
                    tokens.append(float(num_str))
                except ValueError as exc:
                    raise ValueError(f"Invalid number token: {num_str}") from exc

            elif ch in "+-*/()":
                tokens.append(ch)
                i += 1

            else:
                raise ValueError(f"Invalid character: {ch}")

        return tokens


# ------------------------------------------------------------------------- #
# Recursive‑descent parser (inner class – keeps the public API tidy)
# ------------------------------------------------------------------------- #
class _Parser:
    """
    Recursive‑descent parser for the token list produced by ``_tokenize``.
    """

    def __init__(self, tokens: List[Union[float, str]]) -> None:
        self.tokens = tokens
        self.pos = 0  # current token index

    # --------------------------------------------------------------------- #
    # Entry point
    # --------------------------------------------------------------------- #
    def parse(self) -> float:
        """Parse the whole token stream and return the result."""
        return self.expr()

    # --------------------------------------------------------------------- #
    # Grammar rules (left‑associative, precedence‑aware)
    # --------------------------------------------------------------------- #
    def expr(self) -> float:
        """
        expr ::= term (('+' | '-') term)*

        Parse a sum/difference of terms.
        """
        value = self.term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ("+", "-"):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.term()
            if op == "+":
                value += right
            else:
                value -= right
        return value

    def term(self) -> float:
        """
        term ::= factor (('*' | '/') factor)*

        Parse a product/quotient of factors.
        """
        value = self.factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ("*", "/"):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.factor()
            if op == "*":
                value *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                value /= right
        return value

    def factor(self) -> float:
        """
        factor ::= ('-' factor)? | number | '(' expr ')'

        Handles unary minus, parentheses, and numeric literals.
        """
        # Unary minus
        if self.pos < len(self.tokens) and self.tokens[self.pos] == "-":
            self.pos += 1
            return -self.factor()

        # Parenthesised sub‑expression
        if self.pos < len(self.tokens) and self.tokens[self.pos] == "(":
            self.pos += 1
            value = self.expr()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ")":
                raise ValueError("Mismatched parentheses")
            self.pos += 1  # consume ')'
            return value

        # Numeric literal
        if self.pos < len(self.tokens) and isinstance(self.tokens[self.pos], float):
            value = self.tokens[self.pos]
            self.pos += 1
            return value

        raise ValueError("Invalid token in factor")


# ------------------------------------------------------------------------- #
# Example usage (not part of the required library, but handy for REPL)
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    evaler = ExpressionEvaluator()
    examples = [
        "2 + 3 * 4",
        "-(2 + 1) * 5",
        "3.14 * (1 + 2)",
        "-5 + 2",
    ]
    for e in examples:
        print(f"{e} = {evaler.evaluate(e)}")

import pytest


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Provide a fresh evaluator instance for each test."""
    return ExpressionEvaluator()


def test_basic_addition(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0


def test_operator_precedence(evaluator: ExpressionEvaluator) -> None:
    # multiplication has higher precedence than addition
    assert evaluator.evaluate("2 * 3 + 4") == 10.0
    # division and multiplication have the same precedence, left‑to‑right
    assert evaluator.evaluate("8 / 2 * 3") == 12.0


def test_unary_minus(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-3 + 4") == 1.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-( -5 )") == 5.0


def test_parentheses(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("3 * (2 + 1)") == 9.0
    # nested parentheses
    assert evaluator.evaluate("((1+2) * (3-1))") == 6.0


def test_error_conditions(evaluator: ExpressionEvaluator) -> None:
    # division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

    # mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + (3")

    # empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")

    # invalid token
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + a")