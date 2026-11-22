Here's a complete recursive descent implementation with tests.

## `expression_evaluator.py`

```python
"""Recursive descent parser for arithmetic expressions.

Supports +, -, *, /, parentheses, unary minus, and floating-point numbers.
No eval() or ast.literal_eval() is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class _Token:
    """A single lexical token."""

    type: str       # 'NUMBER', 'OP', 'LPAREN', 'RPAREN', or 'END'
    value: str      # raw text of the token
    position: int   # index in the source string


class ExpressionEvaluator:
    """Evaluate arithmetic expressions via recursive descent.

    Example: "3.14 * 2 + (1 - 0.5)" evaluates to 6.78.
    """

    def __init__(self) -> None:
        """Initialize the evaluator with empty state."""
        self._tokens: list[_Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate an arithmetic expression and return the result.

        Args:
            expr: Expression to evaluate, e.g. "2 + 3 * (4 - 1)".

        Returns:
            The computed value of the expression.

        Raises:
            TypeError: If `expr` is not a string.
            ValueError: If the expression is empty, contains invalid tokens,
                has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"Expression must be a string, got {type(expr).__name__}")
        if expr.strip() == "":
            raise ValueError("Expression is empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        token = self._current()
        if token.type == "END":
            return float(result)
        if token.type == "RPAREN":
            raise ValueError(
                f"Mismatched parentheses: unexpected ')' at position {token.position}"
            )
        raise ValueError(
            f"Unexpected token '{token.value}' at position {token.position}"
        )

    def _tokenize(self, expr: str) -> list[_Token]:
        """Split `expr` into tokens, skipping whitespace."""
        tokens: list[_Token] = []
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or ch == ".":
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == "."):
                    i += 1
                text = expr[start:i]
                if re.fullmatch(r"(\d+(\.\d*)?|\.\d+)", text) is None:
                    raise ValueError(f"Invalid number '{text}' at position {start}")
                tokens.append(_Token("NUMBER", text, start))
                continue
            if ch in "+-*/":
                tokens.append(_Token("OP", ch, i))
                i += 1
                continue
            if ch == "(":
                tokens.append(_Token("LPAREN", ch, i))
                i += 1
                continue
            if ch == ")":
                tokens.append(_Token("RPAREN", ch, i))
                i += 1
                continue
            raise ValueError(f"Invalid token '{ch}' at position {i}")
        tokens.append(_Token("END", "", n))
        return tokens

    def _current(self) -> _Token:
        """Return the token currently being processed."""
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        """Consume the current token and return it."""
        token = self._tokens[self._pos]
        if token.type != "END":
            self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse an additive expression: term (('+' | '-') term)*."""
        result = self._parse_term()
        while self._current().type == "OP" and self._current().value in ("+", "-"):
            op = self._advance().value
            rhs = self._parse_term()
            result = result + rhs if op == "+" else result - rhs
        return result

    def _parse_term(self) -> float:
        """Parse a multiplicative term: factor (('*' | '/') factor)*."""
        result = self._parse_factor()
        while self._current().type == "OP" and self._current().value in ("*", "/"):
            op = self._advance().value
            rhs = self._parse_factor()
            if op == "*":
                result = result * rhs
            else:
                if rhs == 0:
                    raise ValueError("Division by zero")
                result = result / rhs
        return result

    def _parse_factor(self) -> float:
        """Parse a factor: optional unary '+'/'-' followed by a primary."""
        token = self._current()
        if token.type == "OP" and token.value == "-":
            self._advance()
            return -self._parse_factor()
        if token.type == "OP" and token.value == "+":
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a primary: a number literal or a parenthesized expression."""
        token = self._current()
        if token.type == "NUMBER":
            self._advance()
            return float(token.value)
        if token.type == "LPAREN":
            self._advance()
            result = self._parse_expression()
            closing = self._current()
            if closing.type == "END":
                raise ValueError("Mismatched parentheses: expected ')' at end of expression")
            if closing.type != "RPAREN":
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found "
                    f"'{closing.value}' at position {closing.position}"
                )
            self._advance()
            return result
        if token.type == "END":
            raise ValueError("Expected a number or '(' but found end of expression")
        if token.type == "RPAREN":
            raise ValueError(
                f"Mismatched parentheses: unexpected ')' at position {token.position}"
            )
        raise ValueError(
            f"Expected a number or '(' but found '{token.value}' at position {token.position}"
        )
```

## `test_expression_evaluator.py`

```python
"""Tests for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == pytest.approx(5.0)
    assert evaluator.evaluate("10 - 4") == pytest.approx(6.0)
    assert evaluator.evaluate("6 * 7") == pytest.approx(42.0)
    assert evaluator.evaluate("20 / 5") == pytest.approx(4.0)
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == pytest.approx(14.0)
    assert evaluator.evaluate("10 - 2 * 3") == pytest.approx(4.0)
    assert evaluator.evaluate("2 * 3 + 4 * 5") == pytest.approx(26.0)
    assert evaluator.evaluate("8 / 2 - 1") == pytest.approx(3.0)


def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == pytest.approx(20.0)
    assert evaluator.evaluate("2 * (3 + 4)") == pytest.approx(14.0)
    assert evaluator.evaluate("((2 + 3) * 4)") == pytest.approx(20.0)
    assert evaluator.evaluate("2 + (3 * 4)") == pytest.approx(14.0)


def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == pytest.approx(-3.0)
    assert evaluator.evaluate("-(2 + 1)") == pytest.approx(-3.0)
    assert evaluator.evaluate("3 - -2") == pytest.approx(5.0)
    assert evaluator.evaluate("2 * -3") == pytest.approx(-6.0)
    assert evaluator.evaluate("-3 * 2") == pytest.approx(-6.0)


@pytest.mark.parametrize(
    "expr,match",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("2 +", "end of expression"),
        ("(2 + 3", "Mismatched"),
        ("2 + 3)", "Mismatched"),
        ("(2 + 3))", "Mismatched"),
        ("2 & 3", "Invalid token"),
        ("3.4.5", "Invalid number"),
        ("4 / 0", "Division by zero"),
        ("1 / (2 - 2)", "Division by zero"),
    ],
)
def test_error_cases(evaluator, expr, match):
    with pytest.raises(ValueError, match=match):
        evaluator.evaluate(expr)
```

## How it works

The parser uses a classic precedence-climbing grammar:

```
expression := term (('+' | '-') term)*
term        := factor (('*' | '/') factor)*
factor      := ('-' | '+')* primary        # unary minus/plus
primary     := NUMBER | '(' expression ')'
```

- **Precedence** falls out naturally: `_parse_expression` only sees `+`/`-`, `_parse_term` only sees `*`/`/`, and unary operators are handled one level down in `_parse_factor`.
- **Unary minus** works because `_parse_factor` recurses, so `-3 * 2` → `(-3) * 2` and `2 * -3` → `2 * (-3)`.
- **Errors** are raised with the offending position and token: mismatched parens (both directions), invalid tokens/numbers, empty expressions, and division by zero (explicitly checked, since `1.0 / 0.0` silently returns `inf` in Python).
- **Tokenizer** validates numbers with a regex (`3.14`, `3.`, `.5` are valid; `3.4.5` and bare `.` are not).

Run the tests with:

```bash
python -m pytest test_expression_evaluator.py -v
```

One note: the parser intentionally does **not** support scientific notation (`1e5`) or exponents — anything outside the listed grammar is reported as an invalid token rather than silently accepted. If you want `^` or `e` notation, that would be one more grammar level, but I've kept the scope to the stated requirements.