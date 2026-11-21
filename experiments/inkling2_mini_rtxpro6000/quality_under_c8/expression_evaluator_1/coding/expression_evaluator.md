# Expression Evaluator

Here's a complete implementation. Two files:

## `expression_evaluator.py`

```python
"""A safe mathematical expression evaluator.

Uses a recursive-descent parser (no eval() or ast.literal_eval()) to support
+, -, *, /, parentheses, unary minus/plus, and floating-point numbers.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class _Token(NamedTuple):
    """A single lexical token produced by the tokenizer.

    Attributes:
        type: Token kind: 'NUMBER', '+', '-', '*', '/', 'LPAREN', or 'RPAREN'.
        value: The numeric value for NUMBER tokens, otherwise the raw character.
        raw: The exact source text of the token.
        position: 0-based index in the source expression where the token starts.
    """

    type: str
    value: float | str
    raw: str
    position: int


class ExpressionEvaluator:
    """Recursive-descent mathematical expression evaluator.

    Grammar (loosest binding first):

        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := ('+' | '-')* primary
        primary    := NUMBER | '(' expression ')'
    """

    _NUMBER_CHARS = re.compile(r"[0-9.]+")
    _NUMBER_STRICT = re.compile(r"\d+(\.\d*)?|\.\d+")

    def __init__(self) -> None:
        """Initialize the evaluator; parser state is reset on every call."""
        self._tokens: list[_Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate ``expr`` and return the result as a float.

        Args:
            expr: The expression to evaluate, e.g. ``"3.14 * (2 + 1)"``.

        Returns:
            The numeric result of the expression.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens or
                malformed numbers, has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise ValueError(
                f"Expression must be a string, got {type(expr).__name__}"
            )
        if not expr.strip():
            raise ValueError("Empty expression: expected a mathematical expression")

        self._tokens = self._tokenize(expr)
        if not self._tokens:
            raise ValueError("Empty expression: expected a mathematical expression")
        self._pos = 0

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            token = self._tokens[self._pos]
            if token.type == "RPAREN":
                raise ValueError(
                    f"Mismatched parentheses: unexpected ')' at position {token.position}"
                )
            raise ValueError(
                f"Unexpected token '{token.raw}' at position {token.position}"
            )
        return result

    def _tokenize(self, expr: str) -> list[_Token]:
        """Split ``expr`` into a list of tokens.

        Args:
            expr: Raw expression text (non-empty).

        Returns:
            The list of tokens in source order.

        Raises:
            ValueError: On invalid tokens, malformed numbers, or overflowing numbers.
        """
        tokens: list[_Token] = []
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or ch == ".":
                match = self._NUMBER_CHARS.match(expr, i)
                assert match is not None  # guaranteed: ch is digit or '.'
                text = match.group()
                if not self._NUMBER_STRICT.fullmatch(text):
                    raise ValueError(f"Invalid number '{text}' at position {i}")
                try:
                    value = float(text)
                except OverflowError:
                    raise ValueError(
                        f"Number '{text}' at position {i} is too large"
                    ) from None
                tokens.append(_Token("NUMBER", value, text, i))
                i = match.end()
                continue
            if ch in "+-*/":
                tokens.append(_Token(ch, ch, ch, i))
                i += 1
                continue
            if ch == "(":
                tokens.append(_Token("LPAREN", "(", "(", i))
                i += 1
                continue
            if ch == ")":
                tokens.append(_Token("RPAREN", ")", ")", i))
                i += 1
                continue
            raise ValueError(f"Invalid token '{ch}' at position {i}")
        return tokens

    def _peek(self) -> _Token | None:
        """Return the current token without consuming it, or None at end of input."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> _Token:
        """Consume and return the current token."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse ``expression := term (('+' | '-') term)*`` (lowest precedence)."""
        left = self._parse_term()
        while (token := self._peek()) is not None and token.type in ("+", "-"):
            self._advance()
            right = self._parse_term()
            left = left + right if token.type == "+" else left - right
        return left

    def _parse_term(self) -> float:
        """Parse ``term := factor (('*' | '/') factor)*``."""
        left = self._parse_factor()
        while (token := self._peek()) is not None and token.type in ("*", "/"):
            self._advance()
            right = self._parse_factor()
            if token.type == "*":
                left = left * right
            else:
                if right == 0:
                    raise ValueError(
                        f"Division by zero at position {token.position}: {left} / {right}"
                    )
                left = left / right
        return left

    def _parse_factor(self) -> float:
        """Parse ``factor := ('+' | '-')* primary`` (unary operators)."""
        token = self._peek()
        if token is not None and token.type == "-":
            self._advance()
            return -self._parse_factor()
        if token is not None and token.type == "+":
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse ``primary := NUMBER | '(' expression ')'``."""
        token = self._peek()
        if token is None:
            raise ValueError(
                "Unexpected end of expression: expected a number or '('"
            )
        if token.type == "NUMBER":
            self._advance()
            return float(token.value)
        if token.type == "LPAREN":
            self._advance()
            result = self._parse_expression()
            closing = self._peek()
            if closing is None:
                raise ValueError(
                    "Mismatched parentheses: expected ')' but reached end of expression"
                )
            if closing.type != "RPAREN":
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found '{closing.raw}'"
                )
            self._advance()
            return result
        if token.type == "RPAREN":
            raise ValueError("Mismatched parentheses: unexpected ')'")
        raise ValueError(
            f"Unexpected token '{token.raw}': expected a number or '('"
        )
```

## `test_expression_evaluator.py`

```python
"""Pytest tests for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


def test_basic_arithmetic() -> None:
    """Basic +, -, *, / operations and floating-point numbers."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3") == 5.0
    assert ev.evaluate("10 - 4") == 6.0
    assert ev.evaluate("6 * 7") == 42.0
    assert ev.evaluate("8 / 2") == 4.0
    assert ev.evaluate("1.5 + 2.25") == pytest.approx(3.75)
    assert ev.evaluate("10 / 4") == pytest.approx(2.5)


def test_operator_precedence() -> None:
    """Multiplication/division bind tighter than addition/subtraction."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 / 2 - 3") == 2.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0
    assert ev.evaluate("100 - 2 * 3") == 94.0


def test_parentheses() -> None:
    """Parentheses override the default operator precedence."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("2 * (3 + 4)") == 14.0
    assert ev.evaluate("((1 + 2) * 3)") == 9.0
    assert ev.evaluate("(3 + 4) * (5 - 2)") == 21.0


def test_unary_minus() -> None:
    """Unary minus (and unary plus) are supported."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("3 + -2") == 1.0
    assert ev.evaluate("--5") == 5.0
    assert ev.evaluate("-2.5 * 2") == -5.0
    assert ev.evaluate("+4") == 4.0


def test_error_cases() -> None:
    """Mismatched parens, division by zero, invalid tokens, and empty input."""
    ev = ExpressionEvaluator()

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(3 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("3 + 2)")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate(")")

    # Division by zero (including zero produced by an expression)
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / (2 - 2)")

    # Invalid tokens / malformed numbers
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("3 @ 4")
    with pytest.raises(ValueError, match="Invalid number"):
        ev.evaluate("3.4.5")

    # Empty expressions
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")

    # Truncated input and stray tokens
    with pytest.raises(ValueError, match="Unexpected end"):
        ev.evaluate("3 +")
    with pytest.raises(ValueError, match="Unexpected token"):
        ev.evaluate("3 4")
```

## Design notes

- **Recursive descent**: the grammar is layered so precedence falls out naturally — `expression` → `term` → `factor` → `primary`, with unary operators handled in `factor` (above `primary`) and parens in `primary` (which recursively calls `expression`).
- **Tokenizer** runs first and validates numbers strictly (`\d+(\.\d*)?|\.\d+`), so malformed numbers like `3.4.5` or `.` raise descriptive `ValueError`s at tokenize time.
- **Error handling**: every failure mode raises `ValueError` with the offending position/text (e.g., `Invalid token '@' at position 2`). Division by zero is checked at parse time in `_parse_term`.
- **State reset**: `evaluate()` re-initializes `self._tokens`/`self._pos` on every call, so the same evaluator instance can be reused safely.

## Running the tests

```bash
python -m pytest test_expression_evaluator.py -v
```

This prints the 5 test functions (each covering multiple assertions) — all should pass.