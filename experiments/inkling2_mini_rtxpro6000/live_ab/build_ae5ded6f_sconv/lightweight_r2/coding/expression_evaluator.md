Here's a complete implementation. It uses a classic recursive-descent design: a tokenizer splits the input into tokens, then a grammar with three precedence levels (expression → term → factor) evaluates it.

## `evaluator.py`

```python
"""Recursive-descent mathematical expression evaluator."""

from __future__ import annotations

from typing import NamedTuple


class _Token(NamedTuple):
    """A single lexical token: kind ('number'|'op'|'lparen'|'rparen'), value, position."""

    kind: str
    value: str
    position: int


class ExpressionEvaluator:
    """Evaluate arithmetic expressions using recursive descent.

    Supports '+', '-', '*', '/', parentheses, unary minus, and
    floating-point numbers. All failures are raised as ValueError
    with a descriptive message.
    """

    def evaluate(self, expr: str) -> float:
        """Evaluate an arithmetic expression.

        Args:
            expr: The expression string, e.g. "(3.14 * 2) - 1".

        Returns:
            The numeric result as a float.

        Raises:
            ValueError: If expr is not a string, is empty, contains invalid
                tokens, has mismatched parentheses, has unexpected trailing
                tokens, or divides by zero.
        """
        if not isinstance(expr, str):
            raise ValueError(f"Expression must be a string, got {type(expr).__name__}")
        if not expr.strip():
            raise ValueError("Expression cannot be empty")

        self._tokens: list[_Token] = self._tokenize(expr)
        self._pos: int = 0
        result = self._parse_expression()

        if self._pos < len(self._tokens):
            leftover = self._tokens[self._pos]
            if leftover.kind == "rparen":
                raise ValueError("Mismatched parentheses: unexpected ')'")
            raise ValueError(
                f"Unexpected token {leftover.value!r} at position {leftover.position}"
            )
        return float(result)

    def _tokenize(self, expr: str) -> list[_Token]:
        """Split an expression string into tokens (whitespace discarded).

        Args:
            expr: The raw expression string.

        Returns:
            A list of _Token objects.

        Raises:
            ValueError: If a character or number is not recognized.
        """
        tokens: list[_Token] = []
        i, n = 0, len(expr)
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
                try:
                    float(text)
                except ValueError:
                    raise ValueError(
                        f"Invalid number {text!r} at position {start}"
                    ) from None
                tokens.append(_Token("number", text, start))
                continue
            if ch in "+-*/":
                tokens.append(_Token("op", ch, i))
                i += 1
                continue
            if ch == "(":
                tokens.append(_Token("lparen", ch, i))
                i += 1
                continue
            if ch == ")":
                tokens.append(_Token("rparen", ch, i))
                i += 1
                continue
            raise ValueError(f"Invalid token {ch!r} at position {i}")
        if not tokens:
            raise ValueError("Expression cannot be empty")
        return tokens

    def _peek(self) -> _Token | None:
        """Return the current token without consuming it (or None at the end)."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> _Token:
        """Consume and return the current token."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse an expression: term (('+' | '-') term)*."""
        value = self._parse_term()
        token = self._peek()
        while token is not None and token.kind == "op" and token.value in ("+", "-"):
            op = self._advance().value
            right = self._parse_term()
            value = value + right if op == "+" else value - right
            token = self._peek()
        return value

    def _parse_term(self) -> float:
        """Parse a term: factor (('*' | '/') factor)*."""
        value = self._parse_factor()
        token = self._peek()
        while token is not None and token.kind == "op" and token.value in ("*", "/"):
            op = self._advance().value
            right = self._parse_factor()
            if op == "*":
                value = value * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                value = value / right
            token = self._peek()
        return value

    def _parse_factor(self) -> float:
        """Parse a factor: ('+' | '-')* primary — handles unary operators."""
        token = self._peek()
        if token is not None and token.kind == "op" and token.value == "-":
            self._advance()
            return -self._parse_factor()
        if token is not None and token.kind == "op" and token.value == "+":
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a primary: a number or a parenthesized expression."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression: expected a number or '('")
        if token.kind == "number":
            self._advance()
            return float(token.value)
        if token.kind == "lparen":
            self._advance()
            value = self._parse_expression()
            closing = self._peek()
            if closing is None:
                raise ValueError(
                    "Mismatched parentheses: expected ')' but reached end of expression"
                )
            if closing.kind != "rparen":
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found {closing.value!r}"
                )
            self._advance()
            return value
        if token.kind == "rparen":
            raise ValueError("Mismatched parentheses: unexpected ')'")
        raise ValueError(f"Expected a number or '(' but found {token.value!r}")


if __name__ == "__main__":
    evaluator = ExpressionEvaluator()
    for sample in ["2 + 3 * 4", "(2 + 3) * 4", "-(2 + 1)", "3.14 * 2", "10 / 4"]:
        print(f"{sample!r:20} = {evaluator.evaluate(sample)}")
```

## `test_evaluator.py`

```python
"""Tests for the ExpressionEvaluator class."""

import pytest

from evaluator import ExpressionEvaluator


def test_basic_arithmetic():
    """Basic +, -, *, / operations and floating-point numbers."""
    ev = ExpressionEvaluator()
    assert isinstance(ev.evaluate("1 + 1"), float)
    assert ev.evaluate("2 + 3") == 5.0
    assert ev.evaluate("10 - 4") == 6.0
    assert ev.evaluate("6 * 7") == 42.0
    assert ev.evaluate("9 / 2") == 4.5
    assert ev.evaluate("3.14") == pytest.approx(3.14)
    assert ev.evaluate("2.5 + 1.5") == pytest.approx(4.0)


def test_operator_precedence():
    """Multiplication/division bind tighter than addition/subtraction."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 - 6 / 3") == 8.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0
    assert ev.evaluate("100 / 10 / 2") == 5.0  # left-associative


def test_parentheses():
    """Parentheses override normal precedence."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("2 * (3 + 4)") == 14.0
    assert ev.evaluate("((2 + 3) * 4)") == 20.0
    assert ev.evaluate("(1 + 2) * (3 + 4)") == 21.0
    assert ev.evaluate("2 + (3 * (4 - 1))") == 11.0


def test_unary_minus():
    """Unary minus (and double negation) is handled."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("2 * -3") == -6.0
    assert ev.evaluate("3 + -2") == 1.0
    assert ev.evaluate("-2 - -3") == 1.0
    assert ev.evaluate("--5") == 5.0


def test_error_cases():
    """Empty expressions, mismatched parens, division by zero, invalid tokens."""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="empty"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="empty"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("1 + 2)")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / (2 - 2)")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + $3")
    with pytest.raises(ValueError, match="Invalid number"):
        ev.evaluate("1.2.3")
    with pytest.raises(ValueError):
        ev.evaluate("2 3")  # unexpected trailing token
```

## How it works

The grammar mirrors precedence levels, so the parser naturally handles precedence and left-associativity:

```
expression := term (('+' | '-') term)*      # lowest precedence
term       := factor (('*' | '/') factor)*
factor     := ('+' | '-')* primary          # unary operators
primary    := number | '(' expression ')'
```

Each `_parse_*` method consumes tokens via `_advance()` and uses `_peek()` to decide what to do next — no `eval()` or `ast` involved. Division by zero is caught at evaluation time in `_parse_term`, and mismatched parens are caught when the closing `)` is missing or wrong.

## Running

```bash
pytest test_evaluator.py -v
```

One design note: repeated calls to `evaluate()` are safe because the parser state (`self._tokens`, `self._pos`) is reset on every call. If you need thread-safety for concurrent `evaluate()` calls on the same instance, you'd either create the parser state locally (pass it as arguments) or use a fresh `ExpressionEvaluator` per call — I kept it instance-level for simplicity, which is typical for this pattern.