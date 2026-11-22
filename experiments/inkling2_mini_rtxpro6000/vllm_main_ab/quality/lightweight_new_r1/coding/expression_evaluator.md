Here's a complete implementation — a recursive descent parser split into a tokenizer, a 4-level grammar (expression → term → factor → primary), and a small pytest suite.

## `expression_evaluator.py`

```python
"""Recursive-descent mathematical expression evaluator.

Supports +, -, *, / with correct precedence, parentheses, unary minus,
and floating-point numbers. Raises ValueError with descriptive messages
for empty expressions, invalid tokens, mismatched parentheses, and
division by zero. Uses no eval()/ast.literal_eval().
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class _Token:
    """A single lexical token produced by the tokenizer.

    Attributes:
        kind: Token type, e.g. 'NUMBER', 'PLUS', 'MINUS', 'STAR', 'SLASH',
            'LPAREN', 'RPAREN'.
        text: Raw source text of the token.
        position: Character offset in the source where the token starts.
    """

    kind: str
    text: str
    position: int


class ExpressionEvaluator:
    """Evaluate arithmetic expressions with a recursive descent parser.

    Grammar (lowest to highest precedence):

        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := ('-' | '+')* primary
        primary    := number | '(' expression ')'
    """

    def __init__(self) -> None:
        """Initialize internal parser state."""
        self._tokens: list[_Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate `expr` and return the result as a float.

        Args:
            expr: The expression to evaluate (e.g. "3.14 * (2 + 1)").

        Returns:
            The numeric value of the expression.

        Raises:
            ValueError: If the expression is empty, contains invalid
                tokens, has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise ValueError(
                f"Expression must be a string, got {type(expr).__name__}"
            )
        if not expr.strip():
            raise ValueError("Empty expression")

        # (Re)initialize parser state so the instance is reusable.
        self._tokens = self._tokenize(expr)
        self._pos = 0

        value = self._parse_expression()

        leftover = self._current()
        if leftover is not None:
            if leftover.kind == 'RPAREN':
                raise ValueError(
                    f"Mismatched parentheses: unexpected ')' at position {leftover.position}"
                )
            if leftover.kind == 'NUMBER':
                raise ValueError(
                    f"Unexpected number {leftover.text!r} at position {leftover.position} "
                    "(expected an operator)"
                )
            raise ValueError(
                f"Unexpected token {leftover.text!r} at position {leftover.position}"
            )
        return value

    def _tokenize(self, expr: str) -> list[_Token]:
        """Tokenize `expr`. Raises ValueError on characters outside the grammar."""
        tokens: list[_Token] = []
        i = 0
        while i < len(expr):
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            number = re.match(r'(?:\d+(?:\.\d+)?|\.\d+)', expr[i:])
            if number is not None:
                tokens.append(_Token('NUMBER', number.group(0), i))
                i += len(number.group(0))
                continue
            kind = {
                '+': 'PLUS', '-': 'MINUS', '*': 'STAR', '/': 'SLASH',
                '(': 'LPAREN', ')': 'RPAREN',
            }.get(ch)
            if kind is not None:
                tokens.append(_Token(kind, ch, i))
                i += 1
                continue
            raise ValueError(f"Invalid token {ch!r} at position {i}")
        return tokens

    def _current(self) -> _Token | None:
        """Return the token at the current position, or None at end of input."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _current_kind(self) -> str | None:
        """Return the kind of the current token, or None at end of input."""
        token = self._current()
        return token.kind if token is not None else None

    def _advance(self) -> _Token:
        """Consume the current token and return it."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse `term (('+' | '-') term)*` — lowest precedence level."""
        value = self._parse_term()
        while True:
            kind = self._current_kind()
            if kind == 'PLUS':
                self._advance()
                value += self._parse_term()
            elif kind == 'MINUS':
                self._advance()
                value -= self._parse_term()
            else:
                return value

    def _parse_term(self) -> float:
        """Parse `factor (('*' | '/') factor)*` — higher precedence level."""
        value = self._parse_factor()
        while True:
            kind = self._current_kind()
            if kind == 'STAR':
                self._advance()
                value *= self._parse_factor()
            elif kind == 'SLASH':
                self._advance()
                divisor = self._parse_factor()
                if divisor == 0.0:
                    raise ValueError("Division by zero")
                value /= divisor
            else:
                return value

    def _parse_factor(self) -> float:
        """Parse unary operators followed by a primary (unary minus/plus)."""
        kind = self._current_kind()
        if kind == 'MINUS':
            self._advance()
            return -self._parse_factor()
        if kind == 'PLUS':
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a number literal or a parenthesized expression."""
        token = self._current()
        if token is None:
            raise ValueError(
                "Unexpected end of expression: expected a number or '('"
            )
        if token.kind == 'NUMBER':
            self._advance()
            return float(token.text)
        if token.kind == 'LPAREN':
            self._advance()
            value = self._parse_expression()
            closing = self._current()
            if closing is None:
                raise ValueError("Mismatched parentheses: missing closing ')'")
            if closing.kind != 'RPAREN':
                raise ValueError(
                    "Mismatched parentheses: expected ')' but found "
                    f"{closing.text!r} at position {closing.position}"
                )
            self._advance()
            return value
        if token.kind == 'RPAREN':
            raise ValueError(
                f"Mismatched parentheses: unexpected ')' at position {token.position}"
            )
        raise ValueError(
            f"Expected a number or '(' but found {token.text!r} at position {token.position}"
        )
```

## `test_expression_evaluator.py`

```python
"""Pytest tests for the ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


def test_basic_arithmetic():
    """Basic +, -, *, / and floating-point literals."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == pytest.approx(5.0)
    assert evaluator.evaluate("10 - 4") == pytest.approx(6.0)
    assert evaluator.evaluate("3 * 4") == pytest.approx(12.0)
    assert evaluator.evaluate("8 / 2") == pytest.approx(4.0)
    assert evaluator.evaluate("3.14 + 1.86") == pytest.approx(5.0)
    assert evaluator.evaluate("2.5 * 4") == pytest.approx(10.0)


def test_operator_precedence():
    """Multiplication/division bind tighter than addition/subtraction."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == pytest.approx(14.0)   # 2 + 12
    assert evaluator.evaluate("10 - 2 * 3") == pytest.approx(4.0)   # 10 - 6
    assert evaluator.evaluate("2 * 3 + 4 * 5") == pytest.approx(26.0)  # 6 + 20
    assert evaluator.evaluate("100 / 10 / 2") == pytest.approx(5.0)  # left-assoc


def test_parentheses():
    """Parentheses override precedence and support nesting."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == pytest.approx(20.0)
    assert evaluator.evaluate("2 * (3 + 4)") == pytest.approx(14.0)
    assert evaluator.evaluate("((2 + 3))") == pytest.approx(5.0)
    assert evaluator.evaluate("10 / (2 + 3)") == pytest.approx(2.0)
    assert evaluator.evaluate("2 * (3 + (4 * 5))") == pytest.approx(46.0)


def test_unary_minus():
    """Unary minus negates a value, including grouped expressions."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == pytest.approx(-3.0)
    assert evaluator.evaluate("-(2 + 1)") == pytest.approx(-3.0)
    assert evaluator.evaluate("2 + -3") == pytest.approx(-1.0)
    assert evaluator.evaluate("-2 * 3") == pytest.approx(-6.0)
    assert evaluator.evaluate("--3") == pytest.approx(3.0)
    assert evaluator.evaluate("3.14 * -2") == pytest.approx(-6.28)


def test_error_cases():
    """Invalid input raises ValueError with a descriptive message."""
    evaluator = ExpressionEvaluator()

    # Empty expressions
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")      # missing closing paren
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")      # extra closing paren
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("()")          # empty group

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / (2 - 2)")

    # Invalid tokens / malformed expressions
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + @3")
    with pytest.raises(ValueError, match="end of expression"):
        evaluator.evaluate("2 +")
    with pytest.raises(ValueError, match="Unexpected number"):
        evaluator.evaluate("2 3")
```

## How it works

- **Tokenizer** — splits the input into tokens (`NUMBER`, `PLUS`, `MINUS`, `STAR`, `SLASH`, `LPAREN`, `RPAREN`), tracking positions for error messages. Any character outside the grammar raises `ValueError("Invalid token ...")`. Numbers match `\d+(\.\d+)?` or `\.\d+`, so `3.14` and `.5` are both valid.
- **Recursive descent** — four mutually recursive methods map directly onto the grammar. Precedence falls out of the structure: `_parse_expression` handles `+`/`-`, `_parse_term` handles `*`/`/`, and `_parse_factor` handles unary operators before delegating to `_parse_primary`.
- **Error handling** — mismatched parens are caught in `_parse_primary` (missing/unexpected `)`) and in the trailing-token check in `evaluate`; division by zero is checked explicitly since `ZeroDivisionError` is not a subclass of `ValueError`; empty input and non-string input are rejected up front.
- **State** — `evaluate` resets the token/position state on entry, so a single evaluator instance is reusable.

Run the tests with:

```bash
python -m pytest test_expression_evaluator.py -v
```

One note: the evaluator is not thread-safe (it stores parser state on the instance), so concurrent `evaluate()` calls on one instance would interleave — create one evaluator per thread or call `evaluate` from a single thread.