Here's a complete solution split into two files: the evaluator and the tests.

## `expression_evaluator.py`

```python
"""Recursive descent mathematical expression evaluator."""

from dataclasses import dataclass


@dataclass
class _Token:
    """A single lexical token produced by the tokenizer."""

    type: str  # 'number' | 'op' | 'lparen' | 'rparen' | 'eof'
    value: str
    position: int


class ExpressionEvaluator:
    """Evaluates arithmetic expressions with +, -, *, /, parentheses, and unary minus.

    Uses a tokenizer plus a recursive descent parser (no eval/ast). Grammar:

        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := '-' factor | '+' factor | number | '(' expression ')'
    """

    def __init__(self) -> None:
        """Initialize parser state (token stream and cursor)."""
        self._tokens: list[_Token] = []
        self._pos: int = 0

    # -- public API ------------------------------------------------------

    def evaluate(self, expr: str) -> float:
        """Evaluate ``expr`` and return the result.

        Args:
            expr: The mathematical expression to evaluate.

        Returns:
            The numeric result of the expression.

        Raises:
            TypeError: If ``expr`` is not a string.
            ValueError: If the expression is empty, contains invalid tokens,
                has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"Expression must be a string, got {type(expr).__name__}")
        if not expr.strip():
            raise ValueError("Empty expression: expected a mathematical expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0
        result = self._parse_expression()

        trailing = self._current()
        if trailing.type == 'rparen':
            raise ValueError(
                f"Mismatched parentheses: unexpected ')' at position {trailing.position}"
            )
        if trailing.type != 'eof':
            raise ValueError(
                f"Unexpected token '{trailing.value}' at position {trailing.position}"
            )
        return result

    # -- tokenizer --------------------------------------------------------

    def _tokenize(self, expr: str) -> list[_Token]:
        """Split ``expr`` into tokens, raising ValueError on invalid characters."""
        tokens: list[_Token] = []
        i, n = 0, len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or ch == '.':
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                text = expr[start:i]
                try:
                    float(text)
                except ValueError:
                    raise ValueError(
                        f"Invalid number '{text}' at position {start}"
                    ) from None
                tokens.append(_Token('number', text, start))
                continue
            if ch in '+-*/':
                tokens.append(_Token('op', ch, i))
                i += 1
                continue
            if ch == '(':
                tokens.append(_Token('lparen', ch, i))
                i += 1
                continue
            if ch == ')':
                tokens.append(_Token('rparen', ch, i))
                i += 1
                continue
            raise ValueError(f"Invalid token '{ch}' at position {i}")
        tokens.append(_Token('eof', '', n))
        return tokens

    # -- parser helpers ---------------------------------------------------

    def _current(self) -> _Token:
        """Return the token currently being inspected."""
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        """Consume and return the current token."""
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    # -- recursive descent grammar ----------------------------------------

    def _parse_expression(self) -> float:
        """Parse the additive level (+ and -), left associative."""
        left = self._parse_term()
        while self._current().type == 'op' and self._current().value in ('+', '-'):
            op = self._advance().value
            right = self._parse_term()
            left = left + right if op == '+' else left - right
        return left

    def _parse_term(self) -> float:
        """Parse the multiplicative level (* and /), left associative."""
        left = self._parse_factor()
        while self._current().type == 'op' and self._current().value in ('*', '/'):
            op = self._advance().value
            right = self._parse_factor()
            if op == '*':
                left = left * right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                left = left / right
        return left

    def _parse_factor(self) -> float:
        """Parse a factor: unary +/-, a number, or a parenthesized expression."""
        tok = self._current()
        if tok.type == 'op' and tok.value == '-':
            self._advance()
            return -self._parse_factor()
        if tok.type == 'op' and tok.value == '+':
            self._advance()
            return self._parse_factor()
        if tok.type == 'number':
            self._advance()
            return float(tok.value)
        if tok.type == 'lparen':
            self._advance()
            value = self._parse_expression()
            closing = self._current()
            if closing.type == 'rparen':
                self._advance()
                return value
            if closing.type == 'eof':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            raise ValueError(
                f"Mismatched parentheses: expected ')' but found '{closing.value}' "
                f"at position {closing.position}"
            )
        if tok.type == 'rparen':
            raise ValueError(
                f"Mismatched parentheses: unexpected ')' at position {tok.position}"
            )
        if tok.type == 'eof':
            raise ValueError(
                "Unexpected end of expression: expected a number, '(', or unary operator"
            )
        raise ValueError(f"Unexpected token '{tok.value}' at position {tok.position}")
```

## `test_expression_evaluator.py`

```python
"""Tests for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


def test_basic_arithmetic() -> None:
    ev = ExpressionEvaluator()
    assert ev.evaluate("1 + 2") == pytest.approx(3.0)
    assert ev.evaluate("10 - 4") == pytest.approx(6.0)
    assert ev.evaluate("6 * 7") == pytest.approx(42.0)
    assert ev.evaluate("8 / 2") == pytest.approx(4.0)
    assert ev.evaluate("1.5 + 2.25") == pytest.approx(3.75)  # floating point


def test_operator_precedence() -> None:
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == pytest.approx(14.0)
    assert ev.evaluate("10 - 4 / 2") == pytest.approx(8.0)
    assert ev.evaluate("2 * 3 + 4 * 5") == pytest.approx(26.0)


def test_parentheses() -> None:
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == pytest.approx(20.0)
    assert ev.evaluate("((2 + 3))") == pytest.approx(5.0)
    assert ev.evaluate("2 * (3 + 4)") == pytest.approx(14.0)


def test_unary_minus() -> None:
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == pytest.approx(-3.0)
    assert ev.evaluate("-(2 + 1)") == pytest.approx(-3.0)
    assert ev.evaluate("2 + -3") == pytest.approx(-1.0)
    assert ev.evaluate("3 * -2") == pytest.approx(-6.0)


def test_error_cases() -> None:
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("1 + 2)")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + @")
```

## Design notes

- **Grammar**: three levels (`expression` → `term` → `factor`) gives correct precedence (`*`/`/` bind tighter than `+`/`-`) and left associativity, and unary minus is handled inside `factor`, so it binds tighter than binary operators (`-3 * 2 == -6`).
- **Floating point**: the tokenizer captures digit/dot runs, and `float()` validation rejects malformed numbers like `1.2.3` with a descriptive error.
- **Errors**: empty input is rejected before tokenizing; mismatched parens are reported as "missing closing ')'" or "unexpected ')'"; division by zero is checked in `_parse_term`; invalid characters and stray tokens raise `ValueError` with the offending character and position.

Run the tests with:

```bash
python -m pytest test_expression_evaluator.py -v
```