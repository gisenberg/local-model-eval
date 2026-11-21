Here's a complete solution. The evaluator uses a classic recursive descent parser with one method per grammar rule, giving correct precedence and grouping for free.

## `expression_evaluator.py`

```python
"""Recursive descent evaluator for arithmetic expressions (no eval/exec)."""

from __future__ import annotations


class ExpressionEvaluator:
    """Evaluate arithmetic expressions safely.

    Supports +, -, *, /, parentheses, unary minus, and decimal numbers.
    """

    def __init__(self) -> None:
        """Create a new evaluator. State is reset on each evaluate() call."""
        self._src: str = ""
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate ``expr`` and return the result as a float.

        Raises:
            TypeError: If ``expr`` is not a string.
            ValueError: If ``expr`` is empty, contains invalid tokens, has
                mismatched parentheses, or attempts division by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"Expression must be a string, got {type(expr).__name__!r}")
        if not expr.strip():
            raise ValueError("Expression is empty")

        self._src = expr
        self._pos = 0

        result = self._parse_expression()
        self._skip_whitespace()
        if self._pos < len(self._src):
            ch = self._peek()
            if ch == ")":
                raise ValueError(f"Unmatched closing parenthesis at position {self._pos}")
            raise ValueError(f"Unexpected token {ch!r} at position {self._pos}")
        return float(result)

    # -- token helpers ------------------------------------------------------

    def _peek(self) -> str:
        """Return the current character ('' at end of input)."""
        if self._pos >= len(self._src):
            return ""
        return self._src[self._pos]

    def _skip_whitespace(self) -> None:
        """Advance past any whitespace characters."""
        while self._pos < len(self._src) and self._src[self._pos].isspace():
            self._pos += 1

    # -- recursive descent rules (lowest to highest precedence) --------------

    def _parse_expression(self) -> float:
        """Parse additive expressions: term (('+' | '-') term)*."""
        self._skip_whitespace()
        value = self._parse_term()
        while True:
            self._skip_whitespace()
            ch = self._peek()
            if ch == "+":
                self._pos += 1
                value += self._parse_term()
            elif ch == "-":
                self._pos += 1
                value -= self._parse_term()
            else:
                return value

    def _parse_term(self) -> float:
        """Parse multiplicative expressions: factor (('*' | '/') factor)*."""
        self._skip_whitespace()
        value = self._parse_factor()
        while True:
            self._skip_whitespace()
            ch = self._peek()
            if ch == "*":
                self._pos += 1
                value *= self._parse_factor()
            elif ch == "/":
                div_pos = self._pos
                self._pos += 1
                divisor = self._parse_factor()
                if divisor == 0.0:
                    raise ValueError(f"Division by zero at position {div_pos}")
                value /= divisor
            else:
                return value

    def _parse_factor(self) -> float:
        """Parse a factor: unary '+'/'-' applied to a primary."""
        self._skip_whitespace()
        ch = self._peek()
        if ch == "-":
            self._pos += 1
            return -self._parse_factor()
        if ch == "+":
            self._pos += 1
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a primary: a number or a parenthesized expression."""
        self._skip_whitespace()
        ch = self._peek()
        if ch == "(":
            self._pos += 1
            value = self._parse_expression()
            self._skip_whitespace()
            if self._peek() != ")":
                raise ValueError(f"Missing closing parenthesis at position {self._pos}")
            self._pos += 1
            return value
        if ch == ")":
            raise ValueError(f"Unmatched closing parenthesis at position {self._pos}")
        if ("0" <= ch <= "9") or ch == ".":
            return self._parse_number()
        if ch == "":
            raise ValueError("Unexpected end of expression")
        raise ValueError(f"Invalid token {ch!r} at position {self._pos}")

    def _parse_number(self) -> float:
        """Parse a decimal literal such as '42', '3.14', '.5', or '5.'."""
        start = self._pos
        while self._pos < len(self._src) and "0" <= self._src[self._pos] <= "9":
            self._pos += 1
        if self._peek() == ".":
            self._pos += 1
            while self._pos < len(self._src) and "0" <= self._src[self._pos] <= "9":
                self._pos += 1
        text = self._src[start:self._pos]
        if not any("0" <= c <= "9" for c in text):
            raise ValueError(f"Invalid number at position {start}")
        return float(text)
```

## `test_expression_evaluator.py`

```python
"""Tests for ExpressionEvaluator. Run with: python -m pytest -v"""

import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3") == pytest.approx(5.0)
    assert evaluator.evaluate("10 - 4") == pytest.approx(6.0)
    assert evaluator.evaluate("3 * 7") == pytest.approx(21.0)
    assert evaluator.evaluate("8 / 2") == pytest.approx(4.0)
    assert evaluator.evaluate("3.14") == pytest.approx(3.14)


def test_precedence(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3 * 4") == pytest.approx(14.0)
    assert evaluator.evaluate("10 - 6 / 3") == pytest.approx(8.0)
    assert evaluator.evaluate("2 * 3 + 4 * 5") == pytest.approx(26.0)
    assert evaluator.evaluate("100 / 10 / 2") == pytest.approx(5.0)  # left associative


def test_parentheses(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("(2 + 3) * 4") == pytest.approx(20.0)
    assert evaluator.evaluate("2 * (3 + 4)") == pytest.approx(14.0)
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == pytest.approx(21.0)


def test_unary_minus(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("-3") == pytest.approx(-3.0)
    assert evaluator.evaluate("-(2 + 1)") == pytest.approx(-3.0)
    assert evaluator.evaluate("2 + -3") == pytest.approx(-1.0)
    assert evaluator.evaluate("-2 * 3") == pytest.approx(-6.0)
    assert evaluator.evaluate("--5") == pytest.approx(5.0)


def test_error_cases(evaluator: ExpressionEvaluator) -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Missing closing parenthesis"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Unmatched closing parenthesis"):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + &")
```

## How to run

```bash
python -m pytest test_expression_evaluator.py -v
```

## Design notes

- **Grammar** (one method per rule, lowest → highest precedence):
  `expression → term (('+'|'-') term)*` → `term → factor (('*'|'/') factor)*` → `factor → ('+'|'-')* primary` → `primary → number | '(' expression ')'`
- Precedence and left associativity fall out naturally from the rule ordering and the loops in `_parse_expression`/`_parse_term`.
- Unary minus is handled in `_parse_factor`, so `-2 * 3` evaluates as `(-2) * 3` and double negation works.
- Numbers support `42`, `3.14`, `.5`, and `5.`; anything else like `.` or `1.2.3` raises `ValueError`.
- All errors are `ValueError` (with position info) except a non-string argument, which raises `TypeError`. Division by zero is detected at the point of the `/` operator and reported with its position.
- Nothing else is supported (no functions, variables, or exponentiation) — the parser rejects anything outside the grammar rather than guessing.