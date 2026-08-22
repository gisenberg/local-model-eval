

# Mathematical Expression Evaluator

## Implementation

```python
"""A mathematical expression evaluator using a recursive descent parser.

Grammar (highest to lowest precedence):
    expression → term (('+' | '-') term)*
    term       → factor (('*' | '/') factor)*
    factor     → ('+' | '-') factor | primary
    primary    → NUMBER | '(' expression ')'
"""

from __future__ import annotations


class ExpressionEvaluator:
    """Evaluates mathematical expressions with +, -, *, /, parentheses,
    unary minus, and floating-point numbers."""

    def __init__(self) -> None:
        """Initialize the evaluator with no persistent state."""
        self._expr: str = ""
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression and return the numeric result.

        Args:
            expr: The expression string (e.g., "3.14 * (2 + 1)").

        Returns:
            The computed result as a float.

        Raises:
            ValueError: If the expression is empty, contains mismatched
                parentheses, division by zero, or invalid tokens.
        """
        self._expr = expr
        self._pos = 0

        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        result: float = self._parse_expression()

        self._skip_whitespace()

        if self._pos < len(self._expr):
            ch = self._expr[self._pos]
            if ch == ')':
                raise ValueError(
                    f"Mismatched parentheses: unexpected ')' at position {self._pos}"
                )
            raise ValueError(
                f"Invalid token '{ch}' at position {self._pos}"
            )

        return result

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence).

        expression → term (('+' | '-') term)*
        """
        result: float = self._parse_term()

        while True:
            self._skip_whitespace()
            if self._pos >= len(self._expr):
                break
            ch = self._expr[self._pos]
            if ch == '+':
                self._pos += 1
                result += self._parse_term()
            elif ch == '-':
                self._pos += 1
                result -= self._parse_term()
            else:
                break

        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence).

        term → factor (('*' | '/') factor)*
        """
        result: float = self._parse_factor()

        while True:
            self._skip_whitespace()
            if self._pos >= len(self._expr):
                break
            ch = self._expr[self._pos]
            if ch == '*':
                self._pos += 1
                result *= self._parse_factor()
            elif ch == '/':
                self._pos += 1
                divisor: float = self._parse_factor()
                if divisor == 0.0:
                    raise ValueError("Division by zero")
                result /= divisor
            else:
                break

        return result

    def _parse_factor(self) -> float:
        """Parse unary plus/minus and delegate to primary.

        factor → ('+' | '-') factor | primary
        """
        self._skip_whitespace()

        if self._pos >= len(self._expr):
            raise ValueError("Unexpected end of expression")

        ch = self._expr[self._pos]

        if ch == '+':
            self._pos += 1
            return self._parse_factor()
        elif ch == '-':
            self._pos += 1
            return -self._parse_factor()
        else:
            return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a numeric literal or a parenthesized sub-expression.

        primary → NUMBER | '(' expression ')'
        """
        self._skip_whitespace()

        if self._pos >= len(self._expr):
            raise ValueError("Unexpected end of expression")

        ch = self._expr[self._pos]

        if ch == '(':
            self._pos += 1
            result: float = self._parse_expression()
            self._skip_whitespace()

            if self._pos >= len(self._expr):
                raise ValueError(
                    "Mismatched parentheses: missing closing ')'"
                )
            if self._expr[self._pos] != ')':
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found "
                    f"'{self._expr[self._pos]}' at position {self._pos}"
                )
            self._pos += 1
            return result

        if ch == ')':
            raise ValueError(
                f"Mismatched parentheses: unexpected ')' at position {self._pos}"
            )

        if ch.isdigit() or ch == '.':
            return self._parse_number()

        raise ValueError(f"Invalid token '{ch}' at position {self._pos}")

    def _parse_number(self) -> float:
        """Parse a numeric literal (integer or floating-point).

        Recognises sequences of digits with at most one decimal point,
        e.g. '42', '3.14', '.5'.
        """
        start: int = self._pos
        has_dot: bool = False

        while self._pos < len(self._expr):
            ch = self._expr[self._pos]
            if ch.isdigit():
                self._pos += 1
            elif ch == '.' and not has_dot:
                has_dot = True
                self._pos += 1
            else:
                break

        num_str: str = self._expr[start : self._pos]

        if num_str in ("", "."):
            raise ValueError(
                f"Invalid number at position {start}: expected a digit"
            )

        return float(num_str)

    def _skip_whitespace(self) -> None:
        """Advance the position past any whitespace characters."""
        while self._pos < len(self._expr) and self._expr[self._pos].isspace():
            self._pos += 1
```

## Tests

```python
"""Pytest tests for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


class TestExpressionEvaluator:
    """Comprehensive tests for the expression evaluator."""

    def test_basic_arithmetic(self) -> None:
        """Verify the four basic operators and floating-point literals."""
        ev = ExpressionEvaluator()
        assert ev.evaluate("2 + 3") == 5.0
        assert ev.evaluate("10 - 4") == 6.0
        assert ev.evaluate("6 * 7") == 42.0
        assert ev.evaluate("20 / 4") == 5.0
        assert ev.evaluate("3.14 + 1.86") == 5.0
        assert ev.evaluate("0.1 + 0.2") == pytest.approx(0.3)
        # Whitespace tolerance
        assert ev.evaluate("  3   +   4  ") == 7.0

    def test_precedence(self) -> None:
        """Verify that * and / bind tighter than + and -."""
        ev = ExpressionEvaluator()
        assert ev.evaluate("2 + 3 * 4") == 14.0
        assert ev.evaluate("10 - 2 * 3") == 4.0
        assert ev.evaluate("2 * 3 + 4 * 5") == 26.0
        assert ev.evaluate("10 / 2 + 3") == 8.0
        assert ev.evaluate("2 + 10 / 2") == 7.0
        assert ev.evaluate("100 / 10 / 2") == 5.0  # left-associative

    def test_parentheses(self) -> None:
        """Verify that parentheses override default precedence."""
        ev = ExpressionEvaluator()
        assert ev.evaluate("(2 + 3) * 4") == 20.0
        assert ev.evaluate("2 * (3 + 4)") == 14.0
        assert ev.evaluate("((2 + 3))") == 5.0
        assert ev.evaluate("(1 + 2) * (3 + 4)") == 21.0
        assert ev.evaluate("100 / (5 + 5)") == 10.0
        assert ev.evaluate("(1) + (2) + (3)") == 6.0

    def test_unary_minus(self) -> None:
        """Verify unary minus at various positions."""
        ev = ExpressionEvaluator()
        assert ev.evaluate("-3") == -3.0
        assert ev.evaluate("-(2 + 1)") == -3.0
        assert ev.evaluate("-3 + 5") == 2.0
        assert ev.evaluate("2 * -3") == -6.0
        assert ev.evaluate("10 / -2") == -5.0
        assert ev.evaluate("-(-5)") == 5.0
        assert ev.evaluate("-(2 * 3)") == -6.0
        assert ev.evaluate("3 - -2") == 5.0

    def test_error_cases(self) -> None:
        """Verify that invalid expressions raise ValueError with clear messages."""
        ev = ExpressionEvaluator()

        # Empty / whitespace-only
        with pytest.raises(ValueError, match="Empty expression"):
            ev.evaluate("")
        with pytest.raises(ValueError, match="Empty expression"):
            ev.evaluate("   ")

        # Mismatched parentheses
        with pytest.raises(ValueError, match="[Mm]ismatched parentheses"):
            ev.evaluate("(1 + 2")
        with pytest.raises(ValueError, match="[Mm]ismatched parentheses"):
            ev.evaluate("1 + 2)")
        with pytest.raises(ValueError, match="[Mm]ismatched parentheses"):
            ev.evaluate("((1 + 2)")
        with pytest.raises(ValueError, match="[Mm]ismatched parentheses"):
            ev.evaluate(")")

        # Division by zero
        with pytest.raises(ValueError, match="[Dd]ivision by zero"):
            ev.evaluate("1 / 0")
        with pytest.raises(ValueError, match="[Dd]ivision by zero"):
            ev.evaluate("5 / (2 - 2)")

        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid token"):
            ev.evaluate("1 + 2 @ 3")
        with pytest.raises(ValueError, match="Invalid token"):
            ev.evaluate("abc")
        with pytest.raises(ValueError, match="[Ii]nvalid number"):
            ev.evaluate("1.2.3")
```

## Running the Tests

```bash
pip install pytest
pytest test_expression_evaluator.py -v
```

## How the Parser Works

```
evaluate("2 + 3 * -4")
│
└─ _parse_expression
   │
   ├─ _parse_term
   │  └─ _parse_factor
   │     └─ _parse_primary  ──►  2.0
   │
   ├─ sees '+'
   │
   └─ _parse_term
      │
      ├─ _parse_factor
      │  └─ _parse_primary  ──►  3.0
      │
      ├─ sees '*'
      │
      └─ _parse_factor
         │  sees '-'  (unary)
         └─ _parse_factor
            └─ _parse_primary  ──►  4.0  →  negated: -4.0

   result = 2.0 + (3.0 * -4.0) = 2.0 + (-12.0) = -10.0
```

Each grammar level calls the next-lower-precedence level, so `*`/`/` are consumed before the `+`/`-` loop in `_parse_expression` can see them — giving the correct precedence without any explicit priority table. Unary operators are handled by `_parse_factor` recursing into itself, which naturally supports chains like `--5`.