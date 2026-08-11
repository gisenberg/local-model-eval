```python
# evaluator.py
from __future__ import annotations

class ExpressionEvaluator:
    """Recursive descent evaluator for + - * / with parentheses and unary minus."""

    def evaluate(self, expr: str) -> float:
        """Public entry point. Returns the value of expr."""
        s = expr.strip()
        if not s:
            raise ValueError("Empty expression")

        self._s = s
        self._pos = 0
        self._len = len(s)

        value = self._parse_expression()
        self._skip_whitespace()
        if self._pos != self._len:
            raise ValueError(f"Invalid token at position {self._pos}")

        return value

    # ---- helpers ----
    def _skip_whitespace(self) -> None:
        while self._pos < self._len and self._s[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> str | None:
        self._skip_whitespace()
        if self._pos < self._len:
            return self._s[self._pos]
        return None

    # ---- grammar ----
    # expression := term (( '+' | '-' ) term)*
    def _parse_expression(self) -> float:
        left = self._parse_term()
        while True:
            op = self._peek()
            if op == '+' or op == '-':
                self._pos += 1
                right = self._parse_term()
                left = left + right if op == '+' else left - right
            else:
                break
        return left

    # term := unary (( '*' | '/' ) unary)*
    def _parse_term(self) -> float:
        left = self._parse_unary()
        while True:
            op = self._peek()
            if op == '*' or op == '/':
                self._pos += 1
                right = self._parse_unary()
                if op == '*':
                    left = left * right
                else:
                    if right == 0:
                        raise ValueError("Division by zero")
                    left = left / right
            else:
                break
        return left

    # unary := ('-' | '+') unary | primary
    def _parse_unary(self) -> float:
        op = self._peek()
        if op == '-':
            self._pos += 1
            return -self._parse_unary()
        if op == '+':
            self._pos += 1
            return self._parse_unary()
        return self._parse_primary()

    # primary := number | '(' expression ')'
    def _parse_primary(self) -> float:
        ch = self._peek()
        if ch == '(':
            self._pos += 1
            val = self._parse_expression()
            self._skip_whitespace()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._pos += 1
            return val

        if ch is None:
            raise ValueError("Invalid token")

        if ch.isdigit() or ch == '.':
            return self._parse_number()

        raise ValueError(f"Invalid token at position {self._pos}: '{ch}'")

    def _parse_number(self) -> float:
        start = self._pos
        while self._pos < self._len and (self._s[self._pos].isdigit() or self._s[self._pos] == '.'):
            self._pos += 1
        num_str = self._s[start:self._pos]
        if not num_str or num_str == '.':
            raise ValueError(f"Invalid number: {num_str}")
        try:
            return float(num_str)
        except ValueError:
            raise ValueError(f"Invalid number: {num_str}")
```

```python
# test_evaluator.py
import pytest
from evaluator import ExpressionEvaluator

def test_basic_arithmetic():
    ev = ExpressionEvaluator()
    assert ev.evaluate("2+3") == 5
    assert ev.evaluate("10-4") == 6
    assert ev.evaluate("3*4") == 12
    assert ev.evaluate("8/2") == 4.0

def test_precedence():
    ev = ExpressionEvaluator()
    assert ev.evaluate("2+3*4") == 14
    assert ev.evaluate("2*3+4") == 10
    assert ev.evaluate("10-2*3") == 4

def test_parentheses():
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2+3)*4") == 20
    assert ev.evaluate("((1+2)*3)") == 9
    assert ev.evaluate("3 + (4 * 2)") == 11

def test_unary_minus():
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3
    assert ev.evaluate("-(2+1)") == -3
    assert ev.evaluate("5 + -3") == 2
    assert ev.evaluate("--4") == 4

def test_error_cases():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1/0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(1+2")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 & 3")
```

Run tests with:

```bash
pytest test_evaluator.py -q
```

The evaluator uses a classic recursive descent parser, respects precedence, handles unary minus and floating point numbers, and raises descriptive `ValueError`s for empty input, mismatched parentheses, division by zero and invalid tokens without using `eval`.