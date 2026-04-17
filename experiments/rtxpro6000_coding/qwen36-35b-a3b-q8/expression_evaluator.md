Here's a complete, production-ready implementation following all your requirements. It uses a classic recursive descent parser with proper operator precedence, handles all specified edge cases, and includes comprehensive type hints and docstrings.

### `evaluator.py`
```python
from typing import Optional

class ExpressionEvaluator:
    """A mathematical expression evaluator using a recursive descent parser."""

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result as a float."""
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        # Pre-check for mismatched parentheses
        depth = 0
        for ch in expr:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth < 0:
                raise ValueError("Mismatched parentheses: unexpected closing parenthesis")
        if depth != 0:
            raise ValueError("Mismatched parentheses: unclosed opening parenthesis")

        self._expr = expr
        self._pos = 0
        self._length = len(expr)

        result = self._parse_expression()
        self._skip_whitespace()

        if self._pos != self._length:
            raise ValueError(f"Invalid token at position {self._pos}: '{self._expr[self._pos]}'")

        return result

    def _parse_expression(self) -> float:
        """Parse addition and subtraction operations (lowest precedence)."""
        result = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division operations (higher precedence)."""
        result = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse numbers, parentheses, and unary operators (highest precedence)."""
        if self._peek() in ('+', '-'):
            op = self._consume()
            val = self._parse_factor()
            return -val if op == '-' else val

        if self._peek() == '(':
            self._consume()  # consume '('
            val = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: expected closing parenthesis")
            self._consume()  # consume ')'
            return val

        return self._parse_number()

    def _parse_number(self) -> float:
        """Parse a numeric literal (integer or float)."""
        start = self._pos
        has_dot = False
        while self._pos < self._length and (self._expr[self._pos].isdigit() or (self._expr[self._pos] == '.' and not has_dot)):
            if self._expr[self._pos] == '.':
                has_dot = True
            self._pos += 1

        if start == self._pos:
            raise ValueError(f"Invalid token at position {self._pos}")

        try:
            return float(self._expr[start:self._pos])
        except ValueError:
            raise ValueError(f"Invalid number format at position {start}")

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters in the expression."""
        while self._pos < self._length and self._expr[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> Optional[str]:
        """Return the current character without advancing the position."""
        self._skip_whitespace()
        if self._pos < self._length:
            return self._expr[self._pos]
        return None

    def _consume(self) -> str:
        """Return the current character and advance the position."""
        char = self._peek()
        if char is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
        return char
```

### `test_evaluator.py`
```python
import pytest
from evaluator import ExpressionEvaluator

evaluator = ExpressionEvaluator()

def test_basic_arithmetic():
    """Test basic +, -, *, / operations."""
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 5") == 15.0
    assert evaluator.evaluate("10 / 2") == 5.0
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)

def test_operator_precedence():
    """Test that * and / bind tighter than + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3 + 4 / 2") == 8.0
    assert evaluator.evaluate("1 + 2 * 3 - 4 / 2") == 4.0

def test_parentheses():
    """Test grouping with parentheses."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 - 1))") == 6.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    """Test unary minus support."""
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-3 + -(2 + 1)") == -6.0
    assert evaluator.evaluate("-(10 - 5)") == -5.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_error_cases():
    """Test ValueError raising for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + @")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1.2.3")
```

### Key Design Notes:
1. **Recursive Descent Grammar**: Follows the standard arithmetic grammar:
   `expression → term (('+' | '-') term)*`
   `term → factor (('*' | '/') factor)*`
   `factor → ('+' | '-') factor | '(' expression ')' | NUMBER`
2. **Precedence & Associativity**: Left-associative operators are handled naturally by the `while` loops in `_parse_expression` and `_parse_term`.
3. **Error Handling**: 
   - Parentheses are pre-scanned for depth mismatches.
   - Division by zero is explicitly checked.
   - Invalid tokens/numbers are caught during parsing with precise position reporting.
   - Empty/whitespace-only strings are rejected upfront.
4. **Type Hints & Docstrings**: Every method includes `typing.Optional` where appropriate and a concise docstring explaining its role in the parser.
5. **No `eval()`/`ast`**: Pure recursive descent parsing with manual token consumption.

Run tests with: `pytest test_evaluator.py -v`