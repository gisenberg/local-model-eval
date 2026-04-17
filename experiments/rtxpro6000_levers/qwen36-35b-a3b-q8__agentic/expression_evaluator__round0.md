Here's a complete, production-ready implementation meeting all your requirements:

```python
from typing import Optional

class ExpressionEvaluator:
    """A recursive descent parser for evaluating mathematical expressions."""

    def __init__(self) -> None:
        self.expr: str = ""
        self.pos: int = 0
        self.length: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result as a float."""
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        self.expr = expr
        self.pos = 0
        self.length = len(expr)
        result: float = self._parse_expr()
        self._skip_whitespace()
        if self.pos < self.length:
            raise ValueError(f"Unexpected character at position {self.pos}: '{self.expr[self.pos]}'")
        return result

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters in the expression."""
        while self.pos < self.length and self.expr[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> Optional[str]:
        """Return the current non-whitespace character, or None if at the end."""
        self._skip_whitespace()
        if self.pos < self.length:
            return self.expr[self.pos]
        return None

    def _advance(self) -> str:
        """Return the current character and advance the position."""
        if self.pos < self.length:
            ch: str = self.expr[self.pos]
            self.pos += 1
            return ch
        return ""

    def _parse_expr(self) -> float:
        """Parse addition and subtraction expressions (lowest precedence)."""
        result: float = self._parse_term()
        while self._peek() in ('+', '-'):
            op: str = self._advance()
            right: float = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division expressions (higher precedence)."""
        result: float = self._parse_factor()
        while self._peek() in ('*', '/'):
            op: str = self._advance()
            right: float = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse factors: numbers, parenthesized expressions, and unary operators."""
        token: Optional[str] = self._peek()
        if token == '(':
            self._advance()
            result: float = self._parse_expr()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        elif token in ('+', '-'):
            self._advance()
            return (1 if token == '+' else -1) * self._parse_factor()
        elif token is not None and (token.isdigit() or token == '.'):
            return self._parse_number()
        else:
            if token is None:
                raise ValueError("Unexpected end of expression")
            raise ValueError(f"Invalid token: '{token}'")

    def _parse_number(self) -> float:
        """Parse a numeric literal (integer or float)."""
        start: int = self.pos
        while self.pos < self.length and (self.expr[self.pos].isdigit() or self.expr[self.pos] == '.'):
            self.pos += 1
        if start == self.pos:
            raise ValueError("Expected number")
        num_str: str = self.expr[start:self.pos]
        try:
            return float(num_str)
        except ValueError:
            raise ValueError(f"Invalid number format: '{num_str}'")
```

### Pytest Tests

```python
import pytest
from expression_evaluator import ExpressionEvaluator  # Adjust import path as needed

evaluator = ExpressionEvaluator()

def test_basic_arithmetic() -> None:
    """Test basic arithmetic operations."""
    assert evaluator.evaluate("2 + 3") == pytest.approx(5.0)
    assert evaluator.evaluate("10 - 4") == pytest.approx(6.0)
    assert evaluator.evaluate("3 * 4") == pytest.approx(12.0)
    assert evaluator.evaluate("15 / 3") == pytest.approx(5.0)

def test_operator_precedence() -> None:
    """Test correct operator precedence (*, / before +, -)."""
    assert evaluator.evaluate("2 + 3 * 4") == pytest.approx(14.0)
    assert evaluator.evaluate("10 - 2 * 3 + 1") == pytest.approx(5.0)
    assert evaluator.evaluate("2 * 3 + 4 * 5") == pytest.approx(26.0)

def test_parentheses() -> None:
    """Test grouping with parentheses."""
    assert evaluator.evaluate("(2 + 3) * 4") == pytest.approx(20.0)
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == pytest.approx(15.0)
    assert evaluator.evaluate("2 * (3 + (4 * 2))") == pytest.approx(22.0)

def test_unary_minus() -> None:
    """Test unary minus and plus operators."""
    assert evaluator.evaluate("-5") == pytest.approx(-5.0)
    assert evaluator.evaluate("-(2 + 3)") == pytest.approx(-5.0)
    assert evaluator.evaluate("--3") == pytest.approx(3.0)
    assert evaluator.evaluate("-2 * 3") == pytest.approx(-6.0)
    assert evaluator.evaluate("+4.5") == pytest.approx(4.5)

def test_error_cases() -> None:
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 @ 2")
    with pytest.raises(ValueError, match="Invalid number format"):
        evaluator.evaluate("3.14.15")
```

### Key Design Notes:
1. **Recursive Descent Grammar**: Follows the standard arithmetic grammar:
   `expr → term (('+' | '-') term)*`
   `term → factor (('*' | '/') factor)*`
   `factor → ('+' | '-') factor | '(' expr ')' | NUMBER`
2. **Precedence Handling**: Achieved naturally by the call hierarchy (`_parse_expr` calls `_parse_term`, which calls `_parse_factor`). Higher precedence operators are parsed deeper in the call stack.
3. **State Management**: Uses a single string pointer (`self.pos`) instead of creating token lists, keeping memory usage O(1) auxiliary space.
4. **Error Handling**: All invalid states are caught early with descriptive `ValueError` messages as requested.
5. **Type Safety**: Full type hints using `typing.Optional` and modern Python conventions. All methods include concise docstrings.