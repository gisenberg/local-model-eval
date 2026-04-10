# Qwen3.5-122B-A10B Q4_K_M — KV f16 (baseline) — Expression Evaluator

## Output

Here is a complete implementation of the `ExpressionEvaluator` using a recursive descent parser, followed by the required `pytest` tests.

### Implementation: `expression_evaluator.py`

```python
from typing import List, Union, Optional
import re

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        self.expression: str = ""
        self.pos: int = 0
        self.length: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.expression = expr.replace(" ", "")
        self.pos = 0
        self.length = len(self.expression)

        if self.length == 0:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        if self.pos < self.length:
            raise ValueError(f"Invalid token at position {self.pos}: '{self.expression[self.pos]}'")

        return result

    def _parse_expression(self) -> float:
        """
        Parses an expression handling addition and subtraction (lowest precedence).
        """
        result = self._parse_term()

        while self.pos < self.length:
            char = self.expression[self.pos]
            if char == '+':
                self.pos += 1
                result += self._parse_term()
            elif char == '-':
                self.pos += 1
                result -= self._parse_term()
            else:
                break
        
        return result

    def _parse_term(self) -> float:
        """
        Parses a term handling multiplication and division (higher precedence).
        """
        result = self._parse_factor()

        while self.pos < self.length:
            char = self.expression[self.pos]
            if char == '*':
                self.pos += 1
                result *= self._parse_factor()
            elif char == '/':
                self.pos += 1
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                result /= divisor
            else:
                break
        
        return result

    def _parse_factor(self) -> float:
        """
        Parses a factor handling unary operators and primary values (numbers/parentheses).
        """
        # Handle unary minus or plus
        if self.pos < self.length and self.expression[self.pos] in ('+', '-'):
            sign = 1
            if self.expression[self.pos] == '-':
                sign = -1
            self.pos += 1
            return sign * self._parse_factor()

        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parses a primary value: a number or a parenthesized expression.
        """
        if self.pos >= self.length:
            raise ValueError("Unexpected end of expression")

        char = self.expression[self.pos]

        if char == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= self.length or self.expression[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            
            self.pos += 1
            return result

        if char.isdigit() or char == '.':
            return self._parse_number()

        raise ValueError(f"Invalid token at position {self.pos}: '{char}'")

    def _parse_number(self) -> float:
        """
        Parses a floating point number from the current position.
        """
        start = self.pos
        has_dot = False

        while self.pos < self.length:
            char = self.expression[self.pos]
            if char.isdigit():
                self.pos += 1
            elif char == '.' and not has_dot:
                has_dot = True
                self.pos += 1
            else:
                break

        if start == self.pos:
            raise ValueError(f"Expected number at position {start}")

        number_str = self.expression[start:self.pos]
        
        # Validate format (e.g., prevent multiple dots if logic above failed, though loop handles it)
        if number_str.count('.') > 1:
            raise ValueError(f"Invalid number format: '{number_str}'")

        try:
            return float(number_str)
        except ValueError:
            raise ValueError(f"Invalid number: '{number_str}'")
```

### Test Suite: `test_expression_evaluator.py`

```python
import pytest
from expression_evaluator import ExpressionEvaluator

evaluator = ExpressionEvaluator()

def test_basic_arithmetic():
    """Tests basic addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("8 / 2") == 4.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # Precedence check
    assert evaluator.evaluate("10 / 2 + 3") == 8.0

def test_operator_precedence():
    """Tests that multiplication/division happens before addition/subtraction."""
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    assert evaluator.evaluate("100 / 10 / 2") == 5.0
    assert evaluator.evaluate("2 + 3 * 4 - 5") == 9.0

def test_parentheses():
    """Tests grouping with parentheses to override precedence."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 4) / 5") == 4.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0

def test_unary_minus_and_floats():
    """Tests unary minus operators and floating point numbers."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-3.14") == -3.14
    assert evaluator.evaluate("3.14 + 2.86") == 6.0
    assert evaluator.evaluate("- (2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0
    assert evaluator.evaluate("-2 * -3") == 6.0
    assert evaluator.evaluate("10 / -2") == -5.0

def test_error_cases():
    """Tests various error conditions."""
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (2 - 2)")

    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")
    
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")

    # Trailing invalid characters
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + 3 x")
```

### How to Run

1.  Save the first code block as `expression_evaluator.py`.
2.  Save the second code block as `test_expression_evaluator.py`.
3.  Ensure you have `pytest` installed (`pip install pytest`).
4.  Run the tests from the terminal:
    ```bash
    pytest test_expression_evaluator.py -v
    ```

### Key Design Decisions

1.  **Recursive Descent**: The parser is structured into three levels of precedence:
    *   `_parse_expression`: Handles `+` and `-`.
    *   `_parse_term`: Handles `*` and `/`.
    *   `_parse_factor`: Handles unary operators (`-`, `+`) and primary values (numbers, parentheses).
    This structure naturally enforces operator precedence without needing complex precedence tables.
2.  **Unary Minus**: Handled in `_parse_factor`. It checks if the current character is `-` (or `+`), consumes it, and recursively calls `_parse_factor` to handle cases like `--3` or `-(2+1)`.
3.  **Error Handling**:
    *   **Empty**: Checked at the start.
    *   **Mismatched Parentheses**: Checked when closing `)` is expected but not found, or if the expression ends with an open `(`.
    *   **Division by Zero**: Checked immediately before performing the division operation.
    *   **Invalid Tokens**: Checked in `_parse_primary` if the character is not a digit, dot, or parenthesis.
    *   **Trailing Characters**: Checked after the main parsing loop in `evaluate` to ensure the entire string was consumed.
4.  **No `eval()`**: The solution strictly uses string manipulation and recursion to parse and compute values.
