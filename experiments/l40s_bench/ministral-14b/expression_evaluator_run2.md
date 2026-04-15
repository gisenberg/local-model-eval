# Ministral-3-14B-Instruct-2512-BF16 (L40S vLLM) — Expression Evaluator — Run 2

## Output

# Mathematical Expression Evaluator in Python

Here's a complete implementation of an `ExpressionEvaluator` class using recursive descent parsing:

```python
import re
from typing import Optional, Tuple, Union

class ExpressionEvaluator:
    """A mathematical expression evaluator that supports basic arithmetic operations,
    parentheses, unary minus, and floating point numbers using recursive descent parsing."""

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result.

        Args:
            expr: The expression string to evaluate

        Returns:
            The result of the evaluation as a float

        Raises:
            ValueError: For various error conditions including:
                - Mismatched parentheses
                - Division by zero
                - Invalid tokens
                - Empty expressions
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        # Remove all whitespace from the expression
        expr = expr.replace(' ', '')

        # Validate the expression has balanced parentheses
        if not self._has_balanced_parentheses(expr):
            raise ValueError("Mismatched parentheses")

        # Initialize the parser with the expression
        parser = self._ExpressionParser(expr)
        return parser.parse_expression()

    def _has_balanced_parentheses(self, expr: str) -> bool:
        """Check if the expression has balanced parentheses."""
        balance = 0
        for char in expr:
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
                if balance < 0:
                    return False
        return balance == 0

    class _ExpressionParser:
        """Internal recursive descent parser class."""

        def __init__(self, expr: str):
            self.expr = expr
            self.pos = 0
            self.current_char = self.expr[self.pos] if self.expr else None

        def parse_expression(self) -> float:
            """Parse and evaluate an expression (handles + and - at top level)."""
            result = self.parse_add_sub()

            # If we're at the end, return the result
            if self.pos == len(self.expr):
                return result

            raise ValueError(f"Unexpected token at position {self.pos}: '{self.current_char}'")

        def parse_add_sub(self) -> float:
            """Parse addition and subtraction expressions (left-associative)."""
            result = self.parse_mul_div()

            while self.current_char in ('+', '-'):
                op = self.current_char
                self.advance()

                if op == '+':
                    result += self.parse_mul_div()
                elif op == '-':
                    result -= self.parse_mul_div()

            return result

        def parse_mul_div(self) -> float:
            """Parse multiplication and division expressions (left-associative)."""
            result = self.parse_unary()

            while self.current_char in ('*', '/'):
                op = self.current_char
                self.advance()

                if op == '*':
                    result *= self.parse_unary()
                elif op == '/':
                    divisor = self.parse_unary()
                    if divisor == 0:
                        raise ValueError("Division by zero")
                    result /= divisor

            return result

        def parse_unary(self) -> float:
            """Parse unary minus or a primary expression."""
            if self.current_char == '-':
                self.advance()
                return -self.parse_primary()

            return self.parse_primary()

        def parse_primary(self) -> float:
            """Parse a primary expression (numbers, parentheses, or unary minus)."""
            if self.current_char == '(':
                self.advance()
                result = self.parse_expression()

                if self.current_char != ')':
                    raise ValueError(f"Expected ')' at position {self.pos}")
                self.advance()
                return result

            return self.parse_number()

        def parse_number(self) -> float:
            """Parse a number (integer or floating point)."""
            match = re.match(r'^-?\d+\.?\d*', self.expr[self.pos:])
            if not match:
                raise ValueError(f"Invalid number at position {self.pos}")

            num_str = match.group()
            self.pos += len(num_str)

            try:
                return float(num_str)
            except ValueError:
                raise ValueError(f"Invalid number format: {num_str}")

        def advance(self) -> None:
            """Move to the next character in the expression."""
            self.pos += 1
            self.current_char = self.expr[self.pos] if self.pos < len(self.expr) else None
```

## Test Cases

Here are 5 pytest test cases covering the requirements:

```python
import pytest
from expression_evaluator import ExpressionEvaluator

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2+3*4") == 14
    assert evaluator.evaluate("(2+3)*4") == 20
    assert evaluator.evaluate("10/2-3") == 2

def test_operator_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2+3*4") == 14  # * before +
    assert evaluator.evaluate("2*3+4") == 10  # * before +
    assert evaluator.evaluate("2+3/4") == 2.75  # / before +
    assert evaluator.evaluate("2/3*4") == 2.666...  # left-associative *

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2+3)*4") == 20
    assert evaluator.evaluate("2*(3+4)") == 14
    assert evaluator.evaluate("((2+3)*4)") == 20
    assert evaluator.evaluate("2+3*(4+5)") == 27

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3
    assert evaluator.evaluate("-(2+1)") == -3
    assert evaluator.evaluate("2*-3") == -6
    assert evaluator.evaluate("-(2*-3)") == 6

def test_error_cases():
    evaluator = ExpressionEvaluator()

    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2+3")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1/0")

    # Invalid token
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2+3@4")

    # Invalid number
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("2.3.4")

    # Floating point numbers
    assert evaluator.evaluate("3.14+2.5") == 5.64
    assert evaluator.evaluate("0.5*2.0") == 1.0
```

## Implementation Notes

1. **Recursive Descent Parsing**: The parser uses a top-down approach with methods for each level of precedence:
   - `parse_expression()` - handles top-level + and -
   - `parse_add_sub()` - handles addition and subtraction
   - `parse_mul_div()` - handles multiplication and division
   - `parse_unary()` - handles unary minus
   - `parse_primary()` - handles numbers and parentheses
   - `parse_number()` - handles number parsing

2. **Error Handling**: The evaluator checks for:
   - Empty expressions
   - Mismatched parentheses
   - Division by zero
   - Invalid tokens
   - Invalid number formats

3. **Operator Precedence**: The parser follows standard mathematical precedence:
   - Parentheses have highest precedence
   - Unary minus is next
   - Then multiplication and division (left-associative)
   - Finally addition and subtraction (left-associative)

4. **Floating Point Support**: The number parser uses regex to match both integers and floating point numbers.

5. **Type Hints**: All methods include proper type hints for better code clarity and IDE support.

To use this implementation, save the evaluator code in a file (e.g., `expression_evaluator.py`) and the test cases in a separate file with pytest. The evaluator can then be imported and used as shown in the test cases.
