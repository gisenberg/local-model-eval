Here's a complete implementation of a mathematical expression evaluator using recursive descent parsing:

```python
from typing import Union, Optional, List, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser for mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self) -> None:
        """Initialize the evaluator with an empty expression and position."""
        self.expr: str = ""
        self.pos: int = 0
        self.length: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result.

        Args:
            expr: The mathematical expression to evaluate

        Returns:
            float: The result of the evaluation

        Raises:
            ValueError: If the expression is invalid or contains errors
        """
        # Reset state for each evaluation
        self.expr = expr
        self.pos = 0
        self.length = len(expr)

        # Check for empty expression
        if not expr or not expr.strip():
            raise ValueError("Expression is empty")

        # Parse the expression
        result = self._parse_expression()

        # Check for trailing tokens
        self._skip_whitespace()
        if self.pos < self.length:
            raise ValueError(f"Unexpected token at position {self.pos}: '{self.expr[self.pos]}'")

        return result

    def _parse_expression(self) -> float:
        """
        Parse an expression: term (('+' | '-') term)*

        Returns:
            float: The value of the expression
        """
        # Parse the first term
        left = self._parse_term()

        # Process addition and subtraction
        while True:
            self._skip_whitespace()
            if self.pos >= self.length:
                break

            if self.expr[self.pos] == '+':
                self.pos += 1
                right = self._parse_term()
                left = left + right
            elif self.expr[self.pos] == '-':
                self.pos += 1
                right = self._parse_term()
                left = left - right
            else:
                break

        return left

    def _parse_term(self) -> float:
        """
        Parse a term: factor (('*' | '/') factor)*

        Returns:
            float: The value of the term
        """
        # Parse the first factor
        left = self._parse_factor()

        # Process multiplication and division
        while True:
            self._skip_whitespace()
            if self.pos >= self.length:
                break

            if self.expr[self.pos] == '*':
                self.pos += 1
                right = self._parse_factor()
                left = left * right
            elif self.expr[self.pos] == '/':
                self.pos += 1
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
            else:
                break

        return left

    def _parse_factor(self) -> float:
        """
        Parse a factor: unary minus or primary expression.

        Returns:
            float: The value of the factor
        """
        self._skip_whitespace()

        # Handle unary minus
        if self.pos < self.length and self.expr[self.pos] == '-':
            self.pos += 1
            return -self._parse_factor()

        # Parse primary expression
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parse a primary expression: number or parenthesized expression.

        Returns:
            float: The value of the primary expression
        """
        self._skip_whitespace()

        if self.pos >= self.length:
            raise ValueError("Unexpected end of expression")

        # Handle parentheses
        if self.expr[self.pos] == '(':
            self.pos += 1
            result = self._parse_expression()
            self._skip_whitespace()

            # Expect closing parenthesis
            if self.pos >= self.length or self.expr[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing parenthesis")
            self.pos += 1
            return result

        # Parse a number
        return self._parse_number()

    def _parse_number(self) -> float:
        """
        Parse a floating point number.

        Returns:
            float: The parsed number
        """
        self._skip_whitespace()

        if self.pos >= self.length:
            raise ValueError("Unexpected end of expression")

        # Check if current character is a digit or decimal point
        if not (self.expr[self.pos].isdigit() or self.expr[self.pos] == '.'):
            raise ValueError(f"Invalid token at position {self.pos}: '{self.expr[self.pos]}'")

        # Parse integer part
        number_str = ""
        has_decimal = False

        # Parse digits before decimal point
        while self.pos < self.length and self.expr[self.pos].isdigit():
            number_str += self.expr[self.pos]
            self.pos += 1

        # Parse decimal part
        if self.pos < self.length and self.expr[self.pos] == '.':
            has_decimal = True
            number_str += self.expr[self.pos]
            self.pos += 1

            # Parse digits after decimal point
            while self.pos < self.length and self.expr[self.pos].isdigit():
                number_str += self.expr[self.pos]
                self.pos += 1

        # Check if we have a valid number
        if not number_str or (has_decimal and number_str == '.'):
            raise ValueError(f"Invalid number at position {self.pos}")

        # Convert to float
        try:
            return float(number_str)
        except ValueError:
            raise ValueError(f"Invalid number format: '{number_str}'")

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters in the expression."""
        while self.pos < self.length and self.expr[self.pos].isspace():
            self.pos += 1
```

Now, here are the pytest tests:

```python
import pytest
from expression_evaluator import ExpressionEvaluator

@pytest.fixture
def evaluator():
    """Fixture to create an ExpressionEvaluator instance."""
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    """Test basic arithmetic operations."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("8 / 2") == 4.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + (3 * 4)
    assert evaluator.evaluate("10 - 2 * 3") == 4.0  # 10 - (2 * 3)

def test_precedence(evaluator):
    """Test operator precedence rules."""
    assert evaluator.evaluate("2 + 3 * 4 - 5") == 9.0  # 2 + 12 - 5
    assert evaluator.evaluate("10 / 2 * 3") == 15.0  # (10 / 2) * 3
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0  # 6 + 20
    assert evaluator.evaluate("20 / 4 / 2") == 2.5  # (20 / 4) / 2
    assert evaluator.evaluate("2 + 3 * 4 / 2") == 8.0  # 2 + (12 / 2)

def test_parentheses(evaluator):
    """Test parentheses for grouping."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("(10 - 4) / 2") == 3.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("(2 + 3) * (4 - 1)") == 15.0
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0
    assert evaluator.evaluate("(2 + (3 * 4))") == 14.0

def test_unary_minus(evaluator):
    """Test unary minus operator."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-2 + 3") == 1.0
    assert evaluator.evaluate("5 - -2") == 7.0
    assert evaluator.evaluate("-(3 * 2)") == -6.0
    assert evaluator.evaluate("--5") == 5.0  # Double negation

def test_error_cases(evaluator):
    """Test error handling for invalid expressions."""

    # Test mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("((2 + 3)")

    # Test division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("(2 + 3) / (1 - 1)")

    # Test invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + abc")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + 3 & 4")

    # Test empty expression
    with pytest.raises(ValueError, match="Expression is empty"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Expression is empty"):
        evaluator.evaluate("   ")

    # Test unexpected tokens
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3 4")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + + 3")
```

The implementation includes:

1. **Recursive descent parser** with separate methods for each grammar level:
   - `_parse_expression()`: Handles addition and subtraction
   - `_parse_term()`: Handles multiplication and division
   - `_parse_factor()`: Handles unary minus
   - `_parse_primary()`: Handles numbers and parentheses
   - `_parse_number()`: Parses floating point numbers

2. **Complete error handling**:
   - Mismatched parentheses detection
   - Division by zero checks
   - Invalid token detection
   - Empty expression validation
   - Unexpected trailing tokens

3. **Type hints** throughout the code

4. **Comprehensive test coverage** with 5 test functions covering all required scenarios

The parser correctly handles:
- Operator precedence (multiplication/division before addition/subtraction)
- Parentheses for grouping
- Unary minus (including double negation)
- Floating point numbers
- Whitespace in expressions