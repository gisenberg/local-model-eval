Here's a complete implementation of a mathematical expression evaluator using recursive descent parsing:

```python
from typing import Union, Optional, List, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser for mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self) -> None:
        """Initialize the evaluator with an empty token list and position."""
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result.

        Args:
            expr: String containing the mathematical expression

        Returns:
            float: The evaluated result

        Raises:
            ValueError: If the expression is invalid
        """
        # Reset state
        self.tokens = []
        self.pos = 0

        # Tokenize the input
        self._tokenize(expr)

        # Check for empty expression
        if not self.tokens:
            raise ValueError("Empty expression")

        # Parse the expression
        result = self._parse_expression()

        # Check for trailing tokens
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: '{self.tokens[self.pos]}'")

        return result

    def _tokenize(self, expr: str) -> None:
        """
        Convert the expression string into a list of tokens.

        Args:
            expr: String containing the mathematical expression

        Raises:
            ValueError: If invalid tokens are found
        """
        i = 0
        while i < len(expr):
            # Skip whitespace
            if expr[i].isspace():
                i += 1
                continue

            # Handle numbers
            if expr[i].isdigit() or expr[i] == '.':
                start = i
                has_decimal = False

                while i < len(expr):
                    if expr[i].isdigit():
                        i += 1
                    elif expr[i] == '.' and not has_decimal:
                        has_decimal = True
                        i += 1
                    else:
                        break

                # Check if we have a valid number
                if start == i:
                    raise ValueError(f"Invalid number at position {start}")

                self.tokens.append(expr[start:i])
                continue

            # Handle operators and parentheses
            if expr[i] in '+-*/()':
                self.tokens.append(expr[i])
                i += 1
                continue

            # Invalid character
            raise ValueError(f"Invalid character: '{expr[i]}' at position {i}")

    def _parse_expression(self) -> float:
        """
        Parse an expression: term (('+' | '-') term)*

        Returns:
            float: The value of the expression
        """
        result = self._parse_term()

        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()

            if op == '+':
                result += right
            else:
                result -= right

        return result

    def _parse_term(self) -> float:
        """
        Parse a term: factor (('*' | '/') factor)*

        Returns:
            float: The value of the term
        """
        result = self._parse_factor()

        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()

            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right

        return result

    def _parse_factor(self) -> float:
        """
        Parse a factor: unary minus or primary expression.

        Returns:
            float: The value of the factor
        """
        # Handle unary minus
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1
            return -self._parse_factor()

        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parse a primary: number or parenthesized expression.

        Returns:
            float: The value of the primary
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")

        token = self.tokens[self.pos]

        # Handle numbers
        if token.replace('.', '', 1).isdigit():
            self.pos += 1
            return float(token)

        # Handle parentheses
        if token == '(':
            self.pos += 1
            result = self._parse_expression()

            # Expect closing parenthesis
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing ')'")

            self.pos += 1
            return result

        # Handle unexpected tokens
        if token == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")

        raise ValueError(f"Unexpected token: '{token}'")
```

Now, here are the pytest tests:

```python
import pytest
from expression_evaluator import ExpressionEvaluator

@pytest.fixture
def evaluator():
    """Create an ExpressionEvaluator instance for testing."""
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    """Test basic arithmetic operations."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("8 / 2") == 4.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + (3*4) = 14

def test_precedence(evaluator):
    """Test operator precedence rules."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # Multiplication first
    assert evaluator.evaluate("10 - 2 * 3") == 4.0  # Multiplication first
    assert evaluator.evaluate("20 / 4 * 2") == 10.0  # Left to right for same precedence
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0  # Both multiplications first
    assert evaluator.evaluate("10 - 2 + 3") == 11.0  # Left to right for + and -

def test_parentheses(evaluator):
    """Test parentheses for grouping."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("(10 - 2) * (3 + 1)") == 32.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0  # Nested parentheses
    assert evaluator.evaluate("(2 + 3) * (4 - 1)") == 15.0

def test_unary_minus(evaluator):
    """Test unary minus operator."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-2 + 3") == 1.0
    assert evaluator.evaluate("5 - -2") == 7.0  # 5 - (-2) = 7
    assert evaluator.evaluate("-(3 * 2)") == -6.0
    assert evaluator.evaluate("--5") == 5.0  # Double negation

def test_error_cases(evaluator):
    """Test error handling for invalid expressions."""

    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")

    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + @")

    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3 & 4")

    # Invalid number format
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("2..3")

    # Unexpected end of expression
    with pytest.raises(ValueError, match="Unexpected end"):
        evaluator.evaluate("2 +")

    # Unexpected token
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3 4")
```

The implementation includes:

1. **Tokenization**: Converts the expression string into tokens (numbers, operators, parentheses)
2. **Recursive descent parsing**: Uses grammar:
   - Expression → Term (('+' | '-') Term)*
   - Term → Factor (('*' | '/') Factor)*
   - Factor → '-' Factor | Primary
   - Primary → Number | '(' Expression ')'
3. **Error handling**: Descriptive error messages for all specified error cases
4. **Type hints**: Throughout the implementation
5. **Docstrings**: Brief documentation on each method
6. **Floating point support**: Handles decimal numbers like "3.14"
7. **Unary minus**: Handles negative numbers and negated expressions

The parser correctly handles:
- Operator precedence (multiplication/division before addition/subtraction)
- Parentheses for grouping
- Unary minus at any level
- Floating point numbers
- Left-to-right evaluation for same precedence operators