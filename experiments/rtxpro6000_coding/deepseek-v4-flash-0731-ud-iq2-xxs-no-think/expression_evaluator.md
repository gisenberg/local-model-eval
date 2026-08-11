Here's a complete implementation of a mathematical expression evaluator using recursive descent parsing:

```python
"""
Mathematical Expression Evaluator
A recursive descent parser for arithmetic expressions supporting:
- Basic operations: +, -, *, /
- Parentheses for grouping
- Unary minus
- Floating point numbers
"""

from typing import List, Union, Optional


class ExpressionEvaluator:
    """A recursive descent parser for mathematical expressions."""

    def __init__(self) -> None:
        """Initialize the evaluator with token position tracking."""
        self.tokens: List[Union[str, float]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.

        Args:
            expr: String containing the mathematical expression

        Returns:
            float: The result of evaluating the expression

        Raises:
            ValueError: If expression is empty, contains invalid tokens,
                       has mismatched parentheses, or division by zero
        """
        # Tokenize the expression
        self.tokens = self._tokenize(expr)
        self.pos = 0

        if not self.tokens:
            raise ValueError("Empty expression")

        # Parse the expression
        result = self._parse_expression()

        # Check if all tokens were consumed
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")

        return result

    def _tokenize(self, expr: str) -> List[Union[str, float]]:
        """
        Convert expression string into a list of tokens.

        Args:
            expr: String containing the mathematical expression

        Returns:
            List of tokens (numbers as float, operators and parens as strings)

        Raises:
            ValueError: If expression is empty or contains invalid characters
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokens: List[Union[str, float]] = []
        i = 0
        expr = expr.strip()

        while i < len(expr):
            char = expr[i]

            # Skip whitespace
            if char.isspace():
                i += 1
                continue

            # Handle numbers (including decimals)
            if char.isdigit() or char == '.':
                num_str = ""
                has_decimal = False

                while i < len(expr):
                    char = expr[i]
                    if char.isdigit():
                        num_str += char
                    elif char == '.' and not has_decimal:
                        num_str += char
                        has_decimal = True
                    elif char == '.' and has_decimal:
                        raise ValueError(f"Invalid number format: multiple decimal points")
                    else:
                        break
                    i += 1

                try:
                    tokens.append(float(num_str))
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")
                continue

            # Handle operators and parentheses
            if char in '+-*/()':
                tokens.append(char)
                i += 1
                continue

            # Invalid character
            raise ValueError(f"Invalid character: '{char}'")

        return tokens

    def _parse_expression(self) -> float:
        """
        Parse expression: expression = term (('+' | '-') term)*

        Returns:
            float: Result of expression evaluation
        """
        result = self._parse_term()

        while self._peek() in ('+', '-'):
            operator = self._advance()
            right = self._parse_term()

            if operator == '+':
                result += right
            else:  # '-'
                result -= right

        return result

    def _parse_term(self) -> float:
        """
        Parse term: term = factor (('*' | '/') factor)*

        Returns:
            float: Result of term evaluation
        """
        result = self._parse_factor()

        while self._peek() in ('*', '/'):
            operator = self._advance()
            right = self._parse_factor()

            if operator == '*':
                result *= right
            else:  # '/'
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right

        return result

    def _parse_factor(self) -> float:
        """
        Parse factor: factor = ('-' factor) | '(' expression ')' | number

        Returns:
            float: Result of factor evaluation
        """
        # Handle unary minus
        if self._peek() == '-':
            self._advance()
            return -self._parse_factor()

        # Handle parentheses
        if self._peek() == '(':
            self._advance()  # consume '('
            result = self._parse_expression()

            # Check for matching closing parenthesis
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._advance()  # consume ')'
            return result

        # Handle numbers
        if self._peek() is not None and isinstance(self._peek(), float):
            return self._advance()

        # Handle unexpected tokens
        if self._peek() is None:
            raise ValueError("Unexpected end of expression")
        else:
            raise ValueError(f"Unexpected token: {self._peek()}")

    def _peek(self) -> Optional[Union[str, float]]:
        """
        Return the current token without consuming it.

        Returns:
            Current token or None if at end of token list
        """
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _advance(self) -> Union[str, float]:
        """
        Consume and return the current token.

        Returns:
            Current token

        Raises:
            ValueError: If no more tokens available
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")

        token = self.tokens[self.pos]
        self.pos += 1
        return token


# Test cases
import pytest


class TestExpressionEvaluator:
    """Test cases for ExpressionEvaluator class."""

    def setup_method(self):
        """Set up a fresh evaluator for each test."""
        self.evaluator = ExpressionEvaluator()

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert self.evaluator.evaluate("2 + 3") == 5
        assert self.evaluator.evaluate("7 - 4") == 3
        assert self.evaluator.evaluate("6 * 5") == 30
        assert self.evaluator.evaluate("8 / 2") == 4
        assert self.evaluator.evaluate("2 + 3 * 4") == 14
        assert self.evaluator.evaluate("10 - 2 * 3") == 4

    def test_precedence(self):
        """Test operator precedence rules."""
        assert self.evaluator.evaluate("2 + 3 * 4") == 14
        assert self.evaluator.evaluate("2 * 3 + 4") == 10
        assert self.evaluator.evaluate("10 - 2 * 3") == 4
        assert self.evaluator.evaluate("20 / 4 * 2") == 10
        assert self.evaluator.evaluate("2 + 3 * 4 - 5") == 9

    def test_parentheses(self):
        """Test parentheses for grouping."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20
        assert self.evaluator.evaluate("2 * (3 + 4)") == 14
        assert self.evaluator.evaluate("(2 + 3) * (4 - 1)") == 15
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20
        assert self.evaluator.evaluate("2 * (3 + (4 - 1))") == 12

    def test_unary_minus(self):
        """Test unary minus operations."""
        assert self.evaluator.evaluate("-3") == -3
        assert self.evaluator.evaluate("-(2 + 1)") == -3
        assert self.evaluator.evaluate("-2 + 3") == 1
        assert self.evaluator.evaluate("2 * -3") == -6
        assert self.evaluator.evaluate("-(2 * 3)") == -6
        assert self.evaluator.evaluate("--5") == 5

    def test_error_cases(self):
        """Test error handling for invalid expressions."""
        # Empty expression
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("")
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("   ")

        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("2 + 3)")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("(2 + 3))")

        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("5 / 0")
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("2 / (3 - 3)")

        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 + a")
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 $ 3")
        with pytest.raises(ValueError, match="Unexpected token"):
            self.evaluator.evaluate("2 + 3 4")
        with pytest.raises(ValueError, match="Unexpected token"):
            self.evaluator.evaluate("2 + + 3")

        # Invalid number format
        with pytest.raises(ValueError, match="Invalid number"):
            self.evaluator.evaluate("2.5.5 + 3")
```

This implementation includes:

1. **Tokenization**: Converts the expression string into a list of tokens (numbers as floats, operators and parentheses as strings)
2. **Recursive Descent Parsing**: Uses three levels of parsing:
   - `_parse_expression()`: Handles addition and subtraction
   - `_parse_term()`: Handles multiplication and division
   - `_parse_factor()`: Handles unary minus, parentheses, and numbers
3. **Error Handling**: Raises `ValueError` with descriptive messages for:
   - Empty expressions
   - Mismatched parentheses
   - Division by zero
   - Invalid tokens/characters
   - Invalid number formats
4. **Type Hints**: All methods have complete type hints
5. **Docstrings**: Each method has a brief docstring explaining its purpose
6. **Test Coverage**: 5 comprehensive test classes covering:
   - Basic arithmetic
   - Operator precedence
   - Parentheses grouping
   - Unary minus
   - Error cases

The parser correctly handles:
- Operator precedence (multiplication/division before addition/subtraction)
- Parentheses for grouping
- Unary minus operations
- Floating point numbers
- Nested expressions
- Multiple unary operators (e.g., `--5`)