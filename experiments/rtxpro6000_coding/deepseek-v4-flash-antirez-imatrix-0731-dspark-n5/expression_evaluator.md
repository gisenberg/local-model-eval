Here's a complete implementation of a mathematical expression evaluator using recursive descent parsing:

```python
from typing import Union, Optional, List, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser and evaluator for mathematical expressions.
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
            ValueError: If the expression is invalid or contains errors
        """
        # Reset state for each evaluation
        self.tokens = self._tokenize(expr)
        self.pos = 0

        if not self.tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        # Check for leftover tokens
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: '{self.tokens[self.pos]}'")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert a string expression into a list of tokens.

        Args:
            expr: String expression to tokenize

        Returns:
            List[str]: List of tokens

        Raises:
            ValueError: If invalid characters are found
        """
        tokens: List[str] = []
        i = 0

        while i < len(expr):
            char = expr[i]

            # Skip whitespace
            if char.isspace():
                i += 1
                continue

            # Handle numbers (including decimals)
            if char.isdigit() or char == '.':
                num_start = i
                has_decimal = False

                while i < len(expr):
                    if expr[i].isdigit():
                        i += 1
                    elif expr[i] == '.' and not has_decimal:
                        has_decimal = True
                        i += 1
                    else:
                        break

                # Check if the number is valid
                num_str = expr[num_start:i]
                if num_str == '.' or num_str == '':
                    raise ValueError(f"Invalid number at position {num_start}")

                tokens.append(num_str)
                continue

            # Handle operators and parentheses
            if char in '+-*/()':
                tokens.append(char)
                i += 1
                continue

            # Handle invalid characters
            raise ValueError(f"Invalid character '{char}' at position {i}")

        return tokens

    def _parse_expression(self) -> float:
        """
        Parse and evaluate an expression (handles + and -).

        Returns:
            float: The evaluated value
        """
        result = self._parse_term()

        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]

            if token == '+':
                self.pos += 1
                result += self._parse_term()
            elif token == '-':
                self.pos += 1
                result -= self._parse_term()
            else:
                break

        return result

    def _parse_term(self) -> float:
        """
        Parse and evaluate a term (handles * and /).

        Returns:
            float: The evaluated value
        """
        result = self._parse_factor()

        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]

            if token == '*':
                self.pos += 1
                result *= self._parse_factor()
            elif token == '/':
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
        Parse and evaluate a factor (handles unary minus and parentheses).

        Returns:
            float: The evaluated value
        """
        # Handle unary minus
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1
            return -self._parse_factor()

        # Handle unary plus (optional)
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '+':
            self.pos += 1
            return self._parse_factor()

        # Handle parentheses
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '(':
            self.pos += 1
            result = self._parse_expression()

            # Check for matching closing parenthesis
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing parenthesis")

            self.pos += 1
            return result

        # Handle numbers
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]

            # Check if token is a valid number
            try:
                value = float(token)
            except ValueError:
                raise ValueError(f"Invalid token '{token}' at position {self.pos}")

            self.pos += 1
            return value

        raise ValueError("Unexpected end of expression")

    def _check_parentheses(self) -> None:
        """
        Check for balanced parentheses in the token list.

        Raises:
            ValueError: If parentheses are mismatched
        """
        stack: List[str] = []

        for token in self.tokens:
            if token == '(':
                stack.append(token)
            elif token == ')':
                if not stack:
                    raise ValueError("Mismatched parentheses: extra closing parenthesis")
                stack.pop()

        if stack:
            raise ValueError("Mismatched parentheses: missing closing parenthesis")


# Test cases
import pytest

class TestExpressionEvaluator:
    """Test cases for ExpressionEvaluator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.evaluator = ExpressionEvaluator()

    def test_basic_arithmetic(self) -> None:
        """Test basic arithmetic operations."""
        assert self.evaluator.evaluate("2 + 3") == 5.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("6 * 7") == 42.0
        assert self.evaluator.evaluate("20 / 5") == 4.0

    def test_precedence(self) -> None:
        """Test operator precedence."""
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
        assert self.evaluator.evaluate("10 - 2 * 3") == 4.0
        assert self.evaluator.evaluate("20 / 2 + 5") == 15.0
        assert self.evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

    def test_parentheses(self) -> None:
        """Test parentheses grouping."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert self.evaluator.evaluate("(10 - 2) * (3 + 1)") == 32.0
        assert self.evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20.0

    def test_unary_minus(self) -> None:
        """Test unary minus operator."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-(2 + 1)") == -3.0
        assert self.evaluator.evaluate("-2 + 5") == 3.0
        assert self.evaluator.evaluate("3 * -2") == -6.0
        assert self.evaluator.evaluate("--5") == 5.0

    def test_error_cases(self) -> None:
        """Test error handling."""
        # Empty expression
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("")

        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("5 / 0")

        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("(2 + 3")

        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("2 + 3)")

        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 + @")

        with pytest.raises(ValueError, match="Invalid token"):
            self.evaluator.evaluate("2 + abc")

        # Invalid number
        with pytest.raises(ValueError, match="Invalid number"):
            self.evaluator.evaluate("2 + .")

        # Unexpected end
        with pytest.raises(ValueError, match="Unexpected end"):
            self.evaluator.evaluate("2 +")

        # Unexpected token
        with pytest.raises(ValueError, match="Unexpected token"):
            self.evaluator.evaluate("2 + 3 4")
```

This implementation includes:

1. **Tokenization**: Converts the string expression into a list of tokens, handling numbers (including decimals), operators, and parentheses.

2. **Recursive Descent Parsing**: Uses three levels of parsing:
   - `_parse_expression()`: Handles addition and subtraction
   - `_parse_term()`: Handles multiplication and division
   - `_parse_factor()`: Handles unary operators, parentheses, and numbers

3. **Error Handling**: Raises `ValueError` with descriptive messages for:
   - Empty expressions
   - Division by zero
   - Mismatched parentheses
   - Invalid tokens
   - Invalid numbers
   - Unexpected end of expression
   - Unexpected tokens

4. **Type Hints**: All methods have complete type hints.

5. **Test Coverage**: 5 pytest test classes covering:
   - Basic arithmetic
   - Operator precedence
   - Parentheses grouping
   - Unary minus
   - Error cases

The parser correctly handles:
- Operator precedence (multiplication/division before addition/subtraction)
- Parentheses for grouping
- Unary minus (including double negatives like `--5`)
- Floating point numbers
- Nested parentheses
- Complex expressions like `((2 + 3) * 4)`