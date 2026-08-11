Here's a complete implementation of a mathematical expression evaluator using recursive descent parsing:

```python
from typing import Union, Optional, List, Tuple


class ExpressionEvaluator:
    """
    A recursive descent parser and evaluator for mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self) -> None:
        """Initialize the evaluator with empty token list and position."""
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.

        Args:
            expr: String containing the mathematical expression

        Returns:
            float: The evaluated result

        Raises:
            ValueError: For invalid expressions, mismatched parentheses,
                       division by zero, or empty expressions
        """
        if not expr or not expr.strip():
            raise ValueError("Expression is empty")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        if self.pos >= len(self.tokens):
            raise ValueError("Expression contains no valid tokens")

        result = self._parse_expression()

        # Check if all tokens were consumed
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at position {self.pos}")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert expression string into a list of tokens.

        Args:
            expr: The expression string

        Returns:
            List[str]: List of tokens (numbers, operators, parentheses)

        Raises:
            ValueError: If invalid characters are found
        """
        tokens: List[str] = []
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
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    num_str += expr[i]
                    i += 1

                # Validate number format
                if num_str.count('.') > 1:
                    raise ValueError(f"Invalid number format: '{num_str}'")
                if num_str == '.':
                    raise ValueError(f"Invalid number format: '{num_str}'")

                tokens.append(num_str)
                continue

            # Handle operators and parentheses
            if char in '+-*/()':
                tokens.append(char)
                i += 1
                continue

            # Invalid character
            raise ValueError(f"Invalid character '{char}' at position {i}")

        return tokens

    def _parse_expression(self) -> float:
        """
        Parse and evaluate an expression (handles + and -).

        Returns:
            float: The evaluated value

        Raises:
            ValueError: For syntax errors
        """
        # Parse first term
        value = self._parse_term()

        # Handle subsequent + and - operations
        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]

            if token == '+':
                self.pos += 1
                right = self._parse_term()
                value += right
            elif token == '-':
                self.pos += 1
                right = self._parse_term()
                value -= right
            else:
                break

        return value

    def _parse_term(self) -> float:
        """
        Parse and evaluate a term (handles * and /).

        Returns:
            float: The evaluated value

        Raises:
            ValueError: For syntax errors or division by zero
        """
        # Parse first factor
        value = self._parse_factor()

        # Handle subsequent * and / operations
        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]

            if token == '*':
                self.pos += 1
                right = self._parse_factor()
                value *= right
            elif token == '/':
                self.pos += 1
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero")
                value /= right
            else:
                break

        return value

    def _parse_factor(self) -> float:
        """
        Parse and evaluate a factor (handles unary minus and parentheses).

        Returns:
            float: The evaluated value

        Raises:
            ValueError: For syntax errors or mismatched parentheses
        """
        # Handle unary minus
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1
            return -self._parse_factor()

        # Handle parentheses
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '(':
            self.pos += 1
            value = self._parse_expression()

            # Check for closing parenthesis
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing parenthesis")
            self.pos += 1
            return value

        # Handle numbers
        if self.pos < len(self.tokens) and self._is_number(self.tokens[self.pos]):
            value = float(self.tokens[self.pos])
            self.pos += 1
            return value

        # Handle unexpected tokens
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        else:
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}'")

    def _is_number(self, token: str) -> bool:
        """
        Check if a token is a valid number.

        Args:
            token: The token to check

        Returns:
            bool: True if token is a valid number, False otherwise
        """
        try:
            float(token)
            return True
        except ValueError:
            return False


# Test functions
import pytest


def test_basic_arithmetic() -> None:
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("15 / 3") == 5.0


def test_precedence() -> None:
    """Test operator precedence."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("20 / 4 + 3") == 8.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0


def test_parentheses() -> None:
    """Test parentheses for grouping."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("(10 - 2) / (3 + 1)") == 2.0
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == 15.0


def test_unary_minus() -> None:
    """Test unary minus operator."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 - -3") == 8.0
    assert evaluator.evaluate("-2 * 3") == -6.0
    assert evaluator.evaluate("-(5 + 2) * 2") == -14.0


def test_error_cases() -> None:
    """Test error handling."""
    evaluator = ExpressionEvaluator()

    # Empty expression
    with pytest.raises(ValueError, match="Expression is empty"):
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
        evaluator.evaluate("2 + 3 *")

    # Invalid number format
    with pytest.raises(ValueError, match="Invalid number format"):
        evaluator.evaluate("2..3 + 4")
```

This implementation provides:

1. **Correct operator precedence**: Multiplication and division are handled at the term level, while addition and subtraction are handled at the expression level, ensuring proper precedence.

2. **Parentheses support**: The factor parser handles parentheses recursively, allowing nested grouping.

3. **Unary minus**: The factor parser handles unary minus by recursively parsing the factor after the minus sign.

4. **Floating point numbers**: The tokenizer handles decimal numbers, and the parser converts them to floats.

5. **Error handling**: All specified error cases raise `ValueError` with descriptive messages.

6. **Class structure**: Implemented as `ExpressionEvaluator` class with `evaluate()` method.

7. **Recursive descent parser**: Uses recursive parsing methods (`_parse_expression`, `_parse_term`, `_parse_factor`) without using `eval()` or `ast.literal_eval()`.

8. **Type hints and docstrings**: All methods have type hints and brief docstrings.

9. **5 pytest tests**: Covering basic arithmetic, precedence, parentheses, unary minus, and error cases.

The parser follows the standard recursive descent pattern:
- `_parse_expression` handles addition and subtraction
- `_parse_term` handles multiplication and division
- `_parse_factor` handles numbers, parentheses, and unary minus

This ensures correct precedence and proper handling of all required features.