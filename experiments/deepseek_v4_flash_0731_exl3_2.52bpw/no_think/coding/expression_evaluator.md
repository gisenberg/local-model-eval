Here's a complete implementation of a mathematical expression evaluator using recursive descent parsing:

```python
from typing import List, Optional, Union


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self) -> None:
        """Initialize the evaluator with an empty token list."""
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.

        Args:
            expr: The expression string to evaluate

        Returns:
            The numerical result of the expression

        Raises:
            ValueError: If the expression is invalid (mismatched parentheses,
                       division by zero, invalid tokens, or empty expression)
        """
        if not expr or not expr.strip():
            raise ValueError("Expression is empty")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        result = self._parse_expression()

        # Check if all tokens were consumed
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at position {self.pos}")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert the expression string into a list of tokens.

        Args:
            expr: The expression string to tokenize

        Returns:
            A list of tokens (numbers, operators, parentheses)

        Raises:
            ValueError: If an invalid character is found
        """
        tokens: List[str] = []
        i = 0

        while i < len(expr):
            char = expr[i]

            if char.isspace():
                i += 1
                continue

            if char.isdigit() or char == '.':
                # Parse number
                num_start = i
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                num_str = expr[num_start:i]

                # Validate number format
                try:
                    float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number format: '{num_str}'")

                tokens.append(num_str)
                continue

            if char in '+-*/()':
                tokens.append(char)
                i += 1
                continue

            raise ValueError(f"Invalid character '{char}' in expression")

        return tokens

    def _parse_expression(self) -> float:
        """
        Parse an expression (handles + and - operators).

        Returns:
            The evaluated result of the expression
        """
        result = self._parse_term()

        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            operator = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()

            if operator == '+':
                result += right
            else:
                result -= right

        return result

    def _parse_term(self) -> float:
        """
        Parse a term (handles * and / operators).

        Returns:
            The evaluated result of the term
        """
        result = self._parse_factor()

        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            operator = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()

            if operator == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right

        return result

    def _parse_factor(self) -> float:
        """
        Parse a factor (handles unary minus and parentheses).

        Returns:
            The evaluated result of the factor
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

            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing parenthesis")

            self.pos += 1
            return result

        # Handle numbers
        if self.pos < len(self.tokens) and self._is_number(self.tokens[self.pos]):
            value = float(self.tokens[self.pos])
            self.pos += 1
            return value

        # Handle unexpected token
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}'")
        else:
            raise ValueError("Unexpected end of expression")

    def _is_number(self, token: str) -> bool:
        """
        Check if a token is a valid number.

        Args:
            token: The token to check

        Returns:
            True if the token is a valid number, False otherwise
        """
        try:
            float(token)
            return True
        except ValueError:
            return False


# Test file: test_expression_evaluator.py
import pytest
from expression_evaluator import ExpressionEvaluator


def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("15 / 3") == 5.0


def test_precedence():
    """Test operator precedence."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 3*4 first, then +2
    assert evaluator.evaluate("10 - 2 * 3") == 4.0  # 2*3 first, then 10-6
    assert evaluator.evaluate("20 / 4 + 2") == 7.0  # 20/4 first, then +2
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0  # Both multiplications first


def test_parentheses():
    """Test parentheses grouping."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("(10 - 2) * (3 + 4)") == 56.0
    assert evaluator.evaluate("((2 + 3) * 4) / 5") == 4.0
    assert evaluator.evaluate("2 * (3 + 4) - 5") == 9.0


def test_unary_minus():
    """Test unary minus operator."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 - -3") == 8.0
    assert evaluator.evaluate("2 * -3") == -6.0
    assert evaluator.evaluate("-(-5)") == 5.0


def test_error_cases():
    """Test error handling."""
    evaluator = ExpressionEvaluator()

    # Empty expression
    with pytest.raises(ValueError, match="Expression is empty"):
        evaluator.evaluate("")

    with pytest.raises(ValueError, match="Expression is empty"):
        evaluator.evaluate("   ")

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

    with pytest.raises(ValueError, match="Invalid number format"):
        evaluator.evaluate("2 + 3.14.15")

    # Unexpected token
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3 *")

    with pytest.raises(ValueError, match="Unexpected end"):
        evaluator.evaluate("2 +")


def test_floating_point():
    """Test floating point numbers."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 + 2.86") == 6.0
    assert evaluator.evaluate("1.5 * 2.0") == 3.0
    assert evaluator.evaluate("10.5 / 2.5") == 4.2
    assert evaluator.evaluate("0.1 + 0.2") == pytest.approx(0.3)
```

This implementation provides:

1. **Correct operator precedence**: The recursive descent parser handles `+`/`-` at the expression level and `*`/`/` at the term level, ensuring multiplication and division are evaluated before addition and subtraction.

2. **Parentheses support**: The factor parser handles parentheses by recursively parsing the expression inside them.

3. **Unary minus**: The factor parser handles unary minus by negating the result of the next factor.

4. **Floating point support**: Numbers are parsed as strings and converted to floats, supporting decimal points.

5. **Error handling**: Comprehensive error messages for all specified error cases.

6. **Class structure**: Implemented as `ExpressionEvaluator` with an `evaluate` method.

7. **No eval()**: Uses pure recursive descent parsing.

8. **Type hints**: All methods have proper type hints and docstrings.

9. **5 pytest tests**: Covering basic arithmetic, precedence, parentheses, unary minus, and error cases (plus an additional floating point test).

The parser works by:
- Tokenizing the input string into numbers, operators, and parentheses
- Using recursive descent to parse expressions with proper precedence
- Handling unary operators at the factor level
- Validating the expression structure throughout parsing