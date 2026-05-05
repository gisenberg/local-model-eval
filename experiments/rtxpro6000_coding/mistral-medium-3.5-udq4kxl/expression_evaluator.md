Here's a complete implementation of the `ExpressionEvaluator` class with all the requirements, including the pytest tests:

```python
from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic operations,
    parentheses, unary minus, and floating point numbers using recursive descent parsing.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.current_token: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: The mathematical expression to evaluate

        Returns:
            The result of the evaluation as a float

        Raises:
            ValueError: For syntax errors, division by zero, or invalid tokens
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.current_token = 0
        result = self._parse_expression()
        if self.current_token < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.current_token]}")
        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert the input string into a list of tokens.

        Args:
            expr: The input expression string

        Returns:
            List of tokens (numbers, operators, parentheses)
        """
        tokens = []
        i = 0
        n = len(expr)

        while i < n:
            c = expr[i]

            if c.isspace():
                i += 1
                continue

            if c in '+-*/()':
                tokens.append(c)
                i += 1
                continue

            if c.isdigit() or c == '.':
                # Parse number (including floating point)
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                num_str = expr[start:i]
                try:
                    float(num_str)  # Validate it's a proper number
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")
                tokens.append(num_str)
                continue

            raise ValueError(f"Invalid character: {c}")

        return tokens

    def _peek(self) -> str:
        """Return the current token without consuming it."""
        if self.current_token >= len(self.tokens):
            return ''
        return self.tokens[self.current_token]

    def _consume(self) -> str:
        """Return the current token and advance to the next one."""
        if self.current_token >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        token = self.tokens[self.current_token]
        self.current_token += 1
        return token

    def _parse_expression(self) -> float:
        """Parse an expression with addition and subtraction."""
        result = self._parse_term()

        while self._peek() in '+-':
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right

        return result

    def _parse_term(self) -> float:
        """Parse a term with multiplication and division."""
        result = self._parse_factor()

        while self._peek() in '*/':
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right

        return result

    def _parse_factor(self) -> float:
        """Parse a factor (number, parenthesized expression, or unary minus)."""
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()  # consume ')'
            return result

        if token == '-':
            self._consume()  # consume '-'
            return -self._parse_factor()

        if token == '+':
            self._consume()  # consume '+'
            return self._parse_factor()

        # Must be a number
        num_str = self._consume()
        try:
            return float(num_str)
        except ValueError:
            raise ValueError(f"Invalid number: {num_str}")

# Pytest tests
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("5 - 3") == 2.0
    assert evaluator.evaluate("4 * 3") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("8 / 2 * 3") == 12.0
    assert evaluator.evaluate("2 + 3 * 4 / 2 - 1") == 7.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 - (2 * 3)") == 4.0
    assert evaluator.evaluate("((8 / 2) * 3)") == 12.0
    assert evaluator.evaluate("2 + (3 * (4 - 1))") == 11.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0
    assert evaluator.evaluate("-3.5 + 2") == -1.5

def test_error_cases():
    evaluator = ExpressionEvaluator()

    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + 3 $ 4")

    # Invalid number
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("2 + 3.4.5")
```

### Key Features:

1. **Recursive Descent Parser**: The implementation uses a classic recursive descent approach with separate methods for each precedence level:
   - `_parse_expression` handles + and -
   - `_parse_term` handles * and /
   - `_parse_factor` handles numbers, parentheses, and unary operators

2. **Tokenization**: The `_tokenize` method properly handles:
   - Whitespace skipping
   - Multi-digit numbers
   - Floating point numbers
   - Operator and parenthesis tokens

3. **Error Handling**: All required error cases are handled with descriptive messages:
   - Empty expressions
   - Mismatched parentheses
   - Division by zero
   - Invalid tokens
   - Invalid numbers

4. **Type Hints**: All methods include proper type hints as requested.

5. **Testing**: The pytest tests cover all required cases:
   - Basic arithmetic operations
   - Operator precedence
   - Parentheses grouping
   - Unary minus
   - Various error cases

To run the tests, you would need to have pytest installed (`pip install pytest`) and then run `pytest` in the same directory as this code.