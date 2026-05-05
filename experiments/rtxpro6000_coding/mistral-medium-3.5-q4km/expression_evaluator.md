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

        Raises:
            ValueError: For invalid characters or malformed numbers
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
                # Parse number (including decimals)
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                num_str = expr[start:i]

                # Validate number format
                if num_str.count('.') > 1:
                    raise ValueError(f"Invalid number: {num_str}")
                if num_str.startswith('.') or num_str.endswith('.'):
                    raise ValueError(f"Invalid number: {num_str}")

                tokens.append(num_str)
                continue

            raise ValueError(f"Invalid character: {c}")

        return tokens

    def _parse_expression(self) -> float:
        """
        Parse an expression with addition and subtraction.

        Returns:
            The result of the expression
        """
        result = self._parse_term()

        while self.current_token < len(self.tokens) and self.tokens[self.current_token] in '+-':
            op = self.tokens[self.current_token]
            self.current_token += 1
            right = self._parse_term()

            if op == '+':
                result += right
            else:
                result -= right

        return result

    def _parse_term(self) -> float:
        """
        Parse a term with multiplication and division.

        Returns:
            The result of the term
        """
        result = self._parse_factor()

        while self.current_token < len(self.tokens) and self.tokens[self.current_token] in '*/':
            op = self.tokens[self.current_token]
            self.current_token += 1
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
        Parse a factor which can be a number, unary minus, or parenthesized expression.

        Returns:
            The result of the factor
        """
        token = self.tokens[self.current_token]

        if token == '(':
            self.current_token += 1
            result = self._parse_expression()
            if self.current_token >= len(self.tokens) or self.tokens[self.current_token] != ')':
                raise ValueError("Mismatched parentheses")
            self.current_token += 1
            return result

        if token == '-':
            # Unary minus
            self.current_token += 1
            return -self._parse_factor()

        if token == '+':
            # Unary plus (though not required, we'll support it for completeness)
            self.current_token += 1
            return self._parse_factor()

        # Should be a number
        try:
            num = float(token)
            self.current_token += 1
            return num
        except ValueError:
            raise ValueError(f"Invalid token: {token}")

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
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # * has higher precedence
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("8 / 2 + 3") == 7.0   # / and * have same precedence, left to right
    assert evaluator.evaluate("2 + 3 * 4 / 2") == 8.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 - (2 * 3)") == 4.0
    assert evaluator.evaluate("((8 / 2) + 3) * 2") == 16.0
    assert evaluator.evaluate("2 * (3 + (4 * 5))") == 46.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 * -2") == -10.0
    assert evaluator.evaluate("-3.14") == -3.14
    assert evaluator.evaluate("2 + -3") == -1.0

def test_error_cases():
    evaluator = ExpressionEvaluator()

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
        evaluator.evaluate("2 + 3 $ 4")

    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + abc")

    # Invalid numbers
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("2 + 3.4.5")

    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("2 + .3")

    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("2 + 3.")
```

### Key Features:

1. **Recursive Descent Parser**: The implementation uses a classic recursive descent approach with separate methods for each precedence level:
   - `_parse_expression()` handles + and -
   - `_parse_term()` handles * and /
   - `_parse_factor()` handles numbers, unary operators, and parentheses

2. **Tokenization**: The `_tokenize()` method converts the input string into a list of tokens, handling numbers (including decimals), operators, and parentheses.

3. **Error Handling**: The code raises `ValueError` with descriptive messages for:
   - Empty expressions
   - Mismatched parentheses
   - Division by zero
   - Invalid characters or tokens
   - Malformed numbers

4. **Type Hints**: All methods include type hints as requested.

5. **Test Coverage**: The pytest tests cover:
   - Basic arithmetic operations
   - Operator precedence
   - Parentheses grouping
   - Unary minus (including with parentheses)
   - Various error cases

To run the tests, you'll need to have pytest installed (`pip install pytest`) and then run `pytest` in the directory containing this code.