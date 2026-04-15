# Ministral-3-14B-Instruct-2512-BF16 (L40S vLLM) — Expression Evaluator — Run 3

## Output

# Mathematical Expression Evaluator

Here's a Python implementation of an `ExpressionEvaluator` class using recursive descent parsing that meets all your requirements:

```python
import re
from typing import Optional, Union

class ExpressionEvaluator:
    """A mathematical expression evaluator that supports basic arithmetic operations,
    parentheses, unary minus, and floating point numbers using recursive descent parsing."""

    def __init__(self):
        self.tokens: list[str] = []
        self.current_token_index: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string.

        Args:
            expr: The expression to evaluate (e.g., "3 + 4 * 2")

        Returns:
            The result of the evaluated expression as a float

        Raises:
            ValueError: For mismatched parentheses, division by zero, invalid tokens, or empty expressions
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.current_token_index = 0

        try:
            result = self._parse_expression()
        except IndexError:
            raise ValueError("Unexpected end of expression")

        if self.current_token_index != len(self.tokens):
            raise ValueError(f"Unexpected token at end: {self._peek_token()}")

        return result

    def _tokenize(self, expr: str) -> list[str]:
        """Convert the input string into a list of tokens.

        Args:
            expr: The input expression string

        Returns:
            List of tokens (numbers, operators, parentheses)
        """
        # This regex matches numbers (including decimals), operators, and parentheses
        token_pattern = r"""
            \d+\.?\d*|          # Numbers (integers or decimals)
            \+\+?|              # + or ++ (unary + is ignored)
            -+?|                # - or -- (unary minus)
            \*+?|               # * or ** (exponentiation not supported)
            /+?|                # / or //
            \(|\)               # Parentheses
        """
        tokens = re.findall(token_pattern, expr, re.VERBOSE)

        # Filter out unary plus (++ becomes +)
        filtered_tokens = []
        for i, token in enumerate(tokens):
            if token == '++':
                filtered_tokens.append('+')
            elif token == '--':
                filtered_tokens.append('-')
            else:
                filtered_tokens.append(token)

        return filtered_tokens

    def _peek_token(self) -> str:
        """Peek at the current token without consuming it.

        Returns:
            The current token
        """
        return self.tokens[self.current_token_index]

    def _consume_token(self) -> str:
        """Consume and return the current token, moving to the next one.

        Returns:
            The consumed token
        """
        token = self.tokens[self.current_token_index]
        self.current_token_index += 1
        return token

    def _parse_expression(self) -> float:
        """Parse an expression (addition and subtraction with lowest precedence).

        Returns:
            The result of the parsed expression
        """
        result = self._parse_term()

        while self._peek_token() in ('+', '-'):
            operator = self._consume_token()
            right = self._parse_term()

            if operator == '+':
                result += right
            else:
                result -= right

        return result

    def _parse_term(self) -> float:
        """Parse a term (multiplication and division with higher precedence).

        Returns:
            The result of the parsed term
        """
        result = self._parse_factor()

        while self._peek_token() in ('*', '/'):
            operator = self._consume_token()
            right = self._parse_factor()

            if operator == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right

        return result

    def _parse_factor(self) -> float:
        """Parse a factor (parentheses, unary minus, or numbers).

        Returns:
            The result of the parsed factor
        """
        token = self._peek_token()

        if token == '(':
            self._consume_token()  # Consume '('
            result = self._parse_expression()

            if self._peek_token() != ')':
                raise ValueError(f"Expected ')' but found '{self._peek_token()}'")

            self._consume_token()  # Consume ')'
            return result

        if token == '-':
            self._consume_token()  # Consume '-'
            return -self._parse_factor()

        if token == '+':
            self._consume_token()  # Consume '+'
            return self._parse_factor()

        # Must be a number
        return self._parse_number()

    def _parse_number(self) -> float:
        """Parse a number token and convert it to float.

        Returns:
            The parsed number as float

        Raises:
            ValueError: If the token is not a valid number
        """
        token = self._consume_token()

        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid number: {token}")
```

## Test Cases

Here are 5 pytest tests covering the requirements:

```python
import pytest
from expression_evaluator import ExpressionEvaluator

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14
    assert evaluator.evaluate("10 - 5 / 2") == 7.5
    assert evaluator.evaluate("3 * 4 + 5") == 17

def test_operator_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14  # Multiplication first
    assert evaluator.evaluate("(2 + 3) * 4") == 20  # Parentheses override
    assert evaluator.evaluate("10 - 5 / 2 + 3") == 8.5

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20
    assert evaluator.evaluate("2 * (3 + 4)") == 14
    assert evaluator.evaluate("((2 + 3) * 4) + 5") == 25

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2
    assert evaluator.evaluate("-(2 + 3)") == -5
    assert evaluator.evaluate("3 * -2") == -6

def test_error_cases():
    evaluator = ExpressionEvaluator()

    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Expected ')'"):
        evaluator.evaluate("(2 + 3")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")

    # Invalid token
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("2 + x")

    # Unexpected token at end
    with pytest.raises(ValueError, match="Unexpected token at end"):
        evaluator.evaluate("2 + 3 +")
```

## Implementation Notes

1. **Tokenization**: The `_tokenize` method uses regex to break the input string into meaningful tokens (numbers, operators, parentheses).

2. **Recursive Descent Parsing**: The parser follows the standard recursive descent approach with methods for each level of precedence:
   - `_parse_expression` handles addition and subtraction (lowest precedence)
   - `_parse_term` handles multiplication and division (higher precedence)
   - `_parse_factor` handles numbers, unary operators, and parentheses (highest precedence)

3. **Unary Minus**: Handled by checking for '-' when we expect a factor and then negating the result of parsing the next factor.

4. **Error Handling**: The evaluator checks for:
   - Empty expressions
   - Mismatched parentheses
   - Division by zero
   - Invalid tokens
   - Unexpected tokens at the end of the expression

5. **Type Hints**: All methods include type hints for better code clarity and IDE support.

The implementation follows the requirements strictly, avoiding `eval()` and using recursive descent parsing instead.
