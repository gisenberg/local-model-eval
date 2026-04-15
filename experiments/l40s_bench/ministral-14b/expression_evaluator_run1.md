# Ministral-3-14B-Instruct-2512-BF16 (L40S vLLM) — Expression Evaluator — Run 1

## Output

# Mathematical Expression Evaluator

Here's a complete implementation of an `ExpressionEvaluator` class using recursive descent parsing:

```python
import re
from typing import Optional, Tuple, Union

class ExpressionEvaluator:
    """A mathematical expression evaluator that supports basic arithmetic operations,
    parentheses, unary minus, and floating point numbers using recursive descent parsing."""

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result as a float.

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

        # Check for balanced parentheses
        if expr.count('(') != expr.count(')'):
            raise ValueError("Mismatched parentheses")

        # Initialize the parser
        self.tokens = []
        self.pos = 0
        self._tokenize(expr)

        # Start parsing from the expression level
        result = self._parse_expression()
        self._consume_eof()
        return result

    def _tokenize(self, expr: str) -> None:
        """Tokenize the input expression into numbers, operators, and parentheses.

        Args:
            expr: The input expression string
        """
        # Regular expression to match numbers (including decimals), operators, and parentheses
        token_pattern = r"""
            \d+\.?\d*|          # Numbers (integer or decimal)
            \.\d+|              # Decimal numbers without leading digit
            [+\-*/()]           # Operators and parentheses
        """

        tokens = re.findall(token_pattern, expr, re.VERBOSE)
        self.tokens = tokens

    def _consume_eof(self) -> None:
        """Check that we've consumed all tokens (end of file)."""
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at end of expression: {self.tokens[self.pos]}")

    def _peek(self) -> Optional[str]:
        """Peek at the next token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self) -> str:
        """Consume and return the next token."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse an expression (handles addition and subtraction with proper precedence)."""
        result = self._parse_term()

        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()

            if op == '+':
                result += right
            else:
                result -= right

        return result

    def _parse_term(self) -> float:
        """Parse a term (handles multiplication and division with proper precedence)."""
        result = self._parse_factor()

        while self._peek() in ('*', '/'):
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
        """Parse a factor (handles numbers, parentheses, and unary minus)."""
        token = self._peek()

        if token == '(':
            self._consume()  # Consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()  # Consume ')'
            return result

        if token == '-':
            self._consume()  # Consume '-'
            return -self._parse_factor()

        if token == '+':
            self._consume()  # Consume '+'
            return self._parse_factor()

        # Parse number
        return self._parse_number()

    def _parse_number(self) -> float:
        """Parse a number (integer or decimal)."""
        token = self._consume()

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
    assert evaluator.evaluate("2+3*4") == 14
    assert evaluator.evaluate("(2+3)*4") == 20
    assert evaluator.evaluate("10/2-3") == 2

def test_operator_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2+3*4") == 14  # Multiplication before addition
    assert evaluator.evaluate("2*3+4") == 10  # Multiplication before addition
    assert evaluator.evaluate("10/2-3") == 2   # Division before subtraction

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2+3)*4") == 20
    assert evaluator.evaluate("2*(3+4)") == 14
    assert evaluator.evaluate("((2+3)*4)") == 20

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3
    assert evaluator.evaluate("-(2+1)") == -3
    assert evaluator.evaluate("2*-3") == -6

def test_error_cases():
    evaluator = ExpressionEvaluator()

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2+3")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1/0")

    # Invalid token
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("2+abc")

    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")

    # Unexpected token
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2+3+")
```

## Implementation Notes

1. **Recursive Descent Parsing**: The implementation uses a classic recursive descent approach with methods for each level of precedence:
   - `_parse_expression()` handles addition and subtraction
   - `_parse_term()` handles multiplication and division
   - `_parse_factor()` handles numbers, parentheses, and unary operators

2. **Tokenization**: The input string is tokenized using regular expressions to identify numbers, operators, and parentheses.

3. **Error Handling**: The evaluator checks for:
   - Mismatched parentheses
   - Division by zero
   - Invalid number formats
   - Empty expressions
   - Unexpected tokens at the end of the expression

4. **Unary Minus**: Handled by checking for a '-' token when parsing factors and applying it to the following factor.

5. **Floating Point Support**: The parser converts all numbers to floats during evaluation.

The implementation follows the requirements precisely, avoiding the use of `eval()` or `ast.literal_eval()` and providing proper error messages for all specified error cases.
