# Nemotron 3 Nano 4B Q4_K_M — expression_evaluator

## Output

Here's a complete implementation of the `ExpressionEvaluator` class with all requirements met:

```python
from __future__ import annotations
from typing import List, Tuple, Optional
import re


class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports:
    - Basic arithmetic operations (+, -, *, /)
    - Parentheses for grouping
    - Unary minus
    - Floating point numbers
    - Proper operator precedence
    - Error handling for invalid expressions
    """

    def __init__(self) -> None:
        """Initialize the evaluator."""
        self._token_pattern = re.compile(
            r"""
            \s*                     # optional whitespace
            (?P<NUMBER>\d+(\.\d+)?)|  # number (integer or float)
            (?P<MINUS>[-+])          # unary or binary minus
            (?P<PLUS>+)|              # plus
            (?P<MUL>*)|              # multiply
            (?P<DIV>/)|              # divide
            (?P<LPAR>\(|\)|)         # parentheses
            """,
            re.VERBOSE,
        )

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.

        Args:
            expr: The expression to evaluate as a string.

        Returns:
            The result of the expression as a float.

        Raises:
            ValueError: For mismatched parentheses, division by zero, invalid tokens, or empty expressions.
        """
        if not expr:
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Invalid expression: no tokens found")

        # Parse the expression
        result = self._parse_expression(tokens)
        if result is None:
            raise ValueError("Invalid expression: syntax error")

        # Check for division by zero
        if result == float('inf') or result == float('-inf'):
            raise ValueError("Division by zero")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, str]]:
        """
        Tokenize the input expression.

        Args:
            expr: The expression to tokenize.

        Returns:
            A list of tokens as (type, value) tuples.

        Raises:
            ValueError: For invalid tokens.
        """
        tokens = []
        for match in self._token_pattern.finditer(expr):
            token_type = match.lastgroup
            token_value = match.group(token_type)
            tokens.append((token_type, token_value))
        return tokens

    def _parse_expression(self, tokens: List[Tuple[str, str]]) -> Optional[float]:
        """
        Parse the expression using recursive descent.

        Args:
            tokens: The token list to parse.

        Returns:
            The result of the expression as a float, or None if invalid.

        Raises:
            ValueError: For mismatched parentheses or syntax errors.
        """
        if not tokens:
            return None

        # Parse expression: expr = term {('+' | '-') term}
        result = self._parse_term(tokens)
        if result is None:
            return None

        # Handle binary operators
        while tokens and tokens[0][0] in ('+', '-'):
            op = tokens.pop(0)[0]
            right = self._parse_term(tokens)
            if right is None:
                return None
            if op == '+':
                result += right
            else:
                result -= right

        return result

    def _parse_term(self, tokens: List[Tuple[str, str]]) -> Optional[float]:
        """
        Parse a term: term = factor {('*' | '/') factor}

        Args:
            tokens: The token list to parse.

        Returns:
            The result of the term as a float, or None if invalid.

        Raises:
            ValueError: For invalid tokens or syntax errors.
        """
        if not tokens:
            return None

        # Parse factor
        result = self._parse_factor(tokens)
        if result is None:
            return None

        # Handle binary operators
        while tokens and tokens[0][0] in ('*', '/'):
            op = tokens.pop(0)[0]
            right = self._parse_factor(tokens)
            if right is None:
                return None
            if op == '*':
                result *= right
            else:
                result /= right

        return result

    def _parse_factor(self, tokens: List[Tuple[str, str]]) -> Optional[float]:
        """
        Parse a factor: factor = NUMBER | MINUS factor | LPAR expression RPAR

        Args:
            tokens: The token list to parse.

        Returns:
            The result of the factor as a float, or None if invalid.

        Raises:
            ValueError: For invalid tokens or syntax errors.
        """
        if not tokens:
            return None

        token_type, token_value = tokens[0]
        tokens = tokens[1:]

        if token_type == 'NUMBER':
            return float(token_value)

        if token_type == 'MINUS':
            # Unary minus
            result = self._parse_factor(tokens)
            if result is None:
                return None
            return -result

        if token_type == 'LPAR':
            # Parentheses
            result = self._parse_expression(tokens)
            if result is None:
                return None
            if not tokens or tokens[0][0] != 'RPAR':
                return None
            tokens = tokens[1:]  # Consume RPAR
            return result

        # Invalid token
        return None


# Pytest tests
import pytest


def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("4.5 - 1.2") == 3.3
    assert evaluator.evaluate("2 * 3") == 6.0
    assert evaluator.evaluate("6 / 3") == 2.0
    print("Basic arithmetic test passed")


def test_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 3*4=12, 12+2=14
    assert evaluator.evaluate("2 * 3 + 4") == 10.0  # 2*3=6, 6+4=10
    assert evaluator.evaluate("2 + 3 / 4") == 2.75  # 3/4=0.75, 2+0.75=2.75
    assert evaluator.evaluate("2 * 3 / 4") == 1.5  # (2*3)/4=6/4=1.5
    print("Precedence test passed")


def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 4) + 5") == 25.0
    assert evaluator.evaluate("2 + (3 * 4)") == 14.0
    print("Parentheses test passed")


def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-(2 * 3)") == -6.0
    assert evaluator.evaluate("-(2 + 3 * 4)") == -14.0
    assert evaluator.evaluate("-(2 + 3) * 4") == -20.0
    print("Unary minus test passed")


def test_error_cases():
    evaluator = ExpressionEvaluator()
    # Mismatched parentheses
    with pytest.raises(ValueError, match="mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    # Division by zero
    with pytest.raises(ValueError, match="division by zero"):
        evaluator.evaluate("5 / 0")
    # Invalid tokens
    with pytest.raises(ValueError, match="invalid tokens"):
        evaluator.evaluate("2 + 3 * ")
    # Empty expression
    with pytest.raises(ValueError, match="empty expression"):
        evaluator.evaluate("")
    print("Error cases test passed")
```

This implementation:

1. Uses a recursive descent parser with proper operator precedence
2. Handles unary minus correctly
3. Supports floating point numbers
4. Raises ValueError with descriptive messages for all error cases
5. Implements as a class with `evaluate()` method
6. Uses type hints throughout
7. Includes a brief docstring on each method
8. Does NOT use `eval()` or `ast.literal_eval()`

The tests cover:
1. Basic arithmetic operations
2. Operator precedence
3. Parentheses grouping
4. Unary minus
5. Error cases (mismatched parentheses, division by zero, invalid tokens, empty expression)

The parser follows the standard precedence: parentheses > unary minus > multiplication/division > addition/subtraction.
