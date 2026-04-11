import re
from typing import List, Tuple

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic,
    operator precedence, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        # Token patterns: floating point numbers, operators, and parentheses
        self._token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The result of the evaluation.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens, 
                         has mismatched parentheses, or division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize the input string
        tokens = self._tokenize(expr)
        self._pos = 0
        self._tokens = tokens

        result = self._parse_expression()

        # If there are tokens left after parsing the main expression, it's a syntax error
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the input string into a list of tokens."""
        # Remove whitespace
        expr = expr.replace(" ", "")
        
        # Check for invalid characters
        if re.search(r'[^0-9.+\-*/() ]', expr):
            raise ValueError("Expression contains invalid characters")

        return self._token_pattern.findall(expr)

    def _peek(self) -> str:
        """Returns the current token without advancing the position."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self) -> str:
        """Returns the current token and advances the position."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Handles addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._consume()
            term = self._parse_term()
            if op == '+':
                result += term
            else:
                result -= term
        return result

    def _parse_term(self) -> float:
        """Handles multiplication and division."""
        result = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            factor = self._parse_factor()
            if op == '*':
                result *= factor
            else:
                if factor == 0:
                    raise ValueError("Division by zero")
                result /= factor
        return result

    def _parse_factor(self) -> float:
        """Handles unary minus, parentheses, and numbers (highest precedence)."""
        token = self._peek()

        if token == '-':
            self._consume()  # Consume '-'
            return -self._parse_factor()

        if token == '+':
            self._consume()  # Consume '+' (unary plus)
            return self._parse_factor()

        if token == '(':
            self._consume()  # Consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()  # Consume ')'
            return result

        # Handle numbers
        try:
            token = self._consume()
            if token is None:
                raise ValueError("Unexpected end of expression")
            return float(token)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid token: {token}")

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 5 * 2") == 13.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_parentheses_and_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(3 + 5) * 2") == 16.0
    assert evaluator.evaluate("2 * (3 + 5)") == 16.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("5 + (-3)") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0

def test_value_errors():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(3 + 2")
    with pytest.raises(ValueError, match="Invalid characters"):
        evaluator.evaluate("3 + a")
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")

def test_complex_expression():
    evaluator = ExpressionEvaluator()
    # -2.5 + (10 / 2) * 3 = -2.5 + 15 = 12.5
    assert evaluator.evaluate("-2.5 + (10 / 2) * 3") == 12.5