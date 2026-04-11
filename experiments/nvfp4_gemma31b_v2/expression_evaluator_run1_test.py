import re
from typing import List, Tuple

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic,
    operator precedence, unary minus, and parentheses.
    """

    def __init__(self):
        # Token specification: numbers (float/int), operators, and parentheses
        self._token_pattern = re.compile(r'\s*([-+]?\.?\d*\.?\d+|[+\-*/() ])\s*')

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr (str): The expression to evaluate.
            
        Returns:
            float: The result of the evaluation.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens, 
                        mismatched parentheses, or division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize the input
        tokens = self._tokenize(expr)
        self._pos = 0
        self._tokens = tokens

        result = self._parse_expression()

        # If there are tokens left after parsing the main expression, it's an invalid format
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the expression string into a list of meaningful tokens."""
        # We use a regex to find all numbers and symbols, ignoring whitespace
        tokens = []
        # This regex handles floating points and separates operators
        # We use a simpler approach: find all non-whitespace sequences of digits/dots or single operators
        raw_tokens = re.findall(r'\d*\.\d+|\d+|[+\-*/()]', expr)
        
        # Basic validation: check if the original string had characters not captured by regex
        # (This catches invalid characters like 'a', '@', etc.)
        cleaned_expr = "".join(raw_tokens)
        # Remove whitespace from original to compare
        if len("".join(expr.split())) != len(cleaned_expr):
            raise ValueError("Expression contains invalid characters")
            
        return raw_tokens

    def _peek(self) -> str:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

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
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Handles multiplication and division."""
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
        """Handles unary minus, parentheses, and numbers (highest precedence)."""
        token = self._peek()

        if token == '-':
            self._consume() # consume '-'
            return -self._parse_factor()
        
        if token == '+':
            self._consume() # consume '+' (unary plus)
            return self._parse_factor()

        if token == '(':
            self._consume() # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume() # consume ')'
            return result

        # Handle numbers
        token = self._consume()
        try:
            return float(token)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid token: {token}")

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 - 3") == 5.0
    assert evaluator.evaluate("10 / 2 + 1") == 6.0

def test_parentheses_and_floats():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("5 + (-2)") == 3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0

def test_value_errors():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="invalid characters"):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")

def test_complex_expression():
    evaluator = ExpressionEvaluator()
    # -5 + (2 * 3.5) / -2  => -5 + 7 / -2 => -5 - 3.5 = -8.5
    assert evaluator.evaluate("-5 + (2 * 3.5) / -2") == -8.5