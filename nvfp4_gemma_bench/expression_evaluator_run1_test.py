import re
from typing import List, Tuple

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic,
    operator precedence, parentheses, and unary minus.
    """

    def __init__(self):
        # Token patterns: float/int, operators, and parentheses
        self._token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: The string expression to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: For division by zero, mismatched parentheses, 
                         invalid tokens, or empty expressions.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize the input
        tokens = self._tokenize(expr)
        self._pos = 0
        self._tokens = tokens

        result = self._parse_expression()

        # If we haven't consumed all tokens, there's a syntax error
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the input string into a list of valid tokens."""
        # Remove whitespace
        expr = expr.replace(" ", "")
        
        # Find all matches for the pattern
        tokens = self._token_pattern.findall(expr)
        
        # Validate that the reconstructed string matches the original (detects invalid characters)
        if "".join(tokens) != expr:
            # Find the first character that didn't match
            raise ValueError("Expression contains invalid characters")
            
        return tokens

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
        node = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                node += right
            else:
                node -= right
        return node

    def _parse_term(self) -> float:
        """Handles multiplication and division."""
        node = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                node /= right
        return node

    def _parse_factor(self) -> float:
        """Handles unary minus, parentheses, and numbers (highest precedence)."""
        token = self._peek()

        if token == '-':
            self._consume()  # consume '-'
            return -self._parse_factor()

        if token == '+':
            self._consume()  # consume '+' (unary plus)
            return self._parse_factor()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()  # consume ')'
            return result

        # Handle numbers
        try:
            token = self._consume()
            if token is None:
                raise ValueError("Expected number or parenthesis")
            return float(token)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid token: {token}")

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 5 * 2") == 13.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_parentheses_and_unary():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    assert evaluator.evaluate("-3 + (-2)") == -5.0
    assert evaluator.evaluate("(-5 * -2)") == 10.0

def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

def test_invalid_syntax():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Invalid characters"):
        evaluator.evaluate("3 + 2 @ 1")
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")

def test_complex_expression():
    evaluator = ExpressionEvaluator()
    # -2.5 * (10 + 2) / 3 = -30 / 3 = -10
    assert evaluator.evaluate("-2.5 * (10 + 2) / 3") == -10.0