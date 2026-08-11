import re
from typing import List, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic operations,
    parentheses, unary minus, and floating-point numbers.

    Supported operators:
        + (addition), - (subtraction), * (multiplication), / (division)

    The parser follows standard operator precedence rules:
        1. Parentheses
        2. Unary minus
        3. Multiplication and Division (left-to-right)
        4. Addition and Subtraction (left-to-right)

    Usage:
        evaluator = ExpressionEvaluator()
        result = evaluator.evaluate("3 + 4 * 2")
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string.

        Args:
            expr: A string containing the mathematical expression to evaluate.

        Returns:
            The result of the evaluated expression as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        # Tokenize the expression
        self.tokens = self._tokenize(expr)
        self.pos = 0

        try:
            result = self._parse_expression()
            if self.pos < len(self.tokens):
                raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
            return result
        except IndexError:
            raise ValueError("Unexpected end of expression")

    def _tokenize(self, expr: str) -> List[str]:
        """Tokenizes the input expression string."""
        token_pattern = r'\d+\.?\d*|\.\d+|[+\-*/()]|\S'
        tokens = re.findall(token_pattern, expr.replace(' ', ''))
        if not tokens:
            raise ValueError("No valid tokens found")
        return tokens

    def _parse_expression(self) -> float:
        """
        Parses an expression according to the grammar rule:
        expression -> term (('+' | '-') term)*
        """
        result = self._parse_term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """
        Parses a term according to the grammar rule:
        term -> factor (('*' | '/') factor)*
        """
        result = self._parse_factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
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
        Parses a factor according to the grammar rule:
        factor -> number | '(' expression ')' | '-' factor
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")

        token = self.tokens[self.pos]

        # Handle unary minus
        if token == '-':
            self.pos += 1
            return -self._parse_factor()

        # Handle parentheses
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result

        # Handle numbers
        try:
            value = float(token)
        except ValueError:
            raise ValueError(f"Invalid token: {token}")

        self.pos += 1
        return value

import pytest

def test_basic_operations():
    """Test basic arithmetic operations and operator precedence."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_unary_minus():
    """Test unary minus functionality."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 * -2") == -6.0
    assert evaluator.evaluate("-3 + 5") == 2.0

def test_floating_point_numbers():
    """Test support for floating point numbers."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 + 2.86") == pytest.approx(6.0)
    assert evaluator.evaluate("1.5 * 2.0") == pytest.approx(3.0)
    assert evaluator.evaluate("10.0 / 4") == pytest.approx(2.5)
    assert evaluator.evaluate(".5 + .25") == pytest.approx(0.75)

def test_error_handling():
    """Test that appropriate errors are raised for invalid inputs."""
    evaluator = ExpressionEvaluator()

    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

    # Invalid token
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + abc")

    # Unexpected end of expression
    with pytest.raises(ValueError, match="Unexpected end of expression"):
        evaluator.evaluate("2 +")

def test_complex_expressions():
    """Test more complex expressions to ensure correctness."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert evaluator.evaluate("100 / (5 * 2) - 3") == 7.0
    assert evaluator.evaluate("-((2 + 3) * 2)") == -10.0
    assert evaluator.evaluate("3.5 * 2 - 1.5 / 3") == pytest.approx(6.5)