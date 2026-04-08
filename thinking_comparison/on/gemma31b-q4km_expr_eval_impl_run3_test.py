import re
from typing import List

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic, 
    parentheses, unary operators, and floating point numbers.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: The mathematical expression to evaluate.
            
        Returns:
            The result of the evaluation.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens, 
                        has mismatched parentheses, or results in division by zero.
        """
        # Tokenize the input string
        # Matches floating point numbers, integers, and operators/parentheses
        self._tokens = re.findall(r'\d*\.\d+|\d+|[+\-*/()]', expr)
        
        # Check for invalid characters by comparing reconstructed string length
        # (ignoring whitespace) to the original string length
        cleaned_expr = "".join(expr.split())
        if len("".join(self._tokens)) != len(cleaned_expr):
            raise ValueError("Expression contains invalid tokens")

        if not self._tokens:
            raise ValueError("Expression is empty")

        self._pos = 0
        result = self._parse_expression()

        # If we finished parsing but tokens remain, it's a syntax error (e.g., mismatched closing paren)
        if self._pos < len(self._tokens):
            raise ValueError("Mismatched parentheses or trailing tokens")

        return float(result)

    def _peek(self) -> str:
        """Returns the current token without advancing the position."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else ""

    def _consume(self) -> str:
        """Returns the current token and advances the position."""
        token = self._peek()
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parses addition and subtraction (lowest precedence)."""
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
        """Parses multiplication and division."""
        node = self._parse_unary()

        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_unary()
            if op == '*':
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                node /= right
        return node

    def _parse_unary(self) -> float:
        """Parses unary minus operators."""
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parses numbers and grouped expressions (highest precedence)."""
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing closing bracket")
            return result

        if token == ')':
            raise ValueError("Mismatched parentheses: unexpected closing bracket")

        # Attempt to parse as a number
        try:
            self._consume()
            return float(token)
        except (ValueError, IndexError):
            raise ValueError(f"Invalid token or unexpected end of expression: {token}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("15 / 4") == 3.75

def test_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0

def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_errors(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3")