import re
from typing import List


class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic,
    operator precedence, parentheses, and unary operators.
    """

    def __init__(self) -> None:
        """Initializes the evaluator with empty state."""
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates a mathematical expression string.

        Args:
            expr: The string expression to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                       has mismatched parentheses, or results in division by zero.
        """
        self._tokens = self._tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        # If we haven't consumed all tokens, the expression is malformed (e.g., "1 + 2 3")
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token: '{self._tokens[self._pos]}'")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the input string into a list of valid tokens."""
        # Regex for numbers (including floats), operators, and parentheses
        token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')
        tokens = token_pattern.findall(expr)

        # Validate that the tokens reconstruct the original string (ignoring whitespace)
        # This effectively detects any invalid characters/tokens.
        if "".join(tokens) != "".join(expr.split()):
            raise ValueError("Invalid tokens in expression")

        return tokens

    def _peek(self) -> str:
        """Returns the current token without consuming it."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return ""

    def _consume(self) -> str:
        """Consumes and returns the current token."""
        token = self._peek()
        if not token:
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
        """Handles unary plus and minus."""
        token = self._peek()
        if token == '-':
            self._consume()
            return -self._parse_factor()
        if token == '+':
            self._consume()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and parenthesized expressions (highest precedence)."""
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()  # consume ')'
            return result

        if token == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")

        if not token:
            raise ValueError("Unexpected end of expression")

        try:
            # Attempt to convert the token to a float
            val = float(self._consume())
            return val
        except ValueError:
            raise ValueError(f"Invalid token: '{token}'")


# --- Pytest Tests ---
# To run these, save this file as `evaluator.py` and run `pytest evaluator.py`

import pytest

def test_basic_arithmetic():
    """Tests basic addition, subtraction, multiplication, and division."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 4") == 2.5
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_precedence():
    """Tests that operator precedence (PEMDAS) is respected."""
    evaluator = ExpressionEvaluator()
    # 1 + (2 * 3) = 7
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    # (10 / 2) + 3 = 8
    assert evaluator.evaluate("10 / 2 + 3") == 8.0
    # 10 - 2 * 4 = 2
    assert evaluator.evaluate("10 - 2 * 4") == 2.0

def test_parentheses():
    """Tests that parentheses correctly group operations."""
    evaluator = ExpressionEvaluator()
    # (1 + 2) * 3 = 9
    assert evaluator.evaluate("(1 + 2) * 3") == 9.0
    # 10 / (2 + 3) = 2
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    # Nested: (1 + (2 * 3)) = 7
    assert evaluator.evaluate("(1 + (2 * 3))") == 7.0

def test_unary_minus():
    """Tests unary minus and nested unary operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 + -3") == 2.0
    assert evaluator.evaluate("--5") == 5.0  # Double negation
    assert evaluator.evaluate("-3 + -(2 + 1)") == -6.0

def test_error_cases():
    """Tests that appropriate ValueErrors are raised for invalid inputs."""
    evaluator = ExpressionEvaluator()
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("1 + 2)")
        
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid tokens"):
        evaluator.evaluate("1 + a")
    
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")