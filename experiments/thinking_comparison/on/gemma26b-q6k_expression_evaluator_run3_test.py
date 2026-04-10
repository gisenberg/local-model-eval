import re
from typing import List

class ExpressionEvaluator:
    """
    A class to evaluate mathematical expressions using a recursive descent parser.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self) -> None:
        """Initializes the evaluator with empty state."""
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.

        Args:
            expr: The string expression to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        self._tokens = self._tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        # If we haven't consumed all tokens, there is trailing garbage
        if self._pos < len(self._tokens):
            raise ValueError(f"Invalid token or unexpected character: {self._tokens[self._pos]}")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of valid mathematical tokens.

        Args:
            expr: The raw input string.

        Returns:
            A list of strings representing tokens.

        Raises:
            ValueError: If the expression contains invalid characters or is empty.
        """
        # Remove all whitespace
        stripped = "".join(expr.split())
        if not stripped:
            return []

        # Regex pattern for: floats (1.2, .2, 2.), integers, or operators/parentheses
        pattern = re.compile(r'(\d+\.\d*|\.\d+|\d+|[+\-*/()])')
        tokens = pattern.findall(stripped)

        # Check if the reconstructed string from tokens matches the stripped input.
        # If not, it means there were characters that didn't match our pattern.
        if "".join(tokens) != stripped:
            raise ValueError("Invalid token found in expression")

        return tokens

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).

        Returns:
            The result of the addition/subtraction operations.
        """
        left = self._parse_term()

        while self._pos < len(self._tokens) and self._tokens[self._pos] in ('+', '-'):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_term()
            if op == '+':
                left += right
            else:
                left -= right
        return left

    def _parse_term(self) -> float:
        """
        Parses multiplication and division.

        Returns:
            The result of the multiplication/division operations.
        """
        left = self._parse_unary()

        while self._pos < len(self._tokens) and self._tokens[self._pos] in ('*', '/'):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_unary()
            if op == '*':
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_unary(self) -> float:
        """
        Parses unary operators (e.g., -5 or +5).

        Returns:
            The result of the unary operation.
        """
        if self._pos < len(self._tokens):
            if self._tokens[self._pos] == '-':
                self._pos += 1
                return -self._parse_unary()
            elif self._tokens[self._pos] == '+':
                self._pos += 1
                return self._parse_unary()
        
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parses numbers and parenthesized expressions (highest precedence).

        Returns:
            The numeric value of the primary unit.

        Raises:
            ValueError: If a number is expected but not found, or parentheses are mismatched.
        """
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression")

        token = self._tokens[self._pos]

        if token == '(':
            self._pos += 1
            result = self._parse_expression()
            if self._pos >= len(self._tokens) or self._tokens[self._pos] != ')':
                raise ValueError("Mismatched parentheses")
            self._pos += 1
            return result
        else:
            try:
                val = float(token)
                self._pos += 1
                return val
            except ValueError:
                raise ValueError(f"Invalid token: {token}")

# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    """Tests standard arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    assert evaluator.evaluate("5.5 + 4.5") == 10.0

def test_precedence():
    """Tests that operator precedence (PEMDAS) is respected."""
    evaluator = ExpressionEvaluator()
    # 10 - (2 * 3) + 4 = 10 - 6 + 4 = 8
    assert evaluator.evaluate("10 - 2 * 3 + 4") == 8.0
    # (2 * (3 + 4)) = 14
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0

def test_parentheses():
    """Tests grouping with parentheses."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    """Tests unary minus and nested unary operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 + -2") == 3.0
    assert evaluator.evaluate("-5 * -2") == 10.0
    assert evaluator.evaluate("--5") == -5.0

def test_error_cases():
    """Tests various error scenarios."""
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
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1 + a")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1.2.3")
        
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")