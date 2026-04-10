import re
from typing import List

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports addition, subtraction, 
    multiplication, division, unary minus, and parentheses using a 
    recursive descent parser.
    """

    def __init__(self) -> None:
        """Initializes the evaluator with empty state."""
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string.

        Args:
            expr: The mathematical expression to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens, 
                        has mismatched parentheses, or involves division by zero.
        """
        self._tokens = self._tokenize(expr)
        if not self._tokens:
            raise ValueError("Empty expression")
        
        self._pos = 0
        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token: {self._tokens[self._pos]}")
        
        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of valid mathematical tokens.

        Args:
            expr: The raw expression string.

        Returns:
            A list of strings representing tokens.

        Raises:
            ValueError: If invalid characters are found in the expression.
        """
        # Remove all whitespace
        stripped = "".join(expr.split())
        if not stripped:
            return []

        tokens: List[str] = []
        last_end = 0
        # Pattern matches floats (e.g., 3.14, .5) or integers, or single-char operators/parens
        pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')

        for match in pattern.finditer(stripped):
            # If there is a gap between matches, it means there's an invalid character
            if match.start() != last_end:
                raise ValueError(f"Invalid token at position {last_end}")
            
            tokens.append(match.group())
            last_end = match.end()

        # If the last match didn't reach the end, there's a trailing invalid token
        if last_end != len(stripped):
            raise ValueError(f"Invalid token at position {last_end}")

        return tokens

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).

        Returns:
            The result of the addition/subtraction operations.
        """
        result = self._parse_term()
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ('+', '-'):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """
        Parses multiplication and division.

        Returns:
            The result of the multiplication/division operations.

        Raises:
            ValueError: If division by zero is attempted.
        """
        result = self._parse_unary()
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ('*', '/'):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_unary()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_unary(self) -> float:
        """
        Parses unary plus and unary minus.

        Returns:
            The result of the unary operation.
        """
        if self._pos < len(self._tokens):
            if self._tokens[self._pos] == '+':
                self._pos += 1
                return self._parse_unary()
            if self._tokens[self._pos] == '-':
                self._pos += 1
                return -self._parse_unary()
        
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parses numbers and parenthesized expressions (highest precedence).

        Returns:
            The numeric value of the primary component.

        Raises:
            ValueError: If parentheses are mismatched or a token is not a valid number.
        """
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression")

        token = self._tokens[self._pos]

        if token == '(':
            self._pos += 1
            result = self._parse_expression()
            if self._pos >= len(self._tokens) or self._tokens[self._pos] != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._pos += 1
            return result

        try:
            # Attempt to convert the token to a float
            val = float(token)
            self._pos += 1
            return val
        except ValueError:
            raise ValueError(f"Invalid token: {token}")

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