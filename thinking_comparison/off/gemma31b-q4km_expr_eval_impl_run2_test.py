import re
from typing import List, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
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
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """
        Splits the input string into a list of tokens using regular expressions.
        """
        # Pattern matches floats/ints, operators, and parentheses
        token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')
        tokens = token_pattern.findall(expr)
        
        # Validate that no invalid characters were skipped by the regex
        reconstructed = "".join(tokens)
        stripped_expr = expr.replace(" ", "")
        
        # This is a simple check to ensure we didn't ignore invalid characters
        # by comparing the length of tokens vs the original string (minus whitespace)
        # However, a more robust way is to check if the regex matched everything.
        # For simplicity in this implementation, we check if the joined tokens 
        # match the stripped input.
        if len(reconstructed) != len(stripped_expr):
            # Find the first character that didn't match
            for char in stripped_expr:
                if char not in "0123456789.+-*/()":
                    raise ValueError(f"Invalid character found in expression: {char}")

        return tokens

    def _peek(self) -> str:
        """Returns the current token without advancing the position."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else ""

    def _consume(self) -> str:
        """Returns the current token and advances the position."""
        token = self._peek()
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
            self._consume()
            return -self._parse_factor()
        
        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            return result

        if token == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")

        try:
            self._consume()
            return float(token)
        except (ValueError, TypeError):
            raise ValueError(f"Expected number or '(' but found '{token}'")

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