import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic,
    operator precedence, parentheses, and unary minus.
    """

    def __init__(self):
        # Token patterns: float/int, operators, and parentheses
        self.token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr (str): The expression to evaluate.
            
        Returns:
            float: The result of the evaluation.
            
        Raises:
            ValueError: For invalid tokens, mismatched parentheses, 
                        division by zero, or empty expressions.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize the input string
        tokens = self.token_pattern.findall(expr)
        
        # Validate that no characters were skipped (invalid tokens)
        # This ensures that characters like 'a' or '@' trigger a ValueError
        reconstructed = "".join(tokens)
        cleaned_expr = "".join(expr.split())
        if len(reconstructed) != len(cleaned_expr):
            raise ValueError("Expression contains invalid characters")

        self.tokens = tokens
        self.pos = 0

        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError("Unexpected tokens at end of expression")
            
        return float(result)

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        """Returns the current token and moves the pointer forward."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self.pos += 1
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
            val = float(self._consume())
            return val
        except (TypeError, ValueError):
            raise ValueError(f"Expected number or '(' but found {token}")

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_parentheses_and_unary():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("-3 + -5") == -8.0
    assert evaluator.evaluate("(-2 * -3)") == 6.0

def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

def test_mismatched_parentheses():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + 3)")

def test_invalid_inputs():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="invalid characters"):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 ++ 3") # Invalid syntax for this parser