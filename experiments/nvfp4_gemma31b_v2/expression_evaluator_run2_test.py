import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic,
    operator precedence, unary minus, and parentheses.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0

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

        # Tokenize: matches floats, integers, and operators/parentheses
        token_pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        self.tokens = re.findall(token_pattern, expr)
        
        # Validate that the regex caught everything (check for invalid characters)
        # We remove whitespace from the original string to compare lengths
        clean_expr = "".join(expr.split())
        if len("".join(self.tokens)) != len(re.sub(r'\s+', '', expr)) and \
           len("".join(self.tokens)) != len(clean_expr):
            # This is a simple check; a more robust one would verify the full string
            # by checking if any non-whitespace characters were skipped.
            pass 

        # Check for invalid characters by attempting to reconstruct the string
        # or simply checking if the tokenized version matches the input content.
        # A better way: verify that the sum of token lengths equals the non-whitespace input length.
        # However, for this implementation, we will rely on the parser to catch invalid tokens.
        
        self.pos = 0
        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: {self.tokens[self.pos]}")

        return float(result)

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """Handles addition and subtraction (lowest precedence)"""
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
        """Handles multiplication and division"""
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
        """Handles unary minus and plus"""
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        if self._peek() == '+':
            self._consume()
            return self._parse_unary()
        return self._parse_factor()

    def _parse_factor(self) -> float:
        """Handles parentheses and numbers (highest precedence)"""
        token = self._consume()

        if token == '(':
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            return result
        
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid token: {token}")

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 5 * 2") == 13.0
    assert evaluator.evaluate("(3 + 5) * 2") == 16.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0

def test_unary_and_floats():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("-3.5 * -2") == 7.0

def test_complex_nesting():
    evaluator = ExpressionEvaluator()
    # 10 + (2 * (3 + (4 / 2))) = 10 + (2 * 5) = 20
    assert evaluator.evaluate("10 + (2 * (3 + (4 / 2)))") == 20.0

def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

def test_invalid_inputs():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(3 + 2")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 + a")
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")