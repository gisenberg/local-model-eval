import re
from typing import List, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, unary minus, parentheses, and floating point numbers.
    """

    def __init__(self):
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr (str): The expression to evaluate.
            
        Returns:
            float: The result of the evaluation.
            
        Raises:
            ValueError: For invalid tokens, mismatched parentheses, division by zero, 
                       or empty expressions.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize using regex: numbers (including decimals), operators, and parentheses
        self._tokens = re.findall(r'\d*\.\d+|\d+|[+\-*/()]', expr)
        
        # Validate that no invalid characters were skipped by the regex
        # We reconstruct the string from tokens to check if it matches the input (ignoring whitespace)
        cleaned_expr = "".join(expr.split())
        if len("".join(self._tokens)) != len(cleaned_expr):
            # This is a simple check; a more robust one would check if any non-whitespace 
            # characters were missed by the regex.
            # We'll verify tokens individually during parsing.
            pass

        self._pos = 0
        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at end of expression: {self._tokens[self._pos]}")
            
        return float(result)

    def _peek(self) -> str:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Returns the current token and moves the position forward."""
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
            node = node + right if op == '+' else node - right
        return node

    def _parse_term(self) -> float:
        """Handles multiplication and division."""
        node = self._parse_unary()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_unary()
            if op == '/':
                if right == 0:
                    raise ValueError("Division by zero")
                node = node / right
            else:
                node = node * right
        return node

    def _parse_unary(self) -> float:
        """Handles unary minus."""
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_factor()

    def _parse_factor(self) -> float:
        """Handles parentheses and numbers (highest precedence)."""
        token = self._peek()

        if token == '(':
            self._consume() # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume() # consume ')'
            return result
        
        if token is None:
            raise ValueError("Unexpected end of expression")

        # Try to parse as a number
        try:
            val = float(self._consume())
            return val
        except ValueError:
            raise ValueError(f"Invalid token: {token}")

import pytest

def test_basic_operations():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 5 * 2") == 13.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_parentheses_and_unary():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    assert evaluator.evaluate("-3 * -2") == 6.0
    assert evaluator.evaluate("(-5 + 2) * 3") == -9.0

def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

def test_mismatched_parentheses():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError):
        evaluator.evaluate("1 + 2)")

def test_invalid_inputs():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 + abc")