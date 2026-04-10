import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports +, -, *, /, 
    parentheses, unary minus, and floating point numbers.
    Uses a recursive descent parser.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string.

        Args:
            expr: The string expression to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or results in division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        try:
            result = self._parse_expression()
            # If we haven't consumed all tokens, the expression is invalid
            if self._pos < len(self._tokens):
                raise ValueError(f"Unexpected token at position {self._pos}")
            return float(result)
        except (IndexError, TypeError):
            raise ValueError("Invalid expression structure or mismatched parentheses.")

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the string into a list of valid tokens."""
        # Pattern matches floats/integers, operators, and parentheses
        token_pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        tokens = re.findall(token_pattern, expr)
        
        # Validation: Check if the reconstructed string matches the original 
        # (ignoring whitespace) to catch invalid characters.
        sanitized_expr = "".join(expr.split())
        sanitized_tokens = "".join(tokens)
        if len(sanitized_tokens) != len(sanitized_expr):
            raise ValueError("Expression contains invalid characters or tokens.")
            
        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Returns the current token and moves the pointer forward."""
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
                    raise ValueError("Division by zero.")
                node /= right
        return node

    def _parse_factor(self) -> float:
        """Handles unary minus and parentheses (highest precedence)."""
        token = self._peek()

        # Handle Unary Minus
        if token == '-':
            self._consume()
            return -self._parse_factor()
        
        # Handle Unary Plus (optional, but good for completeness)
        if token == '+':
            self._consume()
            return self._parse_factor()

        # Handle Parentheses
        if token == '(':
            self._consume()
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()
            return result

        # Handle Numbers
        try:
            return float(self._consume())
        except (TypeError, ValueError):
            raise ValueError(f"Unexpected token: {token}")

# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 * 2") == 20.0
    assert evaluator.evaluate("(10 + 5) * 2") == 30.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0

def test_unary_and_float():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_complex_nesting():
    evaluator = ExpressionEvaluator()
    # 10 + (2 * (3 + 4)) = 10 + 14 = 24
    assert evaluator.evaluate("10 + (2 * (3 + 4))") == 24.0
    # -(-5) = 5
    assert evaluator.evaluate("-(-5)") == 5.0

def test_errors():
    evaluator = ExpressionEvaluator()
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="invalid characters"):
        evaluator.evaluate("1 + a")
    
    # Empty expression
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("")

def test_precedence_edge_cases():
    evaluator = ExpressionEvaluator()
    # Ensure subtraction and division precedence
    # 10 - 6 / 2 = 10 - 3 = 7
    assert evaluator.evaluate("10 - 6 / 2") == 7.0
    # 10 - (6 / 2) = 7
    assert evaluator.evaluate("10 - (6 / 2)") == 7.0