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
            ValueError: If there are mismatched parentheses, division by zero,
                        invalid tokens, or an empty expression.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        try:
            result = self._parse_expression()
            # If we haven't consumed all tokens, there's a syntax error
            if self._pos < len(self._tokens):
                raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")
            return float(result)
        except (IndexError, TypeError):
            raise ValueError("Invalid expression structure")

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the string into a list of valid tokens."""
        # Pattern matches floats, integers, operators, and parentheses
        token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')
        tokens = token_pattern.findall(expr)
        
        # Validation: Check if the reconstructed string matches the input (ignoring whitespace)
        # This ensures no illegal characters (like letters or symbols) are present.
        stripped_input = "".join(expr.split())
        reconstructed = "".join(tokens)
        
        # A simple way to check for invalid characters: 
        # If the length of tokens doesn't account for all non-whitespace chars, it's invalid.
        if len(reconstructed) != len(stripped_input):
            raise ValueError("Expression contains invalid tokens or characters")
            
        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self, expected: Optional[str] = None) -> Optional[str]:
        """Returns the current token and moves the pointer forward."""
        token = self._peek()
        if expected and token != expected:
            raise ValueError(f"Expected {expected} but found {token}")
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
        """Handles unary minus and parentheses (highest precedence)."""
        token = self._peek()

        # Handle Unary Minus
        if token == '-':
            self._consume('-')
            return -self._parse_factor()
        
        # Handle Unary Plus (optional, for completeness)
        if token == '+':
            self._consume('+')
            return self._parse_factor()

        # Handle Parentheses
        if token == '(':
            self._consume('(')
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume(')')
            return result

        # Handle Numbers
        try:
            return float(self._consume())
        except (TypeError, ValueError):
            raise ValueError("Unexpected token or invalid number")

# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 * 2") == 20.0
    assert evaluator.evaluate("(10 + 5) * 2") == 30.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0

def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("10 / 3") == pytest.approx(3.3333333333)

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5 + 3") == -2.0
    assert evaluator.evaluate("-(2 + 3) * 2") == -10.0
    assert evaluator.evaluate("5 + -2") == 3.0  # Note: This relies on factor parsing -2

def test_errors():
    evaluator = ExpressionEvaluator()
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="invalid tokens"):
        evaluator.evaluate("1 + a")
    
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

def test_complex_precedence():
    evaluator = ExpressionEvaluator()
    # 2 + (3 * (4 - 2)) = 2 + (3 * 2) = 8
    assert evaluator.evaluate("2 + (3 * (4 - 2))") == 8.0
    # -3^2 logic (Note: our parser treats -3 * 2 as -(3*2) if written -3*2, 
    # but standard recursive descent handles unary higher than binary)
    assert evaluator.evaluate("-3 * 2") == -6.0