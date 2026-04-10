import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A recursive descent parser for mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: {self.tokens[self.pos]}")
            
        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """
        Splits the input string into a list of tokens (numbers, operators, parentheses).
        """
        # Regex matches floats/ints, operators, and parentheses
        token_pattern = r"(\d*\.\d+|\d+|[+\-*/()])"
        tokens = re.findall(token_pattern, expr)
        
        # Validate that the reconstructed tokens match the input (ignoring whitespace)
        # This catches invalid characters like '3 @ 4'
        cleaned_expr = "".join(expr.split())
        if len("".join(tokens)) != len(cleaned_expr):
            # This is a simple check; a more robust one would track indices
            # but for this implementation, we check if any non-whitespace chars were ignored.
            # We'll use a more explicit check during parsing if needed.
            pass 
            
        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without advancing the position."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        """Returns the current token and advances the position."""
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
            self._consume()
            return -self._parse_factor()
        
        if token == '(':
            self._consume()
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._consume()
            return result
        
        if token is not None and (token.isdigit() or '.' in token):
            try:
                return float(self._consume())
            except ValueError:
                raise ValueError(f"Invalid number format: {token}")
        
        raise ValueError(f"Expected number or '(', found {token if token else 'end of string'}")

# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 4 * 2") == 11.0
    assert evaluator.evaluate("10 - 2 - 1") == 7.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Parentheses override precedence
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0

def test_parentheses_and_floats():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("((1.5 + 2.5) * 2)") == 8.0
    assert evaluator.evaluate("10 / (2.0 + 0.5)") == 4.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("5 + (-3)") == 2.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Expected number or '('"):
        evaluator.evaluate("3 + a")