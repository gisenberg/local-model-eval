import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser
    to evaluate arithmetic expressions supporting +, -, *, /, unary minus, 
    and parentheses.
    """

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates the given mathematical expression.
        
        Args:
            expr: The string expression to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: For division by zero, mismatched parentheses, 
                       invalid tokens, or empty expressions.
        """
        if not expr.strip():
            raise ValueError("Expression is empty")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        if not self.tokens:
            raise ValueError("Expression contains no valid tokens")

        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at end of expression: {self.tokens[self.pos]}")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the input string into a list of meaningful tokens."""
        # Regex to match numbers (including floats), operators, and parentheses
        token_pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        tokens = re.findall(token_pattern, expr)
        
        # Validate that we didn't miss any non-whitespace characters (invalid tokens)
        cleaned_expr = re.sub(r'\s+', '', expr)
        token_stream = "".join(tokens)
        if len(token_stream) != len(cleaned_expr):
            # Find the first character that wasn't matched by the regex
            for char in cleaned_expr:
                if char not in "+-*/()0123456789.":
                    raise ValueError(f"Invalid token detected: {char}")
        
        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        """Consumes and returns the current token."""
        token = self._peek()
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
        """Handles unary minus."""
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_factor()

    def _parse_factor(self) -> float:
        """Handles numbers and parentheses (highest precedence)."""
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._consume()  # consume ')'
            return result
        
        if token is None:
            raise ValueError("Unexpected end of expression")

        # Try to parse a number
        try:
            val = float(self._consume())
            return val
        except ValueError:
            raise ValueError(f"Invalid token or unexpected character: {token}")

# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 - 2") == 13.0
    assert evaluator.evaluate("10 / 2 * 3") == 15.0
    assert evaluator.evaluate("3.5 * 2") == 7.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Parentheses override precedence
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0

def test_parentheses_and_unary():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("2 * (-3)") == -6.0
    assert evaluator.evaluate("-(5 - (2 + 1))") == -2.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("10 @ 5")
    # Empty expression
    with pytest.raises(ValueError, match="Expression is empty"):
        evaluator.evaluate("   ")

def test_complex_expression():
    evaluator = ExpressionEvaluator()
    # Complex mix
    assert evaluator.evaluate("10 + 2 * (3 + 4 / 2) - -5") == 25.0