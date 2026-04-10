import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic, 
    operator precedence, parentheses, and unary minus using a recursive descent parser.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: The expression string to evaluate.
            
        Returns:
            The result of the evaluation.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens, 
                        has mismatched parentheses, or involves division by zero.
        """
        # Remove whitespace and check for empty expression
        expr = expr.replace(" ", "")
        if not expr:
            raise ValueError("Expression cannot be empty")

        # Tokenize the expression
        # This regex captures floating point numbers, operators, and parentheses
        self._tokens = re.findall(r"\d*\.\d+|\d+|[+\-*/()]", expr)
        
        # Validate that the tokenization covered the entire string (detect invalid characters)
        if "".join(self._tokens) != expr:
            # Find the first character that didn't match a token for a descriptive error
            # This is a simplified check; in a production lexer, we'd track indices.
            raise ValueError("Expression contains invalid tokens")

        self._pos = 0
        result = self._parse_expression()

        # If we finished parsing but there are tokens left, it's a syntax error (e.g., "2 + 2 3")
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")

        return float(result)

    def _current_token(self) -> Optional[str]:
        """Returns the token at the current position without advancing."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self) -> None:
        """Advances the current position to the next token."""
        self._pos += 1

    def _parse_expression(self) -> float:
        """Handles addition and subtraction (lowest precedence)."""
        node = self._parse_term()

        while self._current_token() in ('+', '-'):
            op = self._current_token()
            self._advance()
            right = self._parse_term()
            if op == '+':
                node += right
            else:
                node -= right
        return node

    def _parse_term(self) -> float:
        """Handles multiplication and division."""
        node = self._parse_unary()

        while self._current_token() in ('*', '/'):
            op = self._current_token()
            self._advance()
            right = self._parse_unary()
            if op == '*':
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                node /= right
        return node

    def _parse_unary(self) -> float:
        """Handles unary minus (e.g., -5 or -(2+3))."""
        if self._current_token() == '-':
            self._advance()
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and grouped expressions in parentheses."""
        token = self._current_token()

        if token is None:
            raise ValueError("Unexpected end of expression")

        if token == '(':
            self._advance()
            result = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._advance()
            return result
        
        # Attempt to parse as a number
        try:
            self._advance()
            return float(token)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid token: {token}")

# ==========================================
# Pytest Tests
# ==========================================
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 4 * 2") == 11.0
    assert evaluator.evaluate("10 - 2 - 3") == 5.0
    assert evaluator.evaluate("10 / 4") == 2.5

def test_precedence_and_parentheses():
    evaluator = ExpressionEvaluator()
    # Parentheses should override default precedence
    assert evaluator.evaluate("(3 + 4) * 2") == 14.0
    # Nested parentheses
    assert evaluator.evaluate("2 * (3 + (4 / 2))") == 10.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0  # Double negation

def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("0.5 + 0.25") == 0.75

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
        
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
        
    # Invalid tokens
    with pytest.raises(ValueError, match="invalid tokens"):
        evaluator.evaluate("3 @ 4")
        
    # Empty expression
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")

if __name__ == "__main__":
    # To run tests manually without pytest CLI:
    pytest.main([__file__])