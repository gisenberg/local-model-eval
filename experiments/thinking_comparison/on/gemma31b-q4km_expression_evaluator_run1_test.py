import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic, 
    operator precedence, parentheses, and unary minus using a 
    recursive descent parser.
    """

    def __init__(self):
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The result of the evaluation.
            
        Raises:
            ValueError: For invalid tokens, mismatched parentheses, 
                        division by zero, or empty expressions.
        """
        # Remove whitespace and tokenize
        expr = expr.replace(" ", "")
        if not expr:
            raise ValueError("Expression cannot be empty")

        # Tokenize using regex: matches floats/ints or single operator characters
        self._tokens = re.findall(r"\d*\.\d+|\d+|[+\-*/()]", expr)
        
        # Validate that the regex captured everything (detect invalid tokens)
        if "".join(self._tokens) != expr:
            raise ValueError("Expression contains invalid characters")

        self._pos = 0
        result = self._parse_expression()

        # If we finished parsing but tokens remain, there is a syntax error (e.g., mismatched parens)
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")

        return float(result)

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Returns the current token and advances the position."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Handles addition and subtraction (lowest precedence)."""
        node = self._parse_term()

        while self._peek() in ("+", "-"):
            op = self._consume()
            right = self._parse_term()
            if op == "+":
                node += right
            else:
                node -= right
        return node

    def _parse_term(self) -> float:
        """Handles multiplication and division."""
        node = self._parse_unary()

        while self._peek() in ("*", "/"):
            op = self._consume()
            right = self._parse_unary()
            if op == "*":
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                node /= right
        return node

    def _parse_unary(self) -> float:
        """Handles unary minus."""
        if self._peek() == "-":
            self._consume()  # consume '-'
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and grouped expressions (highest precedence)."""
        token = self._peek()

        if token == "(":
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._consume() != ")":
                raise ValueError("Mismatched parentheses: missing closing ')'")
            return result
        
        # Attempt to parse as a number
        try:
            self._consume()
            return float(token)
        except (TypeError, ValueError):
            raise ValueError(f"Expected number or '(', found {token}")

# ==========================================
# Pytest Tests
# ==========================================
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 - 3") == 5.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0

def test_precedence_and_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5 + 2") == -3.0
    assert evaluator.evaluate("5 + (-2)") == 3.0
    assert evaluator.evaluate("- (2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0  # Double negative

def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("1 / 4") == 0.25

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="invalid characters"):
        evaluator.evaluate("2 + a")
        
    # Empty expression
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("")

if __name__ == "__main__":
    # Simple manual check if pytest is not run
    ev = ExpressionEvaluator()
    print(f"Result of '-(2 + 3) * 4.5': {ev.evaluate('-(2 + 3) * 4.5')}") # Expected -22.5