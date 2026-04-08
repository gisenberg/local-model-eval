import re
from typing import List

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic, 
    parentheses, unary operators, and floating point numbers using 
    a recursive descent parser.
    """

    def __init__(self):
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: The expression string to evaluate.
            
        Returns:
            The floating point result of the expression.
            
        Raises:
            ValueError: For invalid tokens, mismatched parentheses, 
                       division by zero, or empty expressions.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0
        
        result = self._parse_expression()
        
        # If we haven't consumed all tokens, there is a syntax error (e.g., "2 + 2 3")
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")
            
        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """
        Splits the input string into a list of tokens (numbers, operators, parentheses).
        """
        # Regex matches: floating point numbers, integers, or single characters (+, -, *, /, (, ))
        token_pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        tokens = re.findall(token_pattern, expr)
        
        # Validation: Check if the joined tokens match the stripped input to detect invalid characters
        # We remove whitespace from the original expr to compare lengths/content
        cleaned_expr = "".join(expr.split())
        if "".join(tokens) != cleaned_expr:
            # Find the first character that wasn't matched by the regex
            # This is a simplified check to identify invalid tokens
            raise ValueError("Expression contains invalid characters")
            
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
        """Handles unary minus (e.g., -5, -(2+2))."""
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and parentheses (highest precedence)."""
        token = self._consume()
        
        if not token:
            raise ValueError("Unexpected end of expression")
            
        if token == '(':
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            return result
        
        if token == ')':
            raise ValueError("Mismatched parentheses: unexpected closing ')'")
            
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid numeric token: {token}")

# ==========================================
# Pytest Tests
# ==========================================
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 * 2") == 20.0
    assert evaluator.evaluate("10 - 2 - 1") == 7.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0

def test_precedence_and_parentheses():
    evaluator = ExpressionEvaluator()
    # Parentheses should override standard precedence
    assert evaluator.evaluate("(10 + 5) * 2") == 30.0
    # Nested parentheses
    assert evaluator.evaluate("2 * (3 + (4 / 2))") == 10.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0  # Double negative

def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("1.5 / 0.5") == 3.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("1 + 2)")
        
    with pytest.raises(ValueError, match="invalid characters"):
        evaluator.evaluate("2 + a")
        
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("   ")