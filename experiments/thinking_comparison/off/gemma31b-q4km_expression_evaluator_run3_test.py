import re
from typing import List, Tuple

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic, 
    operator precedence, parentheses, and unary minus.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical string expression and returns the result as a float.
        
        Args:
            expr: The string expression to evaluate.
        Returns:
            The result of the evaluation.
        Raises:
            ValueError: If the expression is empty, contains invalid tokens, 
                        has mismatched parentheses, or results in division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize the input string using regex
        # Matches floats/ints, operators, and parentheses
        token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')
        self._tokens = token_pattern.findall(expr)
        
        # Validate that no invalid characters were skipped by the regex
        # (Comparing reconstructed string length to original stripped length)
        if len("".join(self._tokens)) != len(expr.replace(" ", "")):
            raise ValueError("Expression contains invalid tokens")

        self._pos = 0
        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError("Mismatched parentheses or trailing tokens")
            
        return float(result)

    def _peek(self) -> str:
        """Returns the current token without advancing the pointer."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else ""

    def _consume(self) -> str:
        """Returns the current token and advances the pointer."""
        token = self._peek()
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
        node = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '/':
                if right == 0:
                    raise ValueError("Division by zero")
                node /= right
            else:
                node *= right
        return node

    def _parse_factor(self) -> float:
        """Handles unary minus, parentheses, and numbers (highest precedence)."""
        token = self._peek()

        if token == '-':
            self._consume()
            return -self._parse_factor()
        
        if token == '(':
            self._consume() # consume '('
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            return result
        
        if token and (token[0].isdigit() or (token[0] == '.' and len(token) > 1)):
            return float(self._consume())
        
        raise ValueError(f"Unexpected token: {token if token else 'end of expression'}")

# --- Pytest Tests ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 2 * 3") == 16.0
    assert evaluator.evaluate("10.5 + 4.5") == 15.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Division before subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="invalid tokens"):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("   ")