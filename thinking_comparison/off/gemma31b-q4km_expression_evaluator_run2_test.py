import re
from typing import List, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser that evaluates mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
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
                       has mismatched parentheses, or results in division by zero.
        """
        if not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize the input string
        # Regex matches: floating point numbers, operators, and parentheses
        self._tokens = re.findall(r'\d*\.\d+|\d+|[+\-*/()]', expr)
        
        # Validate that no invalid characters were skipped by the regex
        cleaned_expr = "".join(re.findall(r'\d*\.\d+|\d+|[+\-*/()]', expr))
        if len(cleaned_expr) != len(expr.replace(" ", "")):
            raise ValueError("Expression contains invalid tokens")

        self._pos = 0
        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")
            
        return float(result)

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
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            return result

        if token and (token[0].isdigit() or (len(token) > 1 and token[1].isdigit())):
            return float(self._consume())

        raise ValueError(f"Expected number or '(' but found '{token}'")

# --- Pytest Tests ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 4 * 2") == 11.0
    assert evaluator.evaluate("10 - 2 - 3") == 5.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Parentheses override precedence
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0

def test_parentheses_and_floats():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * (2 + 1.5)") == 10.99
    assert evaluator.evaluate("((10 + 2) * 2) / 3") == 8.0

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
        evaluator.evaluate("3 + a")
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("   ")