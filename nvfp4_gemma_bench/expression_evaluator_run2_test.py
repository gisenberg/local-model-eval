import re
, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic,
    operator precedence, parentheses, and unary operators.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr (str): The expression to evaluate.
            
        Returns:
            float: The result of the evaluation.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                         has mismatched parentheses, or division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize using regex: numbers (including floats), operators, and parentheses
        self.tokens = re.findall(r'\d*\.\d+|\d+|[+\-*/()]', expr)
        
        # Check for invalid characters by comparing reconstructed string
        # This ensures we don't ignore characters like '?' or 'a'
        reconstructed = "".join(self.tokens)
        if len(reconstructed) != len(expr.replace(" ", "")):
            raise ValueError("Expression contains invalid tokens")

        self.pos = 0
        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError("Mismatched parentheses or trailing tokens")
            
        return float(result)

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
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
        result = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Handles multiplication and division."""
        result = self._parse_unary()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_unary()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_unary(self) -> float:
        """Handles unary minus and plus."""
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        if self._peek() == '+':
            self._consume()
            return self._parse_unary()
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

        # Try to parse as a float
        try:
            val = float(self._consume())
            return val
        except ValueError:
            raise ValueError(f"Invalid token: {token}")

import pytest


def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 5 * 2") == 13.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_parentheses_and_unary():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-(2 + 3) * 2") == -10.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("-5 + -2") == -7.0
    assert evaluator.evaluate("(-3.5 + 1.5)") == -2.0

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
    with pytest.raises(ValueError, match="invalid tokens"):
        evaluator.evaluate("3 + a")
    with pytest.raises(ValueError):
        evaluator.evaluate("++")