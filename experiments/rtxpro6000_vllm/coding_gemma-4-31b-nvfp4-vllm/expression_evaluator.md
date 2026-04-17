```python
import re
from typing import List, Tuple

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic,
    operator precedence, parentheses, and unary minus using a 
    recursive descent parser.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates a mathematical expression string.
        Returns the result as a float.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize the input string
        # Regex captures floats, integers, and operators/parentheses
        token_pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        self.tokens = re.findall(token_pattern, expr)
        
        # Validate that no invalid characters were skipped by the regex
        # by comparing the joined tokens to the stripped input
        cleaned_expr = "".join(expr.split())
        cleaned_tokens = "".join(self.tokens)
        if len(cleaned_expr) != len(cleaned_tokens):
            # This is a simple check; in a production system, a proper lexer 
            escapes would be used.
            raise ValueError("Expression contains invalid tokens")

        self.pos = 0
        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: {self.tokens[self.pos]}")
        
        return float(result)

    def _peek(self) -> str:
        """Returns the current token without consuming it."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ""

    def _consume(self) -> str:
        """Returns the current token and advances the position."""
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
        """Handles unary minus and plus."""
        if self._peek() == '-':
            self._consume() # consume '-'
            return -self._parse_unary()
        if self._peek() == '+':
            self._consume() # consume '+'
            return self._parse_unary()
        return self._parse_factor()

    def _parse_factor(self) -> float:
        """Handles parentheses and numbers (highest precedence)."""
        token = self._peek()

        if token == '(':
            self._consume() # consume '('
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            return result
        
        if token == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")

        if not token:
            raise ValueError("Unexpected end of expression")

        try:
            val = float(self._consume())
            return val
        except ValueError:
            raise ValueError(f"Invalid numeric token: {token}")

# --- Pytest Tests ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 - 2") == 13.0
    assert evaluator.evaluate("10 * 2 / 5") == 4.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_precedence():
    evaluator = ExpressionEvaluator()
    # 2 + 3 * 4 should be 14, not 20
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # 10 - 4 / 2 should be 8, not 3
    assert evaluator.evaluate("10 - 4 / 2") == 8.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    assert evaluator.evaluate("((1 + 1) * (1 + 1))") == 4.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("5 + (-2)") == 3.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("1 + 2)")
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid tokens"):
        evaluator.evaluate("2 @ 3")
```