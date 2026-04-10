# Gemma 4 31B-IT Q8_0 — KV f16 (baseline) — Expression Evaluator

## Output

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
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates a mathematical expression string.
        Returns the result as a float.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        self._tokenize(expr)
        self._pos = 0
        
        result = self._parse_expression()
        
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at end of expression: {self._tokens[self._pos]}")
            
        return float(result)

    def _tokenize(self, expr: str) -> None:
        """
        Converts the input string into a list of tokens (numbers, operators, parentheses).
        """
        # Regex matches: floating point numbers, or single characters (+, -, *, /, (, ))
        token_pattern = r"(\d*\.\d+|\d+|[+\-*/()])"
        self._tokens = re.findall(token_pattern, expr)
        
        # Validate that we didn't miss any invalid characters
        # Reconstruct the string from tokens to check for gaps
        cleaned_expr = "".join(re.split(r'\s+', expr))
        if "".join(self._tokens) != cleaned_expr:
            # Find the first character that wasn't matched by the regex
            # This is a simplified check for invalid tokens
            raise ValueError("Expression contains invalid characters")

    def _peek(self) -> str:
        """Returns the current token without consuming it."""
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
        """Handles unary minus (e.g., -5 or -(2+3))."""
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        elif self._peek() == '+':
            self._consume()
            return self._parse_unary()
        return self._parse_factor()

    def _parse_factor(self) -> float:
        """Handles numbers and parentheses (highest precedence)."""
        token = self._peek()

        if token == '(':
            self._consume() # consume '('
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            return result
        
        if token == '':
            raise ValueError("Unexpected end of expression")

        try:
            self._consume()
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid number token: {token}")

# --- Pytest Tests ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 - 2") == 13.0
    assert evaluator.evaluate("10 * 2 / 4") == 5.0
    assert evaluator.evaluate("3.14 + 2.86") == 6.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Division before subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    # Force addition first
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # Nested parentheses
    assert evaluator.evaluate("2 * (3 + (4 / 2))") == 10.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5 + 2") == -3.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-3.14 * 2") == -6.28

def test_error_cases():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="invalid characters"):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("   ")
```
