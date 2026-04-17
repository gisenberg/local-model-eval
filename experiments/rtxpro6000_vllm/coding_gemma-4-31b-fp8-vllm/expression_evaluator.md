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
        # Regex matches floating point numbers or single characters (+, -, *, /, (, ))
        token_pattern = r"(\d*\.\d+|\d+|[+\-*/()])"
        self._tokens = re.findall(token_pattern, expr)
        
        # Validate that we didn't miss any invalid characters
        # Reconstruct the string from tokens to check if it matches the original (ignoring whitespace)
        cleaned_expr = "".join(expr.split())
        if "".join(self._tokens) != cleaned_expr:
            # Find the first character that didn't match the pattern
            raise ValueError("Expression contains invalid characters")

    def _peek(self) -> str:
        """Returns the current token without advancing the position."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else ""

    def _consume(self) -> str:
        """Returns the current token and advances the position."""
        token = self._peek()
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Handles addition and subtraction (lowest precedence).
        Grammar: expression -> term (( '+' | '-' ) term)*
        """
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
        """
        Handles multiplication and division.
        Grammar: term -> factor (( '*' | '/' ) factor)*
        """
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
        """
        Handles unary minus, parentheses, and numbers (highest precedence).
        Grammar: factor -> '-' factor | '(' expression ')' | number
        """
        token = self._peek()

        if token == '-':
            self._consume()
            return -self._parse_factor()
        
        if token == '(':
            self._consume() # consume '('
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            return result
        
        if token == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")

        # Try to parse as a number
        try:
            self._consume()
            return float(token)
        except (ValueError, IndexError):
            raise ValueError(f"Expected number or '(' but found '{token}'")

# --- Pytest Tests ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 - 2") == 13.0
    assert evaluator.evaluate("10 * 2 / 4") == 5.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Division before subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((1 + 1) * (2 + 2))") == 8.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("5 + -3") == 2.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="invalid characters"):
        evaluator.evaluate("2 + 3abc")
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("   ")
```