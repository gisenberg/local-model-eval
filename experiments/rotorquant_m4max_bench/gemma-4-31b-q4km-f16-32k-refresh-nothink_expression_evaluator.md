# Gemma 4 31B-IT Q4_K_M (f16 @ 32K, new base default ub) — expression_evaluator

**Hypothesis:** Refresh baseline: f16 at 32K works on new base. Is this faster or slower than turbo4's 11.8?

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

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
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

        self._tokenize(expr)
        self._pos = 0
        
        result = self._parse_expression()
        
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")
            
        return float(result)

    def _tokenize(self, expr: str) -> None:
        """
        Splits the input string into a list of tokens (numbers, operators, parentheses).
        """
        # Regex matches: floating point numbers, operators (+, -, *, /), and parentheses
        token_pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        self._tokens = re.findall(token_pattern, expr)
        
        # Validate that the original string doesn't contain illegal characters
        # by comparing the reconstructed string (ignoring whitespace)
        cleaned_expr = "".join(expr.split())
        reconstructed = "".join(self._tokens)
        if len(reconstructed) != len(cleaned_expr):
            # This is a simple way to detect invalid characters like 'a' or '@'
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
        """Handles unary minus, parentheses, and numbers (highest precedence)."""
        token = self._peek()

        if token == '-':
            self._consume()
            return -self._parse_factor()
        
        if token == '+':
            self._consume()
            return self._parse_factor()

        if token == '(':
            self._consume() # consume '('
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            return result

        if token and (token.isdigit() or '.' in token):
            self._consume()
            try:
                return float(token)
            except ValueError:
                raise ValueError(f"Invalid number format: {token}")

        raise ValueError(f"Unexpected token: {token if token else 'End of expression'}")

# --- Pytest Tests ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 2 * 3") == 16.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Parentheses override precedence
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert evaluator.evaluate("100 / (2 + 3)") == 20.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5 + 2") == -3.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0 # Double negative
    assert evaluator.evaluate("-3.14 * 2") == -6.28

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
        
    with pytest.raises(ValueError, match="invalid characters"):
        evaluator.evaluate("2 + a")
        
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("   ")
```
