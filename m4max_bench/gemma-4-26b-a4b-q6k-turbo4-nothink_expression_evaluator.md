# Gemma 4 26B-A4B Q6_K (turbo4 KV) — expression_evaluator

## Output

```python
import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser
    to handle arithmetic operations with correct precedence.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates a mathematical expression string.

        Args:
            expr: The string expression to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                         has mismatched parentheses, or results in division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError("No valid tokens found in expression.")

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at end of expression: '{self._tokens[self._pos]}'")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of meaningful tokens using regex.
        """
        # Pattern matches: floats/integers, operators (+, -, *, /), or parentheses
        token_pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        tokens = re.findall(token_pattern, expr)
        
        # Validation: Check if the reconstructed string from tokens matches the non-whitespace input
        # This is a simple way to detect invalid characters like 'abc' or '@'
        cleaned_expr = re.sub(r'\s+', '', expr)
        reconstructed = "".join(tokens)
        
        # Note: This check is a bit loose because regex findall skips non-matches.
        # A more robust way is to check if the length of characters consumed matches the input.
        # For this implementation, we'll check if any character in the original string 
        # (ignoring whitespace) is not part of our token list.
        stripped_expr = "".join(expr.split())
        if len(reconstructed) != len(stripped_expr):
            # Find the first invalid character for a better error message
            raise ValueError(f"Invalid characters detected in expression: '{expr}'")
            
        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without advancing the pointer."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Returns the current token and advances the pointer."""
        token = self._peek()
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Handles addition and subtraction (lowest precedence).
        Grammar: expression -> term { (+|-) term }
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
        Grammar: term -> factor { (*|/) factor }
        """
        node = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero is not allowed.")
                node /= right
        return node

    def _parse_factor(self) -> float:
        """
        Handles unary minus, parentheses, and numbers.
        Grammar: factor -> '-' factor | '(' expression ')' | number
        """
        token = self._peek()

        if token == '-':
            self._consume()
            return -self._parse_factor()
        
        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._consume()  # consume ')'
            return result

        if token is None:
            raise ValueError("Unexpected end of expression.")

        try:
            # Attempt to parse the token as a float
            val = float(self._consume())
            return val
        except ValueError:
            raise ValueError(f"Invalid token: '{token}'")

# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 - 2") == 13.0
    assert evaluator.evaluate("10 * 2 / 5") == 4.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Parentheses override precedence
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # Division and multiplication same level
    assert evaluator.evaluate("10 / 2 * 5") == 25.0

def test_parentheses_and_unary():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("-3 + -5") == -8.0
    assert evaluator.evaluate("10 / (2 + (3 * 1))") == 2.0
    assert evaluator.evaluate("-(5 - 10)") == 5.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("1 + 2)")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid characters"):
        evaluator.evaluate("10 + abc")

    # Empty expression
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")

def test_complex_expression():
    evaluator = ExpressionEvaluator()
    # Testing a mix of everything
    expr = "-(10 + 2.5) * (4 / -2) + 7"
    # -12.5 * -2 + 7 = 25 + 7 = 32
    assert evaluator.evaluate(expr) == 32.0
```
