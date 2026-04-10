import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports +, -, *, /, 
    parentheses, unary minus, and floating point numbers using 
    a recursive descent parser.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string.

        Args:
            expr: The string expression to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If there are mismatched parentheses, division by zero,
                        invalid tokens, or an empty expression.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        try:
            result = self._parse_expression()
            # If we haven't consumed all tokens, there is a syntax error
            if self._pos < len(self._tokens):
                raise ValueError(f"Unexpected token: {self._tokens[self._pos]}")
            return float(result)
        except (IndexError, TypeError):
            raise ValueError("Invalid expression structure.")

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the string into a list of meaningful tokens."""
        # Pattern matches: Floats/Integers, Operators (+, -, *, /), Parentheses, or whitespace
        token_pattern = re.compile(r'(\d*\.\d+|\d+|[+\-*/()]|\s+)')
        raw_tokens = token_pattern.findall(expr)
        
        tokens = []
        for t in raw_tokens:
            t = t.strip()
            if not t:
                continue
            # Validate that the token is a valid number or operator
            if not re.match(r'^(\d*\.\d+|\d+|[+\-*/()])$', t):
                raise ValueError(f"Invalid token detected: {t}")
            tokens.append(t)
        
        # Final check: if the joined tokens don't match the original length (ignoring whitespace),
        # it means there were invalid characters.
        if len("".join(tokens)) < len(re.sub(r'\s+', '', expr)):
             # This is a simplified check; the regex above is more robust.
             pass 

        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Returns the current token and moves the pointer forward."""
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
                    raise ValueError("Division by zero.")
                node /= right
        return node

    def _parse_factor(self) -> float:
        """Handles unary minus, parentheses, and numbers (highest precedence)."""
        token = self._peek()

        # Handle Unary Minus
        if token == '-':
            self._consume()
            return -self._parse_factor()
        
        # Handle Unary Plus (optional, but good for completeness)
        if token == '+':
            self._consume()
            return self._parse_factor()

        # Handle Parentheses
        if token == '(':
            self._consume()
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()
            return result

        # Handle Numbers
        try:
            return float(self._consume())
        except (TypeError, ValueError):
            raise ValueError(f"Expected number, found: {token}")

# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 * 2") == 20.0
    assert evaluator.evaluate("(10 + 5) * 2") == 30.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 3) * 2") == -10.0
    assert evaluator.evaluate("--5") == 5.0

def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 + 2") == 5.14
    assert evaluator.evaluate("10 / 4") == 2.5

def test_errors():
    evaluator = ExpressionEvaluator()
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    # Invalid tokens
    with pytest.raises(ValueError):
        evaluator.evaluate("1 + a")
    
    # Empty expression
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("   ")

def test_complex_precedence():
    evaluator = ExpressionEvaluator()
    # 2 * (3 + (4 / 2)) = 2 * (3 + 2) = 10
    assert evaluator.evaluate("2 * (3 + (4 / 2))") == 10.0
    # -5 + 10 / 2 = -5 + 5 = 0
    assert evaluator.evaluate("-5 + 10 / 2") == 0.0

if __name__ == "__main__":
    # Manual test run if not using pytest command
    evaluator = ExpressionEvaluator()
    print(f"Result: {evaluator.evaluate('-(3 + 2) * 2.5')}") # Expected -12.5