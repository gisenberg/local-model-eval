import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser
    to handle arithmetic operations, precedence, and unary operators.
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
                         has mismatched parentheses, or involves division by zero.
        """
        if not expr.strip():
            raise ValueError("Expression is empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError("Expression contains no valid tokens")

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at end of expression: '{self._tokens[self._pos]}'")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of meaningful tokens using regex.
        """
        # Pattern matches numbers (including floats), operators, and parentheses
        token_pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        tokens = re.findall(token_pattern, expr)
        
        # Validate that the reconstructed string from tokens matches the non-whitespace input
        # This ensures no invalid characters were skipped by the regex
        cleaned_expr = re.sub(r'\s+', '', expr)
        reconstructed = "".join(tokens)
        
        # Check if we missed any characters (invalid tokens)
        # We use a simplified check: if the length of tokens doesn't account for all non-space chars
        # we check if the regex missed anything.
        all_chars_len = len(re.sub(r'\s+', '', expr))
        # This is a heuristic; a more robust way is to check if any non-whitespace char is not in tokens
        if len(re.sub(r'[\d\.\+\-\*/\(\)]', '', expr)) > 0:
            # Find the first invalid character
            invalid_match = re.search(r'[^\d\.\+\-\*/\(\)\s]', expr)
            if invalid_match:
                raise ValueError(f"Invalid token detected: '{invalid_match.group()}'")

        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without advancing the pointer."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Returns the current token and advances the pointer."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Handles addition and subtraction (lowest precedence).
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
        """
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
        """
        Handles unary minus.
        """
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_factor()

    def _parse_factor(self) -> float:
        """
        Handles parentheses and numeric literals (highest precedence).
        """
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()  # consume ')'
            return result
        
        if token is None:
            raise ValueError("Unexpected end of expression")

        # Try to parse a number
        try:
            # We must ensure the token is a valid number and not just a stray operator
            if token in ('+', '-', '*', '/', ')', '('):
                raise ValueError(f"Unexpected token: '{token}'")
            
            val = float(self._consume())
            return val
        except ValueError:
            raise ValueError(f"Invalid number or token: '{token}'")

# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 - 2") == 13.0
    assert evaluator.evaluate("10 * 5 / 2") == 25.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Parentheses override precedence
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0

def test_parentheses_and_unary():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("-3 * (4 - 6)") == 6.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1 + @")
    # Empty expression
    with pytest.raises(ValueError, match="Expression is empty"):
        evaluator.evaluate("   ")

def test_complex_expression():
    evaluator = ExpressionEvaluator()
    # A mix of everything
    assert evaluator.evaluate("-(5 + 5) * 2 + 10 / 0.5") == 0.0 # -10 * 2 + 20 = 0