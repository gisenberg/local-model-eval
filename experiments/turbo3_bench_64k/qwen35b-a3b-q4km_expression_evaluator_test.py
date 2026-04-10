import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    Supports +, -, *, /, unary minus, and parentheses.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates the given mathematical expression.

        Args:
            expr: The string expression to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                          has mismatched parentheses, or performs division by zero.
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        self._tokenize(expr)
        self._pos = 0
        
        try:
            result = self._parse_expression()
        except IndexError:
            raise ValueError("Unexpected end of expression")

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {_pos}: {self._tokens[self._pos]}")

        return float(result)

    def _tokenize(self, expr: str) -> None:
        """
        Converts the input string into a list of valid tokens.
        
        Raises:
            ValueError: If an invalid character/token is encountered.
        """
        # Regex: matches floats, integers, operators, or any single non-whitespace character
        # The second group captures invalid characters for error reporting
        pattern = r'(\d*\.\d+|\d+|[+\-*/()]|[^\s+\-*/()\d.])'
        raw_tokens = re.findall(pattern, expr)
        
        self._tokens = []
        for token in raw_tokens:
            # If the token is not a valid operator, number, or parenthesis, it's an error
            if re.match(r'^(\d*\.\d+|\d+|[+\-*/()])$', token):
                self._tokens.append(token)
            else:
                raise ValueError(f"Invalid token: {token}")
        
        if not self._tokens:
            raise ValueError("Empty expression")

    def _peek(self) -> Optional[str]:
        """Returns the current token without advancing the pointer."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> Optional[str]:
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
        """Handles unary minus."""
        if self._peek() == '-':
            self._consume()
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and parentheses (highest precedence)."""
        token = self._consume()

        if token == '(':
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            return result
        
        # Check if token is a valid number
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid number: {token}")

# --- Pytest Suite ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 2") == 5.0
    assert evaluator.evaluate("1.5 + 2.5") == 4.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Division before subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0
    # Complex precedence
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    assert evaluator.evaluate("((1 + 2) * 3)") == 9.0
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("1 + 2)")

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-5 + 2") == -3.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-3 * -2") == 6.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1 + abc")
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    # Unclosed parenthesis
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")