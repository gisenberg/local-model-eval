import re
from typing import List, Optional


class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser
    to handle arithmetic operations, operator precedence, parentheses, and unary minus.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.

        Args:
            expr: The mathematical expression to evaluate.

        Returns:
            The result of the evaluation.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or results in division by zero.
        """
        self._tokens = self._tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        # If we haven't consumed all tokens, the expression is syntactically incorrect (e.g., "1 + 2 3")
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected extra tokens at position {self._pos}")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of tokens using regular expressions.

        Args:
            expr: The raw expression string.

        Returns:
            A list of string tokens.

        Raises:
            ValueError: If invalid characters are found in the expression.
        """
        # Pattern matches floats (e.g., 3.14, .5, 3.), integers, or single-character operators/parens
        pattern = r'(\d+\.\d*|\.\d+|\d+|[+\-*/()])'
        tokens = re.findall(pattern, expr)

        # Validate that the tokens reconstructed (ignoring whitespace) match the original input
        # This ensures that characters like 'a', '$', etc., trigger a ValueError.
        reconstructed = "".join(tokens)
        original_stripped = re.sub(r'\s+', '', expr)

        if reconstructed != original_stripped:
            raise ValueError("Invalid tokens in expression")

        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> Optional[str]:
        """Returns the current token and advances the pointer."""
        token = self._peek()
        if token is not None:
            self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Handles addition and subtraction (lowest precedence).
        Grammar: expression -> term { ('+' | '-') term }
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
        Grammar: term -> factor { ('*' | '/') factor }
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
        Handles unary minus.
        Grammar: factor -> '-' factor | primary
        """
        if self._peek() == '-':
            self._consume()
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Handles numbers and parenthesized expressions (highest precedence).
        Grammar: primary -> number | '(' expression ')'
        """
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            return result

        token = self._consume()
        if token is None:
            raise ValueError("Unexpected end of expression")

        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid token: {token}")


# --- Pytest Tests ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # (1 + 2) * 3 = 9
    assert evaluator.evaluate("(1 + 2) * 3") == 9.0
    # 10 / (2 + 3) = 2
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 + -2") == 3.0
    assert evaluator.evaluate("--5") == 5.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid tokens"):
        evaluator.evaluate("1 + a")
    
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("10 / 4") == 2.5