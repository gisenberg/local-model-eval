import re
from typing import List, Optional


class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports addition, subtraction,
    multiplication, division, unary minus, and parentheses using a 
    recursive descent parser.
    """

    def __init__(self) -> None:
        """Initializes the evaluator with empty state."""
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates a mathematical expression.

        Args:
            expr: A string representing the mathematical expression.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or results in division by zero.
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token after expression: {self._tokens[self._pos]}")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of valid tokens.

        Args:
            expr: The raw input string.

        Returns:
            A list of strings representing tokens.

        Raises:
            ValueError: If the expression contains characters that are not valid tokens.
        """
        # Regex pattern for numbers (including decimals), operators, and parentheses
        pattern = r'\d+\.\d*|\.\d+|\d+|[+\-*/()]'
        tokens = re.findall(pattern, expr)

        # Check if the reconstructed tokens match the stripped input to detect invalid characters
        stripped_input = "".join(expr.split())
        if "".join(tokens) != stripped_input:
            raise ValueError("Invalid tokens in expression")

        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without advancing the pointer."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self, expected: Optional[str] = None) -> str:
        """
        Consumes the current token and advances the pointer.

        Args:
            expected: An optional token that must match the current token.

        Returns:
            The consumed token.

        Raises:
            ValueError: If the end of tokens is reached unexpectedly or the token doesn't match.
        """
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        if expected and token != expected:
            raise ValueError(f"Expected '{expected}', found '{token}'")
        
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).

        Returns:
            The result of the expression.
        """
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
        """
        Parses multiplication and division.

        Returns:
            The result of the term.
        """
        result = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """
        Parses unary minus.

        Returns:
            The result of the factor.
        """
        if self._peek() == '-':
            self._consume('-')
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parses numbers and parenthesized expressions (highest precedence).

        Returns:
            The result of the primary component.
        """
        token = self._peek()
        if token == '(':
            self._consume('(')
            result = self._parse_expression()
            self._consume(')')
            return result
        elif token is not None and (token[0].isdigit() or token[0] == '.'):
            val_str = self._consume()
            try:
                return float(val_str)
            except ValueError:
                raise ValueError(f"Invalid number: {val_str}")
        else:
            raise ValueError(f"Unexpected token: {token}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("15 / 4") == 3.75

def test_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0

def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_errors(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3")