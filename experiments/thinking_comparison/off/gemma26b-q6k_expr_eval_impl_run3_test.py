import re
from typing import List, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser
    to evaluate arithmetic expressions with support for precedence and unary operators.
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
                       has mismatched parentheses, or results in division by zero.
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
        Converts the input string into a list of meaningful tokens.

        Args:
            expr: The raw input string.

        Returns:
            A list of strings representing tokens.
        """
        # Regex pattern to match numbers (including floats), operators, and parentheses
        token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')
        tokens = token_pattern.findall(expr)

        # Validate that the reconstructed string from tokens matches the original 
        # (ignoring whitespace) to catch invalid characters.
        sanitized_expr = "".join(expr.split())
        reconstructed = "".join(tokens)
        
        # We check if the tokens cover all non-whitespace characters in the input
        # This is a simple way to detect invalid characters like 'a', '$', etc.
        if len(reconstructed) < len(sanitized_expr):
            # Find the first character that wasn't captured by the regex
            for char in sanitized_expr:
                if char not in reconstructed and not char.isspace():
                    raise ValueError(f"Invalid token detected: '{char}'")
            # If the lengths differ but no specific char was found, it's a general invalidity
            raise ValueError("Expression contains invalid characters")

        return tokens

    def _peek(self) -> Union[str, None]:
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

        Returns:
            The evaluated result of the expression.
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

        Returns:
            The evaluated result of the term.
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

        Returns:
            The evaluated result of the unary operation.
        """
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_factor()

    def _parse_factor(self) -> float:
        """
        Handles parentheses and base numbers (highest precedence).

        Returns:
            The evaluated result of the factor.
        """
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._consume()  # consume ')'
            return result

        if token is None:
            raise ValueError("Unexpected end of expression")

        # Try to parse a number
        try:
            # Check if token is a valid number (digit or decimal)
            if not re.match(r'^\d*\.\d+|\d+$', token):
                raise ValueError(f"Invalid token: '{token}'")
            
            val = float(self._consume())
            return val
        except ValueError as e:
            if "Division by zero" in str(e) or "Mismatched" in str(e):
                raise e
            raise ValueError(f"Invalid token: '{token}'")

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