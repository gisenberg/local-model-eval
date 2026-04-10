import re
from typing import List, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser
    to evaluate arithmetic expressions with operator precedence and unary operators.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates the given mathematical expression string.

        Args:
            expr: The string expression to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens, 
                       has mismatched parentheses, or involves division by zero.
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
        Converts the input string into a list of valid tokens (numbers, operators, parentheses).

        Args:
            expr: The raw input string.

        Returns:
            A list of strings representing tokens.

        Raises:
            ValueError: If an invalid character is encountered.
        """
        # Regex pattern: matches floats/integers, operators (+, -, *, /), or parentheses
        token_pattern = re.compile(r"(\d*\.\d+|\d+|[+\-*/()])")
        tokens = []
        
        # Remove whitespace and find all matches
        clean_expr = expr.replace(" ", "")
        
        # Check for invalid characters by comparing reconstructed string
        # This is a simple way to ensure no illegal characters exist
        found_tokens = token_pattern.findall(clean_expr)
        reconstructed = "".join(found_tokens)
        
        if len(reconstructed) != len(clean_expr):
            # Find the first character that didn't match a token
            for char in clean_expr:
                if not token_pattern.match(char):
                    raise ValueError(f"Invalid token detected: '{char}'")

        return found_tokens

    def _peek(self) -> Union[str, None]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Consumes and returns the current token."""
        token = self._peek()
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).

        Returns:
            The evaluated result.
        """
        node = self._parse_term()
        while self._peek() in ("+", "-"):
            op = self._consume()
            right = self._parse_term()
            if op == "+":
                node += right
            else:
                node -= right
        return node

    def _parse_term(self) -> float:
        """
        Parses multiplication and division.

        Returns:
            The evaluated result.
        """
        node = self._parse_unary()
        while self._peek() in ("*", "/"):
            op = self._consume()
            right = self._parse_unary()
            if op == "*":
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero is not allowed.")
                node /= right
        return node

    def _parse_unary(self) -> float:
        """
        Parses unary minus.

        Returns:
            The evaluated result.
        """
        if self._peek() == "-":
            self._consume()
            return -self._parse_unary()
        return self._parse_factor()

    def _parse_factor(self) -> float:
        """
        Parses numbers and parenthesized expressions (highest precedence).

        Returns:
            The evaluated result.

        Raises:
            ValueError: If parentheses are mismatched or tokens are invalid.
        """
        token = self._peek()

        if token == "(":
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ")":
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._consume()  # consume ')'
            return result

        if token is None:
            raise ValueError("Unexpected end of expression.")

        # Try to parse a number
        try:
            self._consume()
            # Check if the token is a valid number
            return float(token)
        except ValueError:
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