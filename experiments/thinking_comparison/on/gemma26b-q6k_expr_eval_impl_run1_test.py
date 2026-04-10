import re
from typing import List

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
        Evaluates a mathematical expression string and returns the result.

        Args:
            expr: The mathematical expression as a string.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        # If there are remaining tokens, the expression is invalid (e.g., "1 + 2 3" or "1 + 2 )")
        if self._pos < len(self._tokens):
            token = self._tokens[self._pos]
            if token == ')':
                raise ValueError("Mismatched parentheses")
            raise ValueError("Invalid expression")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of valid mathematical tokens.

        Args:
            expr: The raw expression string.

        Returns:
            A list of strings representing tokens.

        Raises:
            ValueError: If the expression contains characters that are not valid tokens.
        """
        # Regex pattern for floats, integers, and operators/parentheses
        pattern = r'\d+\.\d*|\.\d+|\d+|[+\-*/()]'
        tokens = re.findall(pattern, expr)

        # Check if the joined tokens match the original string (ignoring whitespace)
        # to detect invalid characters like '@' or letters.
        if len("".join(tokens)) != len(expr.replace(" ", "").replace("\t", "").replace("\n", "")):
            raise ValueError("Invalid token")

        return tokens

    def _parse_expression(self) -> float:
        """
        Parses an expression involving addition and subtraction.

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
        Parses a term involving multiplication and division.

        Returns:
            The evaluated result of the term.

        Raises:
            ValueError: If division by zero is attempted.
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
        Parses unary operators (specifically unary minus).

        Returns:
            The evaluated result of the unary operation.
        """
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parses the most basic units: numbers and parenthesized expressions.

        Returns:
            The evaluated number or the result of the parenthesized expression.

        Raises:
            ValueError: If parentheses are mismatched or the token is invalid.
        """
        token = self._consume()

        if token == '(':
            val = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()  # consume ')'
            return val
        
        if token == ')':
            raise ValueError("Mismatched parentheses")

        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid token: {token}")

    def _peek(self) -> str:
        """
        Returns the current token without consuming it.

        Returns:
            The current token string, or an empty string if at the end.
        """
        if self._pos >= len(self._tokens):
            return ""
        return self._tokens[self._pos]

    def _consume(self) -> str:
        """
        Consumes and returns the current token.

        Returns:
            The current token string.

        Raises:
            ValueError: If no more tokens are available.
        """
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression")
        token = self._tokens[self._pos]
        self._pos += 1
        return token

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