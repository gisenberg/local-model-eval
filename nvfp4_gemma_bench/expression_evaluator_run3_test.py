import re
, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, unary minus, parentheses, and floating point numbers.
    """

    def __init__(self):
        # Token patterns: floating point numbers or operators/parentheses
        self._token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        :param expr: The expression string to evaluate.
        :raises ValueError: For invalid tokens, mismatched parentheses, or division by zero.
        :return: The result of the evaluation.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize the input string
        tokens = self._tokenize(expr)
        self._pos = 0
        self._tokens = tokens

        result = self._parse_expression()

        # If there are tokens left over, the expression was malformed
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the input string into a list of tokens."""
        # Remove whitespace
        expr = expr.replace(" ", "")
        
        # Validate for invalid characters
        if re.search(r'[^0-9.+\-*/() ]', expr):
            invalid_char = re.search(r'[^0-9.+\-*/() ]', expr).group()
            raise ValueError(f"Invalid token detected: {invalid_char}")

        return self._token_pattern.findall(expr)

    def _peek(self) -> str:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Returns the current token and advances the position."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
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
        """Handles unary minus."""
        if self._peek() == '-':
            self._consume()  # consume '-'
            return -self._parse_unary()
        return self._parse_factor()

    def _parse_factor(self) -> float:
        """Handles parentheses and numbers (highest precedence)."""
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

        # Try to parse as a number
        try:
            val = float(self._consume())
            return val
        except ValueError:
            raise ValueError(f"Invalid number format: {token}")

        # This part is technically unreachable due to _tokenize validation
        raise ValueError(f"Unexpected token: {token}")

import pytest


def test_basic_operations():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 5 * 2") == 13.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_parentheses_and_unary():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("-3 * -2") == 6.0
    assert evaluator.evaluate("10 + (-2 * 3)") == 4.0

def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

def test_mismatched_parentheses():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError):
        evaluator.evaluate("1 + 2)")

def test_invalid_inputs():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 + a")
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")