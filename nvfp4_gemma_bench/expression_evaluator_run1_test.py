import re
, Tuple

class ExpressionEvaluator:
    """
    Evaluates mathematical expressions with support for +, -, *, /, 
    parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        # Token patterns: float/int, operators, and parentheses
        self._token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates a mathematical expression string.
        
        Args:
            expr: The string expression to evaluate.
            
        Returns:
            The result as a float.
            
        Raises:
            ValueError: For syntax errors, mismatched parentheses, or division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize the input string
        tokens = self._tokenize(expr)
        self._pos = 0
        self._tokens = tokens

        result = self._parse_expression()

        # If there are tokens left after parsing the main expression, it's a syntax error
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the input string into a list of valid tokens."""
        # Remove whitespace
        expr = expr.replace(" ", "")
        tokens = self._token_pattern.findall(expr)
        
        # Validate that we didn't miss any characters (invalid tokens)
        if "".join(tokens) != expr:
            # Find the first character that didn't match the pattern
            # This is a simple way to detect invalid characters like '@' or '&'
            raise ValueError("Expression contains invalid characters")
            
        return tokens

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
        """Handles multiplication and division."""
        result = self._parse_unary()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_unary()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_unary(self) -> float:
        """Handles unary minus and plus."""
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        if self._peek() == '+':
            self._consume()
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles parentheses and numbers (highest precedence)."""
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            return result

        if token is None:
            raise ValueError("Unexpected end of expression")

        try:
            val = float(self._consume())
            return val
        except ValueError:
            raise ValueError(f"Invalid token: {token}")

import pytest


def test_basic_operations():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 5 * 2") == 13.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    assert evaluator.evaluate("2.5 * 4") == 10.0

def test_parentheses_and_unary():
    evaluator = ExpressionEvaluator()
    # Test grouping
    assert evaluator.evaluate("(3 + 5) * 2") == 16.0
    # Test unary minus
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-(-5)") == 5.0

def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("10 / 4") == 2.5

def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

def test_error_handling():
    evaluator = ExpressionEvaluator()
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    # Invalid tokens
    with pytest.raises(ValueError, match="invalid characters"):
        evaluator.evaluate("3 @ 2")
    # Empty expression
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("   ")
    # Syntax error (e.g., two numbers in a row)
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 2 + 1")