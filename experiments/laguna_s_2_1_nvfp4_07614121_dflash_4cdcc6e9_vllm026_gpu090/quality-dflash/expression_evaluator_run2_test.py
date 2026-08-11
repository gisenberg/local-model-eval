import re
from typing import List, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic operations,
    parentheses, unary minus, and floating-point numbers.

    Supported operators:
        + : Addition
        - : Subtraction / Unary Minus
        * : Multiplication
        / : Division

    Features:
        - Operator precedence (*, / before +, -)
        - Parentheses for grouping
        - Floating-point number support
        - Error handling for invalid input
    """

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.

        Args:
            expr (str): The mathematical expression to evaluate.

        Returns:
            float: The result of the evaluated expression.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        # Tokenize the expression
        self.tokens = self._tokenize(expr)
        self.pos = 0

        try:
            result = self._parse_expression()
            if self.pos < len(self.tokens):
                raise ValueError("Unexpected token after expression")
            return result
        except IndexError:
            raise ValueError("Mismatched parentheses")

    def _tokenize(self, expr: str) -> List[str]:
        """
        Tokenizes the input expression into a list of valid tokens.

        Args:
            expr (str): The expression string to tokenize.

        Returns:
            List[str]: A list of tokens representing numbers, operators, and parentheses.

        Raises:
            ValueError: If an invalid token is encountered.
        """
        token_pattern = r'(\d+\.?\d*|\.\d+|[+\-*/()])'
        tokens = re.findall(token_pattern, expr)

        # Check for any characters that are not part of valid tokens
        cleaned_expr = re.sub(r'\s+', '', expr)
        reconstructed = ''.join(tokens)
        if reconstructed != cleaned_expr:
            invalid_chars = set(cleaned_expr) - set(reconstructed)
            if invalid_chars:
                raise ValueError(f"Invalid token(s) found: {invalid_chars}")

        return tokens

    def _peek(self) -> Union[str, None]:
        """Returns the current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self) -> str:
        """Consumes and returns the current token."""
        if self.pos >= len(self.tokens):
            raise IndexError("Unexpected end of expression")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses an expression: term ((+|-) term)*
        """
        result = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._consume()
            rhs = self._parse_term()
            if op == '+':
                result += rhs
            elif op == '-':
                result -= rhs
        return result

    def _parse_term(self) -> float:
        """
        Parses a term: factor ((*|/) factor)*
        """
        result = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            rhs = self._parse_factor()
            if op == '*':
                result *= rhs
            elif op == '/':
                if rhs == 0:
                    raise ValueError("Division by zero")
                result /= rhs
        return result

    def _parse_factor(self) -> float:
        """
        Parses a factor: number | '(' expression ')' | '-' factor
        """
        token = self._peek()
        if token == '-':
            self._consume()
            return -self._parse_factor()
        elif token == '(':
            self._consume()
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()
            return result
        else:
            return float(self._consume())

import pytest

evaluator = ExpressionEvaluator()

def test_basic_operations():
    assert evaluator.evaluate("2 + 3 * 4") == 10.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0

def test_unary_minus():
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 * -2") == -6.0

def test_floating_point_numbers():
    assert evaluator.evaluate("3.14 + 2.86") == pytest.approx(6.0)
    assert evaluator.evaluate(".5 + .25") == pytest.approx(0.75)
    assert evaluator.evaluate("2.5 * 4") == 10.0

def test_invalid_inputs():
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")

    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 & 3")

    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")

    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")

    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

def test_complex_expressions():
    assert evaluator.evaluate("((15 - (6 / 2)) * 2)") == 18.0
    assert evaluator.evaluate("3 + 4 * 2 / (1 - 5)") == 1.0
    assert evaluator.evaluate("2 * -(3 + 4)") == -14.0