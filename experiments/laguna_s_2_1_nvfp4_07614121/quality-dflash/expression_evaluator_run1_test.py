# expression_evaluator.py

import re
from typing import List, Union

class ExpressionEvaluator:
    """
    A class to evaluate mathematical expressions using a recursive descent parser.

    Supports +, -, *, / with correct operator precedence, parentheses for grouping,
    unary minus, and floating-point numbers.

    Grammar:
        expression -> term (('+' | '-') term)*
        term       -> factor (('*' | '/') factor)*
        factor     -> ('-' | '+') factor | NUMBER | '(' expression ')'
    """

    def __init__(self):
        """Initializes the evaluator."""
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string.

        Args:
            expr: The mathematical expression to evaluate.

        Returns:
            The result of the evaluated expression as a float.

        Raises:
            ValueError: If the expression is invalid (e.g., empty, mismatched parentheses,
                        division by zero, or contains invalid tokens).
        """
        if not expr.strip():
            raise ValueError("Cannot evaluate an empty expression.")

        self._tokenize(expr)
        self.pos = 0

        try:
            result = self._parse_expression()
            if self.pos != len(self.tokens):
                raise ValueError("Unexpected token(s) after end of expression.")
            return result
        except IndexError:
            # This can happen if we run out of tokens unexpectedly
            raise ValueError("Invalid expression format.")
        except ZeroDivisionError:
            raise ValueError("Division by zero.")

    def _tokenize(self, expr: str) -> None:
        """
        Converts the input string into a list of tokens.

        Args:
            expr: The expression string.

        Raises:
            ValueError: If an invalid token is found.
        """
        # Regular expression to match valid tokens
        token_pattern = r'\s*([0-9]+\.?[0-9]*|[-+*/()])\s*'
        tokens = re.findall(token_pattern, expr)

        # Check for any remaining characters that don't match the pattern
        cleaned_expr = re.sub(r'\s+', '', expr)
        for char in cleaned_expr:
            if char not in '0123456789+-*/().':
                raise ValueError(f"Invalid character '{char}' in expression.")

        self.tokens = tokens

    def _peek_token(self) -> str:
        """Returns the current token without consuming it."""
        if self.pos >= len(self.tokens):
            return ''
        return self.tokens[self.pos]

    def _consume_token(self) -> str:
        """Consumes and returns the current token."""
        if self.pos >= len(self.tokens):
            raise IndexError("No more tokens to consume.")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parses an expression: term (('+' | '-') term)*."""
        result = self._parse_term()
        while True:
            op = self._peek_token()
            if op == '+':
                self._consume_token()
                result += self._parse_term()
            elif op == '-':
                self._consume_token()
                result -= self._parse_term()
            else:
                break
        return result

    def _parse_term(self) -> float:
        """Parses a term: factor (('*' | '/') factor)*."""
        result = self._parse_factor()
        while True:
            op = self._peek_token()
            if op == '*':
                self._consume_token()
                result *= self._parse_factor()
            elif op == '/':
                self._consume_token()
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ZeroDivisionError("Division by zero.")
                result /= divisor
            else:
                break
        return result

    def _parse_factor(self) -> float:
        """Parses a factor: ('-' | '+') factor | NUMBER | '(' expression ')'."""
        token = self._peek_token()

        # Handle unary operators
        if token in ('-', '+'):
            self._consume_token()
            value = self._parse_factor()
            return -value if token == '-' else value

        # Handle number
        if re.match(r'^[0-9]+\.?[0-9]*$', token):
            self._consume_token()
            return float(token)

        # Handle parentheses
        if token == '(':
            self._consume_token()
            value = self._parse_expression()
            if self._peek_token() != ')':
                raise ValueError("Mismatched parentheses.")
            self._consume_token() # Consume ')'
            return value

        raise ValueError(f"Unexpected token '{token}' in expression.")

# test_expression_evaluator.py

import pytest

def test_basic_operations():
    """Tests basic arithmetic operations with correct precedence."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_unary_minus_and_floats():
    """Tests unary minus and floating-point numbers."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2+1)") == -3.0
    assert evaluator.evaluate("3.14 + 2.86") == pytest.approx(6.0)
    assert evaluator.evaluate("-3.5") == -3.5
    assert evaluator.evaluate("2 * -4") == -8.0

def test_parentheses():
    """Tests complex grouping with parentheses."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("((1+2)*(3+4))") == 21.0
    assert evaluator.evaluate("2 * (3 + (4 * 5))") == 46.0

def test_invalid_expressions():
    """Tests that various invalid expressions raise ValueError."""
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + abc")

def test_complex_expression():
    """Tests a more complex expression combining all features."""
    evaluator = ExpressionEvaluator()
    expr = "3 + 4 * 2 / (1 - 5) - 2"
    # Manual calculation: 3 + (4*2)/(1-5) - 2 = 3 + 8/(-4) - 2 = 3 - 2 - 2 = -1
    assert evaluator.evaluate(expr) == -1.0