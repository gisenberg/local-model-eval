from typing import List, Optional
import re

class ExpressionEvaluator:
    """
    A class to evaluate mathematical expressions with support for:
    - +, -, *, /
    - parentheses
    - unary minus
    - floating point numbers
    - correct operator precedence
    - error handling for invalid input
    """

    def __init__(self):
        self.tokens = []

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.

        Args:
            expr (str): The expression to evaluate.

        Returns:
            float: The result of the evaluation.

        Raises:
            ValueError: For invalid expressions, mismatched parentheses, division by zero, etc.
        """
        if not expr:
            raise ValueError("Empty expression")

        # Tokenize the expression
        self.tokens = self.tokenize(expr)

        # Check for mismatched parentheses
        if self.tokens.count('(') != self.tokens.count(')'):
            raise ValueError("Mismatched parentheses")

        # Evaluate the expression
        return self.parse_expression()

    def tokenize(self, expr: str) -> List[str]:
        """
        Tokenize the expression into numbers, operators, and parentheses.

        Args:
            expr (str): The expression to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        # Remove whitespace
        expr = expr.replace(" ", "")

        # Split into tokens
        tokens = re.findall(r'([()+\-*/()])|(\d+\.?\d*)', expr)

        # Flatten the list and filter out empty strings
        tokens = [token for token in [item for sublist in tokens for item in sublist] if token]

        # Check for invalid tokens
        if any(token not in "+-*/()0123456789." for token in tokens):
            raise ValueError("Invalid token")

        return tokens

    def parse_expression(self) -> float:
        """
        Parse and evaluate the expression using recursive descent.

        Returns:
            float: The result of the expression.
        """
        return self.parse_term()

    def parse_term(self) -> float:
        """
        Parse and evaluate a term (addition or subtraction).

        Returns:
            float: The result of the term.
        """
        result = self.parse_factor()
        while self.current_token() in ('+', '-'):
            op = self.consume_token()
            right = self.parse_factor()
            if op == '+':
                result += right
            elif op == '-':
                result -= right
        return result

    def parse_factor(self) -> float:
        """
        Parse and evaluate a factor (multiplication or division).

        Returns:
            float: The result of the factor.
        """
        result = self.parse_primary()
        while self.current_token() in ('*', '/'):
            op = self.consume_token()
            right = self.parse_primary()
            if op == '*':
                result *= right
            elif op == '/':
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def parse_primary(self) -> float:
        """
        Parse and evaluate a primary expression (number, unary minus, or parenthesized expression).

        Returns:
            float: The result of the primary expression.
        """
        token = self.current_token()
        if token == '(':
            self.consume_token()  # consume '('
            result = self.parse_expression()
            if self.current_token() != ')':
                raise ValueError("Mismatched parentheses")
            self.consume_token()  # consume ')'
            return result
        elif token == '-':
            # Unary minus
            self.consume_token()
            return -self.parse_primary()
        else:
            # Number
            result = float(token)
            self.consume_token()
            return result

    def current_token(self) -> str:
        """Return the current token."""
        return self.tokens[0] if self.tokens else ''

    def consume_token(self) -> str:
        """Consume and return the current token."""
        if not self.tokens:
            raise ValueError("Unexpected end of expression")
        token = self.tokens.pop(0)
        return token

import pytest


def test_expression_evaluator():
    evaluator = ExpressionEvaluator()

    # Test 1: Basic arithmetic
    assert evaluator.evaluate("3 + 4") == 7.0

    # Test 2: Multiplication and division
    assert evaluator.evaluate("6 * 7 / 3") == 14.0

    # Test 3: Parentheses
    assert evaluator.evaluate("(3 + 4) * 2") == 14.0

    # Test 4: Unary minus
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0

    # Test 5: Floating point
    assert evaluator.evaluate("3.14 + 2.5") == 5.64

    # Test 6: Division by zero
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")

    # Test 7: Mismatched parentheses
    with pytest.raises(ValueError):
        evaluator.evaluate("(3 + 4")

    # Test 8: Empty expression
    with pytest.raises(ValueError):
        evaluator.evaluate("")

    # Test 9: Invalid token
    with pytest.raises(ValueError):
        evaluator.evaluate("3 + * 4")

    # Test 10: Complex expression
    assert evaluator.evaluate("-(3 + (4 * (5 - 2)))") == -15.0

    print("All tests passed!")

if __name__ == "__main__":
    test_expression_evaluator()