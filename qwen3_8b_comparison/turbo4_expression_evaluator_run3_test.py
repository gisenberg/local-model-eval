from typing import List, Optional, Tuple
import re

class ExpressionEvaluator:
    """
    A class to evaluate mathematical expressions with support for:
    - +, -, *, /
    - Parentheses
    - Unary minus
    - Floating point numbers
    - Correct operator precedence
    - Raises ValueError for invalid expressions
    """

    def __init__(self):
        self.tokens = []
        self.pos = 0

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
        self.tokens = self.tokenize(expr)
        self.pos = 0
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

        # Match numbers (including floating point)
        tokens = re.findall(r'(\d+\.\d+|\d+|[-+*/()])', expr)

        # Check for invalid tokens
        if not tokens:
            raise ValueError("Invalid tokens")

        # Check for mismatched parentheses
        if tokens.count('(') != tokens.count(')'):
            raise ValueError("Mismatched parentheses")

        return tokens

    def parse_expression(self) -> float:
        """Parse the expression using recursive descent."""
        result = self.parse_term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.parse_term()
            result = self.apply_binary_op(result, op, right)
        return result

    def parse_term(self) -> float:
        """Parse terms with * and /."""
        result = self.parse_factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.parse_factor()
            result = self.apply_binary_op(result, op, right)
        return result

    def parse_factor(self) -> float:
        """Parse factors, including parentheses and unary minus."""
        token = self.tokens[self.pos]
        if token == '(':
            self.pos += 1
            result = self.parse_expression()
            if self.pos < len(self.tokens) and self.tokens[self.pos] == ')':
                self.pos += 1
            else:
                raise ValueError("Mismatched parentheses")
            return result
        elif token == '-':
            # Unary minus
            self.pos += 1
            return -self.parse_factor()
        else:
            # Number
            self.pos += 1
            return float(token)

    def apply_binary_op(self, left: float, op: str, right: float) -> float:
        """Apply binary operation."""
        if op == '+':
            return left + right
        elif op == '-':
            return left - right
        elif op == '*':
            return left * right
        elif op == '/':
            if right == 0:
                raise ValueError("Division by zero")
            return left / right
        else:
            raise ValueError(f"Unknown operator: {op}")

    def parse_unary_minus(self) -> float:
        """Handle unary minus."""
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1
            return -self.parse_factor()
        return self.parse_factor()

import pytest


def test_expression_evaluator():
    evaluator = ExpressionEvaluator()

    # Test 1: Basic arithmetic
    assert evaluator.evaluate("3 + 4 * 2") == 11.0

    # Test 2: Parentheses
    assert evaluator.evaluate("(3 + 4) * 2") == 14.0

    # Test 3: Unary minus
    assert evaluator.evaluate("-3 + 4") == 1.0
    assert evaluator.evaluate("-(-3 + 4)") == -1.0

    # Test 4: Floating point
    assert evaluator.evaluate("3.14 + 2.5") == 5.64

    # Test 5: Division by zero
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")

    # Test 6: Mismatched parentheses
    with pytest.raises(ValueError):
        evaluator.evaluate("(3 + 4")

    # Test 7: Empty expression
    with pytest.raises(ValueError):
        evaluator.evaluate("")

    # Test 8: Invalid tokens
    with pytest.raises(ValueError):
        evaluator.evaluate("3 + * 4")

    # Test 9: Complex expression
    assert evaluator.evaluate("-(3 + (4 * (2 - 1)))") == -7.0

    # Test 10: Mixed operations
    assert evaluator.evaluate("3 + 4 * (2 + 3) - 1") == 22.0