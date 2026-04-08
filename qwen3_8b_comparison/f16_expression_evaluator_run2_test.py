from typing import Optional, List, Tuple
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

    def __init__(self, expr: str = ""):
        self.expr = expr
        self.pos = 0
        self.tokens = self.tokenize(expr)

    def tokenize(self, expr: str) -> List[str]:
        """
        Tokenizes the input expression into numbers, operators, and parentheses.
        """
        # Use regex to split the expression into tokens
        token_pattern = r'(\d+\.\d+|\d+|[-+*/()])'
        tokens = re.findall(token_pattern, expr)
        return tokens

    def evaluate(self) -> float:
        """
        Evaluates the expression using a recursive descent parser.
        """
        self.pos = 0
        return self.parse_expression()

    def parse_expression(self) -> float:
        """
        Parses an expression (highest precedence)
        expression := addition_expression
        """
        return self.parse_addition()

    def parse_addition(self) -> float:
        """
        Parses addition and subtraction
        addition_expression := multiplication_expression (('+' | '-') multiplication_expression)*
        """
        result = self.parse_multiplication()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.parse_multiplication()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def parse_multiplication(self) -> float:
        """
        Parses multiplication and division
        multiplication_expression := unary_expression (('*' | '/') unary_expression)*
        """
        result = self.parse_unary()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.parse_unary()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def parse_unary(self) -> float:
        """
        Parses unary minus and numbers
        unary_expression := ('-' unary_expression) | number
        """
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1  # consume the unary minus
            return -self.parse_unary()
        else:
            return self.parse_number()

    def parse_number(self) -> float:
        """
        Parses a number (integer or float)
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        token = self.tokens[self.pos]
        if '.' in token:
            return float(token)
        else:
            return int(token)

    def check_parentheses(self) -> None:
        """
        Checks if the expression has balanced parentheses.
        """
        count = 0
        for token in self.tokens:
            if token == '(':
                count += 1
            elif token == ')':
                count -= 1
                if count < 0:
                    raise ValueError("Mismatched parentheses")
        if count != 0:
            raise ValueError("Mismatched parentheses")

    def evaluate(self, expr: str) -> float:
        """
        Main entry point to evaluate an expression.

        Args:
            expr (str): The mathematical expression to evaluate.

        Returns:
            float: The result of the evaluation.

        Raises:
            ValueError: For invalid expressions, mismatched parentheses, division by zero, etc.
        """
        self.expr = expr
        self.tokens = self.tokenize(expr)
        self.check_parentheses()
        return self.parse_expression()

import pytest


def test_valid_expression():
    evaluator = ExpressionEvaluator("3 + 4 * 2")
    result = evaluator.evaluate()
    assert result == 11.0

def test_unary_minus():
    evaluator = ExpressionEvaluator("-3 + 2")
    result = evaluator.evaluate()
    assert result == -1.0

def test_parentheses():
    evaluator = ExpressionEvaluator("(3 + 4) * 2")
    result = evaluator.evaluate()
    assert result == 14.0

def test_division_by_zero():
    with pytest.raises(ValueError):
        ExpressionEvaluator("5 / 0").evaluate()

def test_invalid_token():
    with pytest.raises(ValueError):
        ExpressionEvaluator("3 + * 2").evaluate()

def test_empty_expression():
    with pytest.raises(ValueError):
        ExpressionEvaluator("").evaluate()

def test_float_numbers():
    evaluator = ExpressionEvaluator("3.14 + 2.71")
    result = evaluator.evaluate()
    assert result == 5.85

def test_nested_parentheses():
    evaluator = ExpressionEvaluator("((3 + 4) * 2) - 1")
    result = evaluator.evaluate()
    assert result == 13.0

def test_invalid_expression():
    with pytest.raises(ValueError):
        ExpressionEvaluator("3 + * 2").evaluate()

def test_unary_minus_with_parentheses():
    evaluator = ExpressionEvaluator("-(-3 + 2)")
    result = evaluator.evaluate()
    assert result == -1.0

def test_invalid_operator():
    with pytest.raises(ValueError):
        ExpressionEvaluator("3 + * 2").evaluate()

def test_invalid_token_in_middle():
    with pytest.raises(ValueError):
        ExpressionEvaluator("3 + * 2").evaluate()

def test_invalid_token_after_number():
    with pytest.raises(ValueError):
        ExpressionEvaluator("3 + *").evaluate()

def test_invalid_token_before_number():
    with pytest.raises(ValueError):
        ExpressionEvaluator("*3").evaluate()

evaluator = ExpressionEvaluator("3 + 4 * 2")
result = evaluator.evaluate()
print(result)  # Output: 11.0