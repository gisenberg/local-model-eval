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
    - Error handling for invalid expressions, mismatched parentheses, division by zero

    Methods:
        evaluate(expr: str) -> float: Evaluates the given expression string and returns the result.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.current_token: Optional[str] = None
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the given mathematical expression string.

        Args:
            expr (str): The expression to evaluate.

        Returns:
            float: The result of the evaluation.

        Raises:
            ValueError: For invalid tokens, mismatched parentheses, division by zero.
        """
        if not expr:
            raise ValueError("Empty expression")

        self.tokens = self.tokenize(expr)
        self.pos = 0
        self.current_token = self.tokens[self.pos]

        try:
            result = self.parse_expression()
            return result
        except IndexError:
            raise ValueError("Invalid token")
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except Exception as e:
            raise ValueError(str(e))

    def tokenize(self, expr: str) -> List[str]:
        """
        Tokenizes the input expression into a list of tokens.

        Args:
            expr (str): The expression to tokenize.

        Returns:
            List[str]: A list of tokens (numbers, operators, parentheses).
        """
        # Remove all whitespace
        expr = expr.replace(" ", "")

        # Tokenize using regular expressions
        tokens = re.findall(r'([()+\-*/])|(\d+\.\d+|\d+)', expr)

        result = []
        for token in tokens:
            if token[0]:  # operator or parenthesis
                result.append(token[0])
            elif token[1]:  # number
                result.append(token[1])

        # Validate token list
        if not result:
            raise ValueError("Empty expression")

        # Check for invalid characters
        for token in result:
            if not (token in "+-*/()" or re.match(r'^\d+\.\d+$', token) or re.match(r'^\d+$', token)):
                raise ValueError(f"Invalid token: {token}")

        return result

    def parse_expression(self) -> float:
        """Parses an expression with operator precedence."""
        return self.parse_additive()

    def parse_additive(self) -> float:
        """Parses additive expressions (addition and subtraction)."""
        result = self.parse_multiplicative()

        while self.current_token in ('+', '-'):
            op = self.current_token
            self.pos += 1
            self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None

            right = self.parse_multiplicative()
            if op == '+':
                result += right
            else:
                result -= right

        return result

    def parse_multiplicative(self) -> float:
        """Parses multiplicative expressions (multiplication and division)."""
        result = self.parse_unary()

        while self.current_token in ('*', '/'):
            op = self.current_token
            self.pos += 1
            self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None

            right = self.parse_unary()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right

        return result

    def parse_unary(self) -> float:
        """Parses unary expressions (unary minus and numbers)."""
        if self.current_token == '-':
            self.pos += 1
            self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None
            return -self.parse_unary()
        else:
            return self.parse_primary()

    def parse_primary(self) -> float:
        """Parses primary expressions (numbers and parentheses)."""
        if self.current_token == '(':
            self.pos += 1
            self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None
            result = self.parse_expression()
            if self.current_token != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None
            return result
        elif re.match(r'^\d+\.\d+$', self.current_token) or re.match(r'^\d+$', self.current_token):
            value = float(self.current_token)
            self.pos += 1
            self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None
            return value
        else:
            raise ValueError(f"Unexpected token: {self.current_token}")

import pytest


def test_valid_expression():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 4 * 2") == 11.0
    assert evaluator.evaluate("(3 + 4) * 2") == 14.0
    assert evaluator.evaluate("-3 + 4") == 1.0
    assert evaluator.evaluate("-(-3 + 4)") == -1.0
    assert evaluator.evaluate("3.14 + 2.5") == 5.64
    assert evaluator.evaluate("10 / 2") == 5.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    assert evaluator.evaluate("10 * (2 + 3)") == 50.0
    assert evaluator.evaluate("10 * (2 + 3) - 5") == 45.0

def test_invalid_expression():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 + * 4")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("3 + (4")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 + 4 * 2 +")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 + 4 * 2 + 5 *")

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 4") == 1.0
    assert evaluator.evaluate("-(-3 + 4)") == -1.0
    assert evaluator.evaluate("-3 * -4") == 12.0
    assert evaluator.evaluate("-3.14") == -3.14

def test_float_numbers():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 + 2.5") == 5.64
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("3.14 / 2") == 1.57
    assert evaluator.evaluate("3.14 - 2.5") == 0.64

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(3 + 4) * (2 + 3)") == 35.0
    assert evaluator.evaluate("((3 + 4) * 2) + 5") == 19.0
    assert evaluator.evaluate("((3 + 4) * (2 + 3)) / 2") == 17.5