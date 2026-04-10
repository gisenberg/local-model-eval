from typing import List, Optional, Union
import re

class ExpressionEvaluator:
    """
    A class to evaluate mathematical expressions with support for:
    - +, -, *, /
    - Parentheses
    - Unary minus
    - Floating point numbers
    - Error handling for invalid expressions

    Attributes:
        expr (str): The input expression to evaluate
        tokens (List[str]): Tokenized expression
        pos (int): Current position in the tokens list

    Methods:
        evaluate: Evaluate the expression
        tokenize: Tokenize the expression
        parse: Parse and evaluate the expression using recursive descent
        parse_expression: Handle expression with operator precedence
        parse_term: Handle multiplication/division
        parse_factor: Handle numbers and parentheses
        parse_unary: Handle unary minus
    """

    def __init__(self, expr: str):
        self.expr = expr
        self.tokens = self.tokenize(expr)
        self.pos = 0

    def evaluate(self) -> float:
        """
        Evaluate the expression.

        Returns:
            float: The result of the expression evaluation

        Raises:
            ValueError: For invalid expressions, mismatched parentheses, division by zero
        """
        if not self.tokens:
            raise ValueError("Empty expression")
        result = self.parse_expression()
        if self.pos != len(self.tokens):
            raise ValueError("Invalid expression: unexpected tokens at the end")
        return result

    def tokenize(self, expr: str) -> List[str]:
        """
        Tokenize the expression into numbers, operators, and parentheses.

        Args:
            expr (str): The input expression

        Returns:
            List[str]: List of tokens
        """
        # Regular expression to match numbers, operators, and parentheses
        token_pattern = r'(\d+\.\d+|\d+|[-+*/()])'
        tokens = re.findall(token_pattern, expr)
        return tokens

    def parse_expression(self) -> float:
        """
        Parse an expression with operator precedence: +, -
        """
        result = self.parse_term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.parse_term()
            if op == '+':
                result += right
            elif op == '-':
                result -= right
        return result

    def parse_term(self) -> float:
        """
        Parse a term with operator precedence: *, /
        """
        result = self.parse_factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.parse_factor()
            if op == '*':
                result *= right
            elif op == '/':
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def parse_factor(self) -> float:
        """
        Parse a factor, which can be a number or a parenthesized expression
        """
        if self.tokens[self.pos] == '(':
            self.pos += 1
            result = self.parse_expression()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result
        else:
            return self.parse_unary()

    def parse_unary(self) -> float:
        """
        Parse unary minus
        """
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1
            return -self.parse_factor()
        else:
            return self.parse_number()

    def parse_number(self) -> float:
        """
        Parse a number (integer or float)
        """
        token = self.tokens[self.pos]
        if '.' in token:
            return float(token)
        else:
            return int(token)

    def __str__(self):
        return f"ExpressionEvaluator(expr='{self.expr}')"

import pytest


def test_valid_expression():
    evaluator = ExpressionEvaluator("3 + 4 * 2")
    assert evaluator.evaluate() == 11.0

def test_unary_minus():
    evaluator = ExpressionEvaluator("-3 + 4")
    assert evaluator.evaluate() == 1.0
    evaluator = ExpressionEvaluator("-(-3 + 4)")
    assert evaluator.evaluate() == -1.0

def test_parentheses():
    evaluator = ExpressionEvaluator("(3 + 4) * 2")
    assert evaluator.evaluate() == 14.0
    evaluator = ExpressionEvaluator("((3 + 4) * 2) / 2")
    assert evaluator.evaluate() == 7.0

def test_float_numbers():
    evaluator = ExpressionEvaluator("3.14 + 2.71")
    assert evaluator.evaluate() == 5.85
    evaluator = ExpressionEvaluator("3.14 * 2")
    assert evaluator.evaluate() == 6.28

def test_invalid_expression():
    with pytest.raises(ValueError):
        ExpressionEvaluator("3 + 4 *")
    with pytest.raises(ValueError):
        ExpressionEvaluator("3 + 4 * (")
    with pytest.raises(ValueError):
        ExpressionEvaluator("3 / 0")
    with pytest.raises(ValueError):
        ExpressionEvaluator("3 + 4 * 2 +")
    with pytest.raises(ValueError):
        ExpressionEvaluator("invalid")