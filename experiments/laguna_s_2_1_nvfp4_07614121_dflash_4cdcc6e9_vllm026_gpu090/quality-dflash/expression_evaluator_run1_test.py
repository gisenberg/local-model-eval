from typing import List, Optional

class ExpressionEvaluator:
    """
    A recursive descent parser-based expression evaluator that supports
    basic arithmetic operations, parentheses, unary minus, and floats.

    Supported operators: +, -, *, /
    Supported features: parentheses, unary minus, floating point numbers
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0

    def tokenize(self, expr: str) -> List[str]:
        """
        Tokenizes the input string into a list of tokens.

        Args:
            expr (str): The input expression.

        Returns:
            List[str]: List of tokens.

        Raises:
            ValueError: If an invalid token is found.
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        tokens: List[str] = []
        i = 0
        while i < len(expr):
            char = expr[i]

            if char.isspace():
                i += 1
                continue

            if char in "+-*/()":
                tokens.append(char)
                i += 1
            elif char.isdigit() or char == '.':
                num_str = ""
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    num_str += expr[i]
                    i += 1
                tokens.append(num_str)
            else:
                raise ValueError(f"Invalid token: '{char}'")

        return tokens

    def peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected: Optional[str] = None) -> str:
        """
        Consumes the next token. If `expected` is provided, it must match.

        Args:
            expected (Optional[str]): Expected token value.

        Returns:
            str: The consumed token.

        Raises:
            ValueError: If the token doesn't match the expected one or is missing.
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")

        token = self.tokens[self.pos]
        if expected is not None and token != expected:
            raise ValueError(f"Expected '{expected}', got '{token}'")

        self.pos += 1
        return token

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the given mathematical expression.

        Args:
            expr (str): The expression to evaluate.

        Returns:
            float: The result of the evaluation.

        Raises:
            ValueError: For syntax errors, division by zero, etc.
        """
        self.tokens = self.tokenize(expr)
        self.pos = 0
        result = self.parse_expression()
        if self.pos != len(self.tokens):
            raise ValueError("Unexpected tokens after expression")
        return result

    def parse_expression(self) -> float:
        """
        Parses an expression: term ((+|-) term)*
        """
        result = self.parse_term()
        while self.peek() in ('+', '-'):
            op = self.consume()
            right = self.parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def parse_term(self) -> float:
        """
        Parses a term: factor ((*|/) factor)*
        """
        result = self.parse_factor()
        while self.peek() in ('*', '/'):
            op = self.consume()
            right = self.parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def parse_factor(self) -> float:
        """
        Parses a factor: number | '(' expression ')' | '-' factor
        """
        token = self.peek()

        if token == '-':
            self.consume()
            return -self.parse_factor()

        if token == '(':
            self.consume('(')
            result = self.parse_expression()
            self.consume(')')
            return result

        if token and (token.replace('.', '', 1).isdigit()):
            self.consume()
            return float(token)

        raise ValueError(f"Unexpected token: '{token}'")

import pytest


def test_basic_operations():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0


def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0


def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 + 2.86") == pytest.approx(6.0)
    assert evaluator.evaluate("10.5 / 2") == pytest.approx(5.25)


def test_invalid_tokens():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + abc")
    with pytest.raises(ValueError):
        evaluator.evaluate("")


def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("1 / (2 - 2)")