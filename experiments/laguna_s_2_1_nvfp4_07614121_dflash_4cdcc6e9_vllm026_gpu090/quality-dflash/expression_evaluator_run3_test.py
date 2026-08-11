import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic operations,
    parentheses, unary minus, and floating-point numbers.

    Supported operators:
        + : Addition
        - : Subtraction / Unary Minus
        * : Multiplication
        / : Division

    The parser uses recursive descent to handle operator precedence and associativity.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0

    def tokenize(self, expr: str) -> List[str]:
        """
        Tokenizes the input string into a list of numbers, operators, and parentheses.

        Args:
            expr: The mathematical expression string.

        Returns:
            A list of tokens.

        Raises:
            ValueError: If an invalid token is encountered.
        """
        # Regular expression to match valid tokens
        token_pattern = r'(\d+\.?\d*|\.\d+|[-+*/()])'
        raw_tokens = re.findall(token_pattern, expr.replace(' ', ''))

        # Check for any invalid characters
        if not re.fullmatch(r'[\d+\-*/().\s]*', expr):
            raise ValueError(f"Invalid character in expression: {expr}")

        tokens = []
        for token in raw_tokens:
            tokens.append(token)
        return tokens

    def peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> str:
        """Consumes and returns the current token."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse_expression(self) -> float:
        """
        Parses an expression. Handles addition and subtraction.
        Expression -> Term (('+' | '-') Term)*
        """
        result = self.parse_term()
        while self.peek() in ('+', '-'):
            op = self.consume()
            right = self.parse_term()
            if op == '+':
                result += right
            elif op == '-':
                result -= right
        return result

    def parse_term(self) -> float:
        """
        Parses a term. Handles multiplication and division.
        Term -> Factor (('*' | '/') Factor)*
        """
        result = self.parse_factor()
        while self.peek() in ('*', '/'):
            op = self.consume()
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
        Parses a factor. Handles numbers, unary minus, and parentheses.
        Factor -> Number | '-' Factor | '(' Expression ')'
        """
        token = self.peek()
        if token == '-':
            self.consume()
            return -self.parse_factor()
        elif token == '(':
            self.consume()
            result = self.parse_expression()
            if self.consume() != ')':
                raise ValueError("Mismatched parentheses")
            return result
        else:
            # Must be a number
            try:
                num_token = self.consume()
                return float(num_token)
            except ValueError:
                raise ValueError(f"Invalid token: {token}")

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression.

        Args:
            expr: The mathematical expression string.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: For various parsing errors including mismatched parentheses,
                        division by zero, invalid tokens, and empty expressions.
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self.tokenize(expr)
        self.pos = 0

        try:
            result = self.parse_expression()
            # Ensure all tokens were consumed
            if self.pos != len(self.tokens):
                raise ValueError("Unexpected token at end of expression")
            return result
        except IndexError:
            raise ValueError("Unexpected end of expression")

import pytest

def test_basic_operations():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 10.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 + -2") == 1.0

def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 + 2.86") == pytest.approx(6.0)
    assert evaluator.evaluate("1.5 * 2.0") == pytest.approx(3.0)

def test_parentheses_and_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert evaluator.evaluate("2 * (3 + 4) - 1") == 13.0

def test_errors():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="invalid token"):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError, match="empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="empty expression"):
        evaluator.evaluate("   ")