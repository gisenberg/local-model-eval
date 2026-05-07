import re
from typing import Iterator, Tuple, Union
from __future__ import annotations

Token = Tuple[str, Union[float, str]]

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14, .5)
    
    Raises ValueError for invalid input, mismatched parentheses, 
    division by zero, or empty expressions.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        parser = self._Parser(tokens)
        result = parser.parse_expression()

        if parser.current_token[0] != 'EOF':
            raise ValueError(f"Unexpected token after expression: {parser.current_token}")

        return result

    def _tokenize(self, expr: str) -> Iterator[Token]:
        """Convert expression string into an iterator of tokens."""
        # Matches optional whitespace followed by a number or operator/parenthesis
        pattern = re.compile(r'\s*(\d+\.\d+|\d+|\.\d+|[+\-*/()])')
        pos = 0
        while pos < len(expr):
            match = pattern.match(expr, pos)
            if not match:
                raise ValueError(f"Invalid character at position {pos}: '{expr[pos]}'")
            
            token_str = match.group(1)
            pos = match.end()
            
            if token_str in '+-*/()':
                yield (token_str, token_str)
            else:
                yield ('NUM', float(token_str))
        yield ('EOF', None)

    class _Parser:
        """Internal recursive descent parser."""

        def __init__(self, tokens: Iterator[Token]):
            self.tokens = tokens
            self.current_token: Token = next(self.tokens)

        def eat(self, token_type: str) -> None:
            """Consume the current token if it matches the expected type."""
            if self.current_token[0] == token_type:
                self.current_token = next(self.tokens)
            else:
                raise ValueError(f"Expected {token_type}, got {self.current_token[0]}")

        def parse_expression(self) -> float:
            """Handle addition and subtraction (lowest precedence)."""
            result = self.parse_term()
            while self.current_token[0] in ('+', '-'):
                op = self.current_token[0]
                self.eat(op)
                right = self.parse_term()
                result = result + right if op == '+' else result - right
            return result

        def parse_term(self) -> float:
            """Handle multiplication and division (higher precedence)."""
            result = self.parse_factor()
            while self.current_token[0] in ('*', '/'):
                op = self.current_token[0]
                self.eat(op)
                right = self.parse_factor()
                if op == '*':
                    result *= right
                else:
                    if right == 0:
                        raise ValueError("Division by zero")
                    result /= right
            return result

        def parse_factor(self) -> float:
            """Handle numbers, parentheses, and unary operators."""
            token = self.current_token
            if token[0] == 'NUM':
                self.eat('NUM')
                return token[1]
            elif token[0] == '(':
                self.eat('(')
                result = self.parse_expression()
                self.eat(')')  # Raises ValueError if mismatched
                return result
            elif token[0] == '-':
                self.eat('-')
                return -self.parse_factor()
            elif token[0] == '+':
                self.eat('+')
                return self.parse_factor()
            else:
                raise ValueError(f"Unexpected token: {token}")

import pytest

class TestExpressionEvaluator:
    @pytest.fixture
    def evaluator(self):
        return ExpressionEvaluator()

    def test_operator_precedence(self, evaluator):
        """Test that * and / are evaluated before + and -"""
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        assert evaluator.evaluate("10 / 2 - 3") == 2.0
        assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

    def test_parentheses_grouping(self, evaluator):
        """Test that parentheses override default precedence"""
        assert evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert evaluator.evaluate("2 * (3 + 4) / 2") == 7.0
        assert evaluator.evaluate("((10 - 2) / 4) + 1") == 3.0

    def test_unary_minus(self, evaluator):
        """Test unary minus support for numbers and grouped expressions"""
        assert evaluator.evaluate("-3") == -3.0
        assert evaluator.evaluate("-(2 + 1)") == -3.0
        assert evaluator.evaluate("- - 5") == 5.0
        assert evaluator.evaluate("10 + -3 * 2") == 4.0

    def test_floating_point_numbers(self, evaluator):
        """Test support for decimal and leading-dot floats"""
        assert evaluator.evaluate("3.14 + 2.86") == pytest.approx(6.0)
        assert evaluator.evaluate(".5 * 4") == 2.0
        assert evaluator.evaluate("1.5 / 0.3") == pytest.approx(5.0)

    def test_error_handling(self, evaluator):
        """Test ValueError raising for invalid inputs"""
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("   ")
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("10 / 0")
        with pytest.raises(ValueError, match="Invalid character"):
            evaluator.evaluate("2 @ 3")
        with pytest.raises(ValueError, match="Expected"):
            evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluator.evaluate("2 + 3)")