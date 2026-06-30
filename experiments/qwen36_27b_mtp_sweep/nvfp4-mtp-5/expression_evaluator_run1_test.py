import re
from typing import NoReturn


class _Parser:
    """Internal recursive descent parser for mathematical expressions."""

    def __init__(self, expr: str) -> None:
        self.expr = expr
        self.pos = 0
        self.length = len(expr)

    def skip_whitespace(self) -> None:
        """Advance position past any whitespace characters."""
        while self.pos < self.length and self.expr[self.pos].isspace():
            self.pos += 1

    def parse_number(self) -> float:
        """Parse a positive integer or floating-point number."""
        self.skip_whitespace()
        start = self.pos
        has_dot = False

        while self.pos < self.length and (self.expr[self.pos].isdigit() or self.expr[self.pos] == '.'):
            if self.expr[self.pos] == '.':
                if has_dot:
                    raise ValueError("Invalid number format")
                has_dot = True
            self.pos += 1

        if self.pos == start:
            raise ValueError("Expected number")

        num_str = self.expr[start:self.pos]
        try:
            return float(num_str)
        except ValueError:
            raise ValueError(f"Invalid number: '{num_str}'")

    def parse_factor(self) -> float:
        """Parse factors: unary minus, parentheses, or numbers."""
        self.skip_whitespace()
        if self.pos >= self.length:
            raise ValueError("Unexpected end of expression")

        char = self.expr[self.pos]

        # Unary minus
        if char == '-':
            self.pos += 1
            return -self.parse_factor()

        # Parentheses
        if char == '(':
            self.pos += 1
            result = self.parse_expression()
            self.skip_whitespace()
            if self.pos >= self.length or self.expr[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result

        # Numbers (integers and floats)
        if char.isdigit() or char == '.':
            return self.parse_number()

        raise ValueError(f"Invalid token: '{char}'")

    def parse_term(self) -> float:
        """Parse terms: multiplication and division (higher precedence)."""
        left = self.parse_factor()
        while True:
            self.skip_whitespace()
            if self.pos < self.length and self.expr[self.pos] in ('*', '/'):
                op = self.expr[self.pos]
                self.pos += 1
                right = self.parse_factor()
                if op == '*':
                    left *= right
                else:
                    if right == 0.0:
                        raise ValueError("Division by zero")
                    left /= right
            else:
                break
        return left

    def parse_expression(self) -> float:
        """Parse expressions: addition and subtraction (lowest precedence)."""
        left = self.parse_term()
        while True:
            self.skip_whitespace()
            if self.pos < self.length and self.expr[self.pos] in ('+', '-'):
                op = self.expr[self.pos]
                self.pos += 1
                right = self.parse_term()
                if op == '+':
                    left += right
                else:
                    left -= right
            else:
                break
        return left

    def is_end(self) -> bool:
        """Check if all tokens have been consumed."""
        self.skip_whitespace()
        return self.pos >= self.length


class ExpressionEvaluator:
    """Evaluates mathematical expressions using a recursive descent parser."""

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression with +, -, *, /,
                  parentheses, unary minus, and floating point numbers.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        parser = _Parser(expr)
        result = parser.parse_expression()

        if not parser.is_end():
            raise ValueError("Invalid expression: unexpected tokens after result")

        return result

import pytest

def test_basic_precedence():
    """Test correct operator precedence for +, -, *, /."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 - 2 * 3") == 4.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary():
    """Test parentheses grouping and unary minus support."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("-3 * 2") == -6.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("--5") == 5.0
    assert ev.evaluate("2 * -3") == -6.0

def test_floating_point():
    """Test floating-point number parsing and arithmetic."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate(".5 + .5") == 1.0
    assert ev.evaluate("10 / 4") == 2.5
    assert ev.evaluate("1.5 * 2.0") == 3.0

def test_division_by_zero():
    """Test that division by zero raises ValueError."""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("5 / (2 - 2)")

def test_invalid_expressions():
    """Test error handling for malformed inputs."""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + 3 @ 4")
    with pytest.raises(ValueError, match="Invalid expression"):
        ev.evaluate("2 +")