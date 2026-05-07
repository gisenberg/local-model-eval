from __future__ import annotations


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Operators: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating-point numbers (e.g., 3.14, .5, 5.)
    
    Raises ValueError for invalid syntax, mismatched parentheses, 
    division by zero, or empty expressions.
    """

    def __init__(self) -> None:
        self.expr: str = ""
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.

        Args:
            expr: A string containing a valid mathematical expression.

        Returns:
            The numerical result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.expr = expr
        self.pos = 0
        result = self._parse_expression()

        self._skip_whitespace()
        if self.pos < len(self.expr):
            raise ValueError(f"Invalid token at position {self.pos}: '{self.expr[self.pos]}'")

        return result

    def _skip_whitespace(self) -> None:
        """Advance position past any whitespace characters."""
        while self.pos < len(self.expr) and self.expr[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str:
        """Return the current non-whitespace character without consuming it."""
        self._skip_whitespace()
        return self.expr[self.pos] if self.pos < len(self.expr) else ''

    def _advance(self) -> str:
        """Consume and return the current non-whitespace character."""
        self._skip_whitespace()
        if self.pos >= len(self.expr):
            raise ValueError("Unexpected end of expression")
        char = self.expr[self.pos]
        self.pos += 1
        return char

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._advance()
            right = self._parse_term()
            result = result + right if op == '+' else result - right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._advance()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary minus, parentheses, and numbers (highest precedence)."""
        # Handle unary minus
        if self._peek() == '-':
            self._advance()
            return -self._parse_factor()

        # Handle parentheses
        if self._peek() == '(':
            self._advance()
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result

        # Handle numbers
        if self._peek().isdigit() or self._peek() == '.':
            return self._parse_number()

        raise ValueError(f"Invalid token: '{self._peek()}'")

    def _parse_number(self) -> float:
        """Parse an integer or floating-point number."""
        start = self.pos
        has_dot = False
        while self.pos < len(self.expr) and (self.expr[self.pos].isdigit() or self.expr[self.pos] == '.'):
            if self.expr[self.pos] == '.':
                if has_dot:
                    raise ValueError("Invalid number format")
                has_dot = True
            self.pos += 1

        if start == self.pos:
            raise ValueError("Expected number")

        try:
            return float(self.expr[start:self.pos])
        except ValueError as e:
            raise ValueError(f"Invalid number format: '{self.expr[start:self.pos]}'") from e

import pytest


def test_basic_precedence():
    """Test correct operator precedence for +, -, *, /"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 / 2 - 3") == 2.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0


def test_parentheses():
    """Test grouping with parentheses"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert ev.evaluate("10 / (2 + 3)") == pytest.approx(2.0)


def test_unary_minus():
    """Test unary minus in various positions"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("2 * -3 + 4") == -2.0
    assert ev.evaluate("--5") == 5.0  # Double negation


def test_floating_point():
    """Test parsing and arithmetic with floating-point numbers"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate(".5 + 1.5") == 2.0
    assert ev.evaluate("10.0 / 4.0") == 2.5


def test_error_handling():
    """Test ValueError raises for invalid inputs"""
    ev = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
        
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("5 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
        
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + a")
        
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + 3)")