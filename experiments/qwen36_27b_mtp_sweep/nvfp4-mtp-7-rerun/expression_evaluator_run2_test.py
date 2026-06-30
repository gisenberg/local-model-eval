import re
from typing import Optional

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Binary operators: +, -, *, / with correct precedence
    - Parentheses for grouping
    - Unary plus/minus (e.g., '-3', '-(2+1)')
    - Floating point numbers (e.g., '3.14', '.5')
    
    Raises ValueError for:
    - Empty expressions
    - Invalid tokens
    - Mismatched parentheses
    - Division by zero
    """
    
    def __init__(self) -> None:
        pass

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.expr = expr
        self.pos = 0
        self.token: Optional[str] = None
        self.current_value: float = 0.0

        self._advance()
        result = self._parse_expr()

        if self.token is not None:
            raise ValueError("Invalid token or mismatched parentheses")

        return result

    def _advance(self) -> None:
        """Skip whitespace and load the next token."""
        while self.pos < len(self.expr) and self.expr[self.pos].isspace():
            self.pos += 1

        if self.pos >= len(self.expr):
            self.token = None
            return

        char = self.expr[self.pos]
        if char in '+-*/()':
            self.token = char
            self.pos += 1
        elif char.isdigit() or char == '.':
            start = self.pos
            while self.pos < len(self.expr) and (self.expr[self.pos].isdigit() or self.expr[self.pos] == '.'):
                self.pos += 1
            self.token = 'NUMBER'
            try:
                self.current_value = float(self.expr[start:self.pos])
            except ValueError:
                raise ValueError(f"Invalid number format: {self.expr[start:self.pos]}")
        else:
            raise ValueError(f"Invalid token: {char}")

    def _parse_expr(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        while self.token in {'+', '-'}:
            op = self.token
            self._advance()
            right = self._parse_term()
            if op == '+':
                left += right
            else:
                left -= right
        return left

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        left = self._parse_factor()
        while self.token in {'*', '/'}:
            op = self.token
            self._advance()
            right = self._parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_factor(self) -> float:
        """Parse unary plus/minus and primary expressions."""
        if self.token == '+':
            self._advance()
            return self._parse_factor()
        if self.token == '-':
            self._advance()
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        if self.token == '(':
            self._advance()
            result = self._parse_expr()
            if self.token != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        if self.token == 'NUMBER':
            val = self.current_value
            self._advance()
            return val
        raise ValueError(f"Invalid token: {self.token}")

import pytest

def test_basic_precedence():
    """Test correct operator precedence for +, -, *, /"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 / 2 - 3") == 2.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping():
    """Test parentheses override default precedence"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert ev.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

def test_unary_minus():
    """Test unary minus support in various positions"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("2 * -3 + 4") == -2.0
    assert ev.evaluate("- -5") == 5.0

def test_floating_point_numbers():
    """Test floating point parsing and arithmetic"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate(".5 + .5") == 1.0
    assert ev.evaluate("1.5 / 3") == pytest.approx(0.5)

def test_error_handling():
    """Test all required ValueError conditions"""
    ev = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
        
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + abc")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 @ 3")
        
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("5 / 0.0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("2 + 3)")