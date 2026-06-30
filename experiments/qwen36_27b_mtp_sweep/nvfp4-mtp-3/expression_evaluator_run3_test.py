from typing import Optional


class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    
    Supports:
    - Binary operators: +, -, *, / with correct precedence
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers (e.g., '3.14')
    
    Raises ValueError for:
    - Empty expressions
    - Mismatched parentheses
    - Division by zero
    - Invalid tokens
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The numerical result of the expression.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._expr = expr
        self._pos = 0
        self._skip_whitespace()

        result = self._parse_expression()
        self._skip_whitespace()

        if self._pos < len(self._expr):
            raise ValueError(f"Invalid token at position {self._pos}")

        return result

    def _skip_whitespace(self) -> None:
        """Advance position past any whitespace characters."""
        while self._pos < len(self._expr) and self._expr[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> Optional[str]:
        """Return the next non-whitespace character without consuming it."""
        self._skip_whitespace()
        if self._pos < len(self._expr):
            return self._expr[self._pos]
        return None

    def _consume(self, expected: Optional[str] = None) -> str:
        """
        Consume and return the next character.
        If expected is provided, verify the character matches.
        """
        self._skip_whitespace()
        if self._pos >= len(self._expr):
            raise ValueError("Unexpected end of expression")
        char = self._expr[self._pos]
        if expected is not None and char != expected:
            raise ValueError(f"Expected '{expected}', got '{char}'")
        self._pos += 1
        return char

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                left += right
            else:
                left -= right
        return left

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        left = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_factor(self) -> float:
        """Parse unary minus, parentheses, and numeric literals."""
        if self._peek() == '-':
            self._consume()
            return -self._parse_factor()
        if self._peek() == '(':
            self._consume('(')
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._consume(')')
            return result
        return self._parse_number()

    def _parse_number(self) -> float:
        """Parse an integer or floating point number."""
        start = self._pos
        has_dot = False
        while self._pos < len(self._expr) and (self._expr[self._pos].isdigit() or self._expr[self._pos] == '.'):
            if self._expr[self._pos] == '.':
                if has_dot:
                    raise ValueError("Invalid number format")
                has_dot = True
            self._pos += 1
        if self._pos == start:
            raise ValueError(f"Invalid token at position {self._pos}")
        return float(self._expr[start:self._pos])

import pytest


def test_operator_precedence():
    """Test that * and / bind tighter than + and -."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 - 2 / 2") == 9.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0


def test_parentheses_and_unary_minus():
    """Test grouping and unary minus operator."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-(2 + 3)") == -5.0
    assert ev.evaluate("-3 * 2") == -6.0
    assert ev.evaluate("-(2 * (3 + 1))") == -8.0
    assert ev.evaluate("--5") == 5.0


def test_floating_point_numbers():
    """Test support for decimal numbers."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate("1.5 + 2.5") == 4.0
    assert ev.evaluate(".5 * 4") == 2.0


def test_division_by_zero():
    """Test that division by zero raises ValueError."""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("5 / (2 - 2)")


def test_invalid_expressions():
    """Test error handling for invalid inputs."""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("1 + 2)")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("1 + a")