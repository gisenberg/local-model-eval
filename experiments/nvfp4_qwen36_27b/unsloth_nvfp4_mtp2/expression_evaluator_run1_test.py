from typing import Optional
import pytest


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating-point numbers.
    """
    def __init__(self) -> None:
        self._expr: str = ""
        self._pos: int = 0
        self._len: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: The mathematical expression to evaluate.

        Returns:
            The result of the evaluation.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._expr = expr
        self._pos = 0
        self._len = len(expr)

        result = self._parse_expression()
        self._skip_whitespace()

        if self._pos < self._len:
            raise ValueError(f"Invalid token at position {self._pos}: '{self._expr[self._pos]}'")

        return result

    def _skip_whitespace(self) -> None:
        """Advance position past any whitespace characters."""
        while self._pos < self._len and self._expr[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> Optional[str]:
        """Return the next non-whitespace character without consuming it."""
        self._skip_whitespace()
        if self._pos < self._len:
            return self._expr[self._pos]
        return None

    def _advance(self) -> str:
        """Consume and return the next non-whitespace character."""
        self._skip_whitespace()
        if self._pos >= self._len:
            raise ValueError("Unexpected end of expression")
        char = self._expr[self._pos]
        self._pos += 1
        return char

    def _parse_number(self) -> float:
        """Parse a floating-point or integer literal."""
        self._skip_whitespace()
        start = self._pos
        has_dot = False
        has_digit = False

        while self._pos < self._len and (self._expr[self._pos].isdigit() or self._expr[self._pos] == '.'):
            if self._expr[self._pos] == '.':
                if has_dot:
                    raise ValueError(f"Invalid number format at position {start}")
                has_dot = True
            else:
                has_digit = True
            self._pos += 1

        if start == self._pos or not has_digit:
            raise ValueError(f"Expected number at position {self._pos}")

        num_str = self._expr[start:self._pos]
        try:
            return float(num_str)
        except ValueError:
            raise ValueError(f"Invalid number format: '{num_str}'")

    def _parse_primary(self) -> float:
        """Parse primary expressions: numbers or parenthesized expressions."""
        char = self._peek()
        if char is None:
            raise ValueError("Unexpected end of expression")

        if char == '(':
            self._advance()
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._advance()
            return result
        elif char == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")
        elif char.isdigit() or char == '.':
            return self._parse_number()
        else:
            raise ValueError(f"Invalid token: '{char}'")

    def _parse_unary(self) -> float:
        """Parse unary operators: + and -."""
        char = self._peek()
        if char == '-':
            self._advance()
            return -self._parse_unary()
        elif char == '+':
            self._advance()
            return self._parse_unary()
        else:
            return self._parse_primary()

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_unary()
        while True:
            char = self._peek()
            if char == '*':
                self._advance()
                right = self._parse_unary()
                result *= right
            elif char == '/':
                self._advance()
                right = self._parse_unary()
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
            else:
                break
        return result

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lower precedence)."""
        result = self._parse_term()
        while True:
            char = self._peek()
            if char == '+':
                self._advance()
                result += self._parse_term()
            elif char == '-':
                self._advance()
                result -= self._parse_term()
            else:
                break
        return result


# ========================
# Pytest Test Suite
# ========================

def test_basic_arithmetic_precedence():
    """Test correct operator precedence for +, -, *, /"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("3 + 4 * 2") == 11.0
    assert ev.evaluate("10 - 2 / 2") == 9.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping():
    """Test parentheses override default precedence"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(3 + 4) * 2") == 14.0
    assert ev.evaluate("((2 + 3) * (4 - 1))") == 15.0
    assert ev.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

def test_unary_minus():
    """Test unary minus in various positions"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("- - 5") == 5.0
    assert ev.evaluate("3 * -2") == -6.0

def test_floating_point_numbers():
    """Test parsing and arithmetic with floating-point literals"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate(".5 + .5") == 1.0
    assert ev.evaluate("1.5 / 3") == pytest.approx(0.5)

def test_error_handling():
    """Test ValueError raises for all specified error conditions"""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(3 + 4")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("3 + a")