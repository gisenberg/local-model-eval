from typing import Union


class ExpressionEvaluator:
    """
    A recursive descent parser for mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating-point numbers.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        self._expr = expr
        self._pos = 0
        self._length = len(self._expr)

        result = self._parse_expression()

        # Ensure entire expression was consumed (ignoring trailing whitespace)
        if self._pos < self._length:
            if not self._expr[self._pos:].strip():
                return result
            raise ValueError(f"Invalid token at position {self._pos}")

        return result

    def _skip_whitespace(self) -> None:
        """Skips whitespace characters in the expression."""
        while self._pos < self._length and self._expr[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> str:
        """Returns the current character without advancing the position."""
        self._skip_whitespace()
        if self._pos < self._length:
            return self._expr[self._pos]
        return ''

    def _advance(self) -> str:
        """Advances the position by one character and returns it."""
        char = self._peek()
        self._pos += 1
        return char

    def _parse_expression(self) -> float:
        """Parses addition and subtraction operations (lowest precedence)."""
        left = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._advance()
            right = self._parse_term()
            if op == '+':
                left += right
            else:
                left -= right
        return left

    def _parse_term(self) -> float:
        """Parses multiplication and division operations (higher precedence)."""
        left = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._advance()
            right = self._parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_factor(self) -> float:
        """Parses unary minus and primary expressions."""
        self._skip_whitespace()
        if self._peek() == '-':
            self._advance()
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parses numbers and parenthesized expressions."""
        self._skip_whitespace()
        char = self._peek()

        if char == '':
            raise ValueError("Unexpected end of expression")

        if char == '(':
            self._advance()
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()  # consume ')'
            return result

        if char.isdigit() or char == '.':
            return self._parse_number()

        raise ValueError(f"Invalid token: '{char}' at position {self._pos}")

    def _parse_number(self) -> float:
        """Parses a floating-point number from the expression."""
        start = self._pos
        has_dot = False
        while self._pos < self._length:
            char = self._expr[self._pos]
            if char.isdigit():
                self._pos += 1
            elif char == '.':
                if has_dot:
                    break
                has_dot = True
                self._pos += 1
            else:
                break

        if self._pos == start:
            raise ValueError(f"Invalid number format at position {self._pos}")

        num_str = self._expr[start:self._pos]
        try:
            return float(num_str)
        except ValueError:
            raise ValueError(f"Invalid number format: '{num_str}'")

import pytest

class TestExpressionEvaluator:
    def test_basic_precedence(self) -> None:
        """Tests correct operator precedence for +, -, *, /."""
        ev = ExpressionEvaluator()
        assert ev.evaluate("2 + 3 * 4") == 14.0
        assert ev.evaluate("10 / 2 - 3") == 2.0
        assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

    def test_parentheses_grouping(self) -> None:
        """Tests parentheses override default precedence."""
        ev = ExpressionEvaluator()
        assert ev.evaluate("(2 + 3) * 4") == 20.0
        assert ev.evaluate("((1 + 2) * (3 + 4))") == 21.0
        assert ev.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

    def test_unary_minus(self) -> None:
        """Tests unary minus in various contexts."""
        ev = ExpressionEvaluator()
        assert ev.evaluate("-3") == -3.0
        assert ev.evaluate("-(2 + 1)") == -3.0
        assert ev.evaluate("- - 5") == 5.0
        assert ev.evaluate("2 * -3 + 4") == -2.0

    def test_floating_point(self) -> None:
        """Tests floating-point number parsing and arithmetic."""
        ev = ExpressionEvaluator()
        assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
        assert ev.evaluate(".5 + .5") == 1.0
        assert ev.evaluate("10.0 / 4.0") == 2.5

    def test_error_handling(self) -> None:
        """Tests ValueError raising for invalid inputs."""
        ev = ExpressionEvaluator()
        
        with pytest.raises(ValueError, match="Division by zero"):
            ev.evaluate("1 / 0")
            
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            ev.evaluate("(2 + 3")
            
        with pytest.raises(ValueError, match="Invalid token"):
            ev.evaluate("2 + a")
            
        with pytest.raises(ValueError, match="empty"):
            ev.evaluate("")