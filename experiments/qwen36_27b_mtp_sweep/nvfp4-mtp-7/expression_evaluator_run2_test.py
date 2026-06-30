import pytest
from typing import NoReturn


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Binary operators: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating-point numbers (e.g., '3.14', '.5', '5.')
    
    Raises ValueError for:
    - Empty expressions
    - Mismatched parentheses
    - Division by zero
    - Invalid tokens or malformed numbers
    """

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result as a float."""
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._text = expr
        self._pos = 0
        self._skip_whitespace()

        result = self._parse_expression()
        self._skip_whitespace()

        if self._pos < len(self._text):
            raise ValueError(f"Invalid token at position {self._pos}")

        return result

    def _skip_whitespace(self) -> None:
        """Advance position past any whitespace characters."""
        while self._pos < len(self._text) and self._text[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> str:
        """Return the next non-whitespace character without consuming it."""
        self._skip_whitespace()
        return self._text[self._pos] if self._pos < len(self._text) else ''

    def _advance(self) -> str:
        """Consume and return the next non-whitespace character."""
        self._skip_whitespace()
        if self._pos < len(self._text):
            ch = self._text[self._pos]
            self._pos += 1
            return ch
        return ''

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
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
        """Parse multiplication and division (higher precedence)."""
        left = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._advance()
            right = self._parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_factor(self) -> float:
        """Parse unary minus, parentheses, and numbers (highest precedence)."""
        if self._peek() == '-':
            self._advance()
            return -self._parse_factor()
        
        if self._peek() == '(':
            self._advance()
            val = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return val
        
        return self._parse_number()

    def _parse_number(self) -> float:
        """Parse an integer or floating-point number."""
        start = self._pos
        num_str = ""
        has_dot = False

        while self._pos < len(self._text) and (self._text[self._pos].isdigit() or self._text[self._pos] == '.'):
            if self._text[self._pos] == '.':
                if has_dot:
                    break  # Stop at second decimal point
                has_dot = True
            num_str += self._text[self._pos]
            self._pos += 1

        if not num_str:
            raise ValueError(f"Invalid token at position {self._pos}")

        try:
            return float(num_str)
        except ValueError:
            raise ValueError(f"Invalid number format: {num_str}")


# =============================================================================
# Pytest Test Suite
# =============================================================================

def test_basic_arithmetic_and_precedence() -> None:
    """Test standard operators and correct precedence handling."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 / 2 - 1") == 4.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0
    assert ev.evaluate("1 + 2 + 3 + 4") == 10.0


def test_parentheses_and_unary_minus() -> None:
    """Test grouping and unary minus operator."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("-3 * 2") == -6.0
    assert ev.evaluate("((5))") == 5.0
    assert ev.evaluate("-(-2)") == 2.0
    assert ev.evaluate("10 - (2 * 3)") == 4.0


def test_floating_point_numbers() -> None:
    """Test decimal number parsing and arithmetic."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == 6.28
    assert ev.evaluate(".5 + .5") == 1.0
    assert ev.evaluate("5. / 2") == 2.5
    assert ev.evaluate("0.1 + 0.2") == pytest.approx(0.3, abs=1e-10)


def test_division_by_zero_and_mismatched_parens() -> None:
    """Test runtime and structural error handling."""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("5 / (2 - 2)")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("2 + 3)")


def test_invalid_tokens_and_empty_expressions() -> None:
    """Test lexical and input validation errors."""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 @ 3")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("10 +")
    with pytest.raises(ValueError, match="Invalid number format"):
        ev.evaluate("3..14")