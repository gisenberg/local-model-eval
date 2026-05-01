from typing import Optional


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary minus/plus (e.g., '-3', '-(2+1)')
    - Floating-point numbers (e.g., '3.14', '.5', '2.')
    
    Raises ValueError for empty expressions, invalid tokens, 
    mismatched parentheses, and division by zero.
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

        self.text = expr
        self.pos = 0
        self.length = len(expr)

        result = self._parse_expression()

        self._skip_whitespace()
        if self.pos < self.length:
            raise ValueError(f"Invalid token at position {self.pos}: '{self.text[self.pos]}'")

        return result

    def _skip_whitespace(self) -> None:
        """Advance position past any whitespace characters."""
        while self.pos < self.length and self.text[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> Optional[str]:
        """Return the next non-whitespace character without advancing, or None if at EOF."""
        self._skip_whitespace()
        if self.pos >= self.length:
            return None
        return self.text[self.pos]

    def _advance(self) -> str:
        """Consume and return the current character, advancing the position."""
        char = self.text[self.pos]
        self.pos += 1
        return char

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._advance()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
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
        """Parse unary operators, parentheses, and numbers (highest precedence)."""
        # Handle unary plus/minus
        if self._peek() == '-':
            self._advance()
            return -self._parse_factor()
        if self._peek() == '+':
            self._advance()
            return self._parse_factor()

        # Handle parentheses
        if self._peek() == '(':
            self._advance()
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result

        # Handle numbers
        if self._peek() is not None and (self._peek().isdigit() or self._peek() == '.'):
            return self._parse_number()

        # Invalid token or unexpected EOF
        if self._peek() is None:
            raise ValueError("Unexpected end of expression")
        raise ValueError(f"Invalid token: '{self._peek()}'")

    def _parse_number(self) -> float:
        """Parse an integer or floating-point number."""
        start = self.pos
        has_dot = False
        has_digit = False

        while self.pos < self.length:
            char = self.text[self.pos]
            if char.isdigit():
                has_digit = True
                self.pos += 1
            elif char == '.':
                if has_dot:
                    break
                has_dot = True
                self.pos += 1
            else:
                break

        if not has_digit:
            raise ValueError(f"Invalid number at position {start}")

        return float(self.text[start:self.pos])

import pytest

class TestExpressionEvaluator:
    @pytest.fixture
    def evaluator(self):
        return ExpressionEvaluator()

    def test_operator_precedence(self, evaluator):
        """Test that * and / bind tighter than + and -."""
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        assert evaluator.evaluate("10 - 2 / 2") == 9.0
        assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

    def test_parentheses_and_unary_minus(self, evaluator):
        """Test grouping and unary negation."""
        assert evaluator.evaluate("-(2 + 3)") == -5.0
        assert evaluator.evaluate("- - 3") == 3.0
        assert evaluator.evaluate("(-2) * 3 + 4") == -2.0
        assert evaluator.evaluate("-( -( 5 ) )") == 5.0

    def test_floating_point_numbers(self, evaluator):
        """Test decimal number parsing and arithmetic."""
        assert evaluator.evaluate("3.14 + 2.86") == 6.0
        assert evaluator.evaluate(".5 * 4") == 2.0
        assert evaluator.evaluate("10 / 3.0") == pytest.approx(3.3333333333333335)
        assert evaluator.evaluate("2.5 * 2.5") == 6.25

    def test_division_by_zero(self, evaluator):
        """Test that division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("5 / 0")
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("10 / (2 - 2)")

    def test_error_handling(self, evaluator):
        """Test invalid inputs, mismatched parentheses, and empty strings."""
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("   ")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Invalid token"):
            evaluator.evaluate("2 & 3")
        with pytest.raises(ValueError, match="Invalid token"):
            evaluator.evaluate("2 + 3)")