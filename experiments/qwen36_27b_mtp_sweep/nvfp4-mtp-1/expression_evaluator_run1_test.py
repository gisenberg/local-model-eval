import math

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the mathematical expression string.

        Args:
            expr: A string containing the mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is invalid, empty, or contains errors.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._expr = expr
        self._pos = 0
        self._length = len(expr)

        # Skip initial whitespace
        self._skip_whitespace()

        # If string is just whitespace (handled by check above, but safe guard)
        if self._pos >= self._length:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        # Check if there is trailing content
        self._skip_whitespace()
        if self._pos < self._length:
            # Specific check for unexpected closing parenthesis
            if self._expr[self._pos] == ')':
                raise ValueError("Mismatched parentheses")
            raise ValueError(f"Invalid token at index {self._pos}: '{self._expr[self._pos]}'")

        return result

    def _skip_whitespace(self):
        """Skips whitespace characters."""
        while self._pos < self._length and self._expr[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> str:
        """Returns the current character without consuming it."""
        if self._pos < self._length:
            return self._expr[self._pos]
        return None

    def _consume(self) -> str:
        """Consumes and returns the current character."""
        char = self._expr[self._pos]
        self._pos += 1
        return char

    def _parse_expression(self) -> float:
        """Handles addition and subtraction (lowest precedence)."""
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
        """Handles multiplication and division (medium precedence)."""
        left = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_factor(self) -> float:
        """Handles unary minus and primary expressions (highest precedence)."""
        if self._peek() == '-':
            self._consume()
            # Recursive call handles chained unary minuses like --3
            return -self._parse_factor()
        else:
            return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and parenthesized expressions."""
        self._skip_whitespace()

        char = self._peek()

        # Check for number (digit or decimal point)
        if char is not None and (char.isdigit() or char == '.'):
            return self._parse_number()

        # Check for parenthesis
        if char == '(':
            self._consume() # consume '('
            self._skip_whitespace()
            val = self._parse_expression()
            self._skip_whitespace()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._consume() # consume ')'
            return val

        # If we are here, it's an invalid token
        raise ValueError(f"Invalid token '{char}' at index {self._pos}")

    def _parse_number(self) -> float:
        """Parses a floating point number."""
        start = self._pos
        while self._pos < self._length and (self._expr[self._pos].isdigit() or self._expr[self._pos] == '.'):
            self._pos += 1

        if self._pos == start:
            # Should not happen if called from _parse_primary check
            raise ValueError("Invalid number format")

        number_str = self._expr[start:self._pos]
        try:
            return float(number_str)
        except ValueError:
            raise ValueError(f"Invalid number format: '{number_str}'")


# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    """Tests basic arithmetic and operator precedence."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 - 2 / 2") == 9.0
    assert ev.evaluate("2 * 3 + 4") == 10.0

def test_parentheses():
    """Tests grouping with parentheses."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("((2))") == 2.0
    assert ev.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    """Tests unary minus support."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("--5") == 5.0
    assert ev.evaluate("2 * -3") == -6.0

def test_floating_point():
    """Tests floating point number support."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate(".5 + .5") == 1.0
    assert ev.evaluate("10.5 / 2") == 5.25

def test_errors():
    """Tests error handling for invalid inputs."""
    ev = ExpressionEvaluator()
    
    # Empty expression
    with pytest.raises(ValueError):
        ev.evaluate("")
    with pytest.raises(ValueError):
        ev.evaluate("   ")

    # Division by zero
    with pytest.raises(ValueError):
        ev.evaluate("1 / 0")

    # Mismatched parentheses
    with pytest.raises(ValueError):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        ev.evaluate("2 + 3 )")

    # Invalid tokens
    with pytest.raises(ValueError):
        ev.evaluate("2 + a")
    with pytest.raises(ValueError):
        ev.evaluate("2 ++ 3")