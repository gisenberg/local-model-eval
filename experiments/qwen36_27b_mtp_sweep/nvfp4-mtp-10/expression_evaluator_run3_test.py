class ExpressionEvaluator:
    """A recursive descent parser for evaluating mathematical expressions.
    
    Supports: +, -, *, / with standard precedence, parentheses, unary minus,
    and floating-point numbers. Raises ValueError for invalid inputs.
    """

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result as a float.
        
        Args:
            expr: A string containing a mathematical expression.
            
        Returns:
            The numerical result of the expression.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        self._expr = expr
        self._pos = 0
        self._len = len(expr)
        
        self._skip_whitespace()
        if self._pos == self._len:
            raise ValueError("Empty expression")
            
        result = self._parse_expression()
        self._skip_whitespace()
        
        if self._pos != self._len:
            raise ValueError("Invalid token or mismatched parentheses")
            
        return result

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            left = left + right if op == '+' else left - right
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
        elif self._peek() == '(':
            self._consume()
            val = self._parse_expression()
            self._skip_whitespace()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()
            return val
        else:
            return self._parse_number()

    def _parse_number(self) -> float:
        """Parse an integer or floating-point number."""
        start = self._pos
        has_dot = False
        
        while self._pos < self._len and (self._expr[self._pos].isdigit() or self._expr[self._pos] == '.'):
            if self._expr[self._pos] == '.':
                if has_dot:
                    raise ValueError("Invalid number format")
                has_dot = True
            self._pos += 1
            
        if self._pos == start:
            raise ValueError("Invalid token")
            
        return float(self._expr[start:self._pos])

    def _peek(self) -> str:
        """Return the next non-whitespace character without consuming it."""
        self._skip_whitespace()
        return self._expr[self._pos] if self._pos < self._len else ''

    def _consume(self) -> str:
        """Consume and return the next non-whitespace character."""
        self._skip_whitespace()
        if self._pos >= self._len:
            raise ValueError("Unexpected end of expression")
        char = self._expr[self._pos]
        self._pos += 1
        return char

    def _skip_whitespace(self) -> None:
        """Advance position past any whitespace characters."""
        while self._pos < self._len and self._expr[self._pos].isspace():
            self._pos += 1

import pytest

def test_operator_precedence():
    """Test that * and / bind tighter than + and -."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 / 2 - 1") == 4.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus():
    """Test grouping and unary minus handling."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-(2 + 3)") == -5.0
    assert ev.evaluate("-3.14") == -3.14
    assert ev.evaluate("2 * (3 + 4)") == 14.0
    assert ev.evaluate("-(-5)") == 5.0

def test_floating_point_support():
    """Test decimal number parsing and arithmetic."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("1.5 + 2.5") == 4.0
    assert ev.evaluate("3.14 * 2") == 6.28
    assert ev.evaluate(".5 + .5") == 1.0

def test_error_handling():
    """Test all required ValueError conditions."""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError):
        ev.evaluate("")  # Empty expression
    with pytest.raises(ValueError):
        ev.evaluate("   ")  # Whitespace only
    with pytest.raises(ValueError):
        ev.evaluate("(3 + 2")  # Mismatched parentheses
    with pytest.raises(ValueError):
        ev.evaluate("5 / 0")  # Division by zero
    with pytest.raises(ValueError):
        ev.evaluate("3 + 2a")  # Invalid token
    with pytest.raises(ValueError):
        ev.evaluate("1.2.3")  # Malformed number

def test_complex_expressions():
    """Test nested operations and edge cases."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("((2 + 3) * 4 - 1) / 3") == 6.0
    assert ev.evaluate("-(-5) + 2.5") == 7.5
    assert ev.evaluate("  10  /  2  ") == 5.0  # Whitespace tolerance