from typing import Union, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)', '--3')
    - Floating point numbers (e.g., '3.14', '.5')
    
    Raises ValueError for:
    - Empty or whitespace-only expressions
    - Invalid tokens
    - Mismatched parentheses
    - Division by zero
    """

    def __init__(self) -> None:
        self._expr: str = ""
        self._pos: int = 0
        self._token: Optional[Union[str, float]] = None

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._expr = expr
        self._pos = 0
        self._token = None
        self._next_token()

        result = self._parse_expression()

        if self._token != "EOF":
            raise ValueError("Invalid token or mismatched parentheses")

        return result

    def _next_token(self) -> None:
        """Advance the parser to the next token in the expression."""
        # Skip whitespace
        while self._pos < len(self._expr) and self._expr[self._pos].isspace():
            self._pos += 1

        if self._pos >= len(self._expr):
            self._token = "EOF"
            return

        char = self._expr[self._pos]

        if char in "+-*/()":
            self._token = char
            self._pos += 1
        elif char.isdigit() or char == '.':
            start = self._pos
            while self._pos < len(self._expr) and (self._expr[self._pos].isdigit() or self._expr[self._pos] == '.'):
                self._pos += 1
            num_str = self._expr[start:self._pos]
            
            if not num_str or num_str.count('.') > 1:
                raise ValueError(f"Invalid number format: {num_str}")
            try:
                self._token = float(num_str)
            except ValueError:
                raise ValueError(f"Invalid number format: {num_str}")
        else:
            raise ValueError(f"Invalid token: {char}")

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        while self._token in ('+', '-'):
            op = self._token
            self._next_token()
            right = self._parse_term()
            if op == '+':
                left += right
            else:
                left -= right
        return left

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        left = self._parse_factor()
        while self._token in ('*', '/'):
            op = self._token
            self._next_token()
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
        if self._token == '-':
            self._next_token()
            return -self._parse_factor()
        if self._token == '+':
            self._next_token()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions (highest precedence)."""
        if isinstance(self._token, float):
            val = self._token
            self._next_token()
            return val
        if self._token == '(':
            self._token = None  # Consume '('
            self._next_token()
            val = self._parse_expression()
            if self._token != ')':
                raise ValueError("Mismatched parentheses")
            self._next_token()
            return val
        raise ValueError("Invalid token or mismatched parentheses")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence_and_parentheses(evaluator):
    """Test correct precedence of * / over + - and grouping with parentheses."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0
    assert evaluator.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

def test_unary_minus(evaluator):
    """Test unary minus in various contexts."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("- - 3") == 3.0
    assert evaluator.evaluate("5 * -2") == -10.0

def test_floating_point_numbers(evaluator):
    """Test support for decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("1.5 / 3") == pytest.approx(0.5)
    assert evaluator.evaluate("-2.5 + 1.5") == -1.0

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("0 / (2 - 2)")

def test_error_handling(evaluator):
    """Test ValueError for invalid inputs."""
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
        
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")
        
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Invalid token or mismatched parentheses"):
        evaluator.evaluate("2 + )")