from typing import Optional


class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    
    Supports:
    - Operators: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers (e.g., '3.14', '.5')
    
    Raises ValueError for empty expressions, mismatched parentheses, 
    division by zero, and invalid tokens.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.
        
        Args:
            expr: A string containing a mathematical expression.
            
        Returns:
            The numerical result of the expression.
            
        Raises:
            ValueError: If the expression is empty, contains mismatched parentheses,
                        division by zero, or invalid tokens.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        self._expr = expr
        self._pos = 0
        self._skip_whitespace()

        result = self._parse_expression()

        self._skip_whitespace()
        if self._pos < len(self._expr):
            raise ValueError(f"Invalid token at position {self._pos}: '{self._expr[self._pos]}'")

        return result

    def _skip_whitespace(self) -> None:
        """Advance position past any whitespace characters."""
        while self._pos < len(self._expr) and self._expr[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> Optional[str]:
        """Return the current character without consuming it, or None if at EOF."""
        if self._pos < len(self._expr):
            return self._expr[self._pos]
        return None

    def _consume(self, expected: Optional[str] = None) -> str:
        """Consume and return the current character. Optionally validate it."""
        char = self._peek()
        if char is None:
            raise ValueError("Unexpected end of expression")
        if expected is not None and char != expected:
            raise ValueError(f"Expected '{expected}' but found '{char}' at position {self._pos}")
        self._pos += 1
        return char

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        self._skip_whitespace()
        left = self._parse_term()
        while True:
            self._skip_whitespace()
            if self._peek() in ('+', '-'):
                op = self._consume()
                right = self._parse_term()
                left = left + right if op == '+' else left - right
            else:
                break
        return left

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        self._skip_whitespace()
        left = self._parse_factor()
        while True:
            self._skip_whitespace()
            if self._peek() in ('*', '/'):
                op = self._consume()
                right = self._parse_factor()
                if op == '*':
                    left *= right
                else:
                    if right == 0.0:
                        raise ValueError("Division by zero")
                    left /= right
            else:
                break
        return left

    def _parse_factor(self) -> float:
        """Parse unary minus, parentheses, and numbers (highest precedence)."""
        self._skip_whitespace()
        
        # Handle unary minus
        if self._peek() == '-':
            self._consume()
            return -self._parse_factor()
            
        # Handle parentheses
        if self._peek() == '(':
            self._consume()
            result = self._parse_expression()
            self._skip_whitespace()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()
            return result
            
        # Handle numbers
        return self._parse_number()

    def _parse_number(self) -> float:
        """Parse an integer or floating point number."""
        self._skip_whitespace()
        start = self._pos
        has_dot = False
        
        while self._pos < len(self._expr) and (self._expr[self._pos].isdigit() or self._expr[self._pos] == '.'):
            if self._expr[self._pos] == '.':
                if has_dot:
                    raise ValueError(f"Invalid number format at position {start}")
                has_dot = True
            self._pos += 1
            
        if self._pos == start:
            raise ValueError(f"Expected number at position {start}")
            
        return float(self._expr[start:self._pos])

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / bind tighter than + and -"""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping and unary minus handling"""
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("(-2) * 3") == -6.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("- - - 3") == -3.0

def test_floating_point_numbers(evaluator):
    """Test decimal number parsing and arithmetic"""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("1.5 / 3") == pytest.approx(0.5)

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError"""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0.0")

def test_invalid_expressions(evaluator):
    """Test error handling for malformed inputs"""
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(3 + 4")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 @ 4")
    with pytest.raises(ValueError, match="Expected number"):
        evaluator.evaluate("3 +")