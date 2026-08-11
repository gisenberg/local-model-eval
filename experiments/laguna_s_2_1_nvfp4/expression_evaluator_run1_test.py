import pytest

class ExpressionEvaluator:
    """Mathematical expression evaluator supporting +, -, *, /, parentheses, and unary minus."""

    def __init__(self):
        self.pos = 0
        self.expr = ""

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression and return its result.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The result of evaluating the expression.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                       has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.expr = expr
        self.pos = 0
        result = self._parse_expression()
        self._skip_whitespace()
        if self.pos != len(self.expr):
            raise ValueError(f"Invalid token at position {self.pos}")
        return result

    def _parse_expression(self) -> float:
        """Parse an expression (handles + and -)."""
        self._skip_whitespace()
        result = self._parse_term()
        while self.pos < len(self.expr):
            self._skip_whitespace()
            if self.pos < len(self.expr) and self.expr[self.pos] == '+':
                self.pos += 1
                result += self._parse_term()
            elif self.pos < len(self.expr) and self.expr[self.pos] == '-':
                self.pos += 1
                result -= self._parse_term()
            else:
                break
        return result

    def _parse_term(self) -> float:
        """Parse a term (handles * and /)."""
        self._skip_whitespace()
        result = self._parse_factor()
        while self.pos < len(self.expr):
            self._skip_whitespace()
            if self.pos < len(self.expr) and self.expr[self.pos] == '*':
                self.pos += 1
                result *= self._parse_factor()
            elif self.pos < len(self.expr) and self.expr[self.pos] == '/':
                self.pos += 1
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                result /= divisor
            else:
                break
        return result

    def _parse_factor(self) -> float:
        """Parse a factor (handles numbers, parentheses, and unary minus)."""
        self._skip_whitespace()
        if self.pos >= len(self.expr):
            raise ValueError("Unexpected end of expression")

        # Handle unary minus
        if self.expr[self.pos] == '-':
            self.pos += 1
            return -self._parse_factor()

        # Handle parentheses
        if self.expr[self.pos] == '(':
            self.pos += 1
            result = self._parse_expression()
            self._skip_whitespace()
            if self.pos >= len(self.expr) or self.expr[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result

        # Parse number
        return self._parse_number()

    def _parse_number(self) -> float:
        """Parse a number (integer or floating point)."""
        self._skip_whitespace()
        start = self.pos
        while self.pos < len(self.expr) and (self.expr[self.pos].isdigit() or self.expr[self.pos] == '.'):
            self.pos += 1
        if start == self.pos:
            raise ValueError(f"Invalid token at position {start}")
        try:
            return float(self.expr[start:self.pos])
        except ValueError:
            raise ValueError(f"Invalid number at position {start}")

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.pos < len(self.expr) and self.expr[self.pos] in ' \t\n\r':
            self.pos += 1

# Tests
def test_basic_operations():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0

def test_unary_minus():
    """Test unary minus operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_floating_point():
    """Test floating point number parsing."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14") == 3.14
    assert evaluator.evaluate("2.5 + 1.5") == 4.0
    assert abs(evaluator.evaluate("1.1 + 2.2") - 3.3) < 1e-10

def test_error_cases():
    """Test error handling for invalid inputs."""
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 +")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 / 0")

def test_complex_expressions():
    """Test complex nested expressions."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == 15.0
    assert evaluator.evaluate("-(2 + 3) * 4") == -20.0
    assert evaluator.evaluate("2.5 * (3 + 1.5)") == 11.25