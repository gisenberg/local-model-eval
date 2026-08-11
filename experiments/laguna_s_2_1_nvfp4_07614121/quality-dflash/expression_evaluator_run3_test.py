class ExpressionEvaluator:
    """
    Mathematical expression evaluator supporting +, -, *, /, parentheses,
    unary minus, and floating point numbers.
    """

    def __init__(self):
        self.expr = ""
        self.pos = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.

        Args:
            expr: Mathematical expression string

        Returns:
            Result as float

        Raises:
            ValueError: For invalid expressions, mismatched parentheses,
                       division by zero, or invalid tokens
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.expr = expr.strip()
        self.pos = 0
        result = self._parse_expression()

        # Check for trailing characters
        self._skip_whitespace()
        if self.pos < len(self.expr):
            raise ValueError(f"Invalid token at position {self.pos}")

        return result

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.pos < len(self.expr) and self.expr[self.pos].isspace():
            self.pos += 1

    def _parse_expression(self) -> float:
        """Parse addition and subtraction operations."""
        self._skip_whitespace()
        result = self._parse_term()

        while self.pos < len(self.expr):
            self._skip_whitespace()
            if self.pos >= len(self.expr):
                break

            op = self.expr[self.pos]
            if op in '+-':
                self.pos += 1
                right = self._parse_term()
                if op == '+':
                    result += right
                else:
                    result -= right
            else:
                break

        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division operations."""
        self._skip_whitespace()
        result = self._parse_factor()

        while self.pos < len(self.expr):
            self._skip_whitespace()
            if self.pos >= len(self.expr):
                break

            op = self.expr[self.pos]
            if op in '*/':
                self.pos += 1
                right = self._parse_factor()
                if op == '*':
                    result *= right
                else:
                    if right == 0:
                        raise ValueError("Division by zero")
                    result /= right
            else:
                break

        return result

    def _parse_factor(self) -> float:
        """Parse factors: numbers, parentheses, unary minus."""
        self._skip_whitespace()

        if self.pos >= len(self.expr):
            raise ValueError("Unexpected end of expression")

        # Unary minus
        if self.expr[self.pos] == '-':
            self.pos += 1
            return -self._parse_factor()

        # Parentheses
        if self.expr[self.pos] == '(':
            self.pos += 1
            result = self._parse_expression()
            self._skip_whitespace()
            if self.pos >= len(self.expr) or self.expr[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result

        # Number
        return self._parse_number()

    def _parse_number(self) -> float:
        """Parse a number (integer or float)."""
        self._skip_whitespace()

        if self.pos >= len(self.expr):
            raise ValueError("Unexpected end of expression")

        start = self.pos
        is_float = False

        # Handle leading digits
        while self.pos < len(self.expr) and self.expr[self.pos].isdigit():
            self.pos += 1

        # Handle decimal point
        if self.pos < len(self.expr) and self.expr[self.pos] == '.':
            is_float = True
            self.pos += 1
            while self.pos < len(self.expr) and self.expr[self.pos].isdigit():
                self.pos += 1

        if self.pos == start:
            raise ValueError(f"Invalid token at position {start}")

        num_str = self.expr[start:self.pos]
        return float(num_str) if is_float else float(int(num_str))


# Tests
import pytest

def test_basic_operations():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0

def test_operator_precedence():
    """Test operator precedence."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("2 * 3 + 4") == 10.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0

def test_parentheses():
    """Test parentheses grouping."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3))") == 5.0

def test_unary_minus():
    """Test unary minus."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_floating_point():
    """Test floating point numbers."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14") == 3.14
    assert evaluator.evaluate("2.5 + 1.5") == 4.0
    assert abs(evaluator.evaluate("1.1 + 2.2") - 3.3) < 1e-10

def test_errors():
    """Test error conditions."""
    evaluator = ExpressionEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate("")

    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")

    with pytest.raises(ValueError):
        evaluator.evaluate("2 + + 3")

    with pytest.raises(ValueError):
        evaluator.evaluate("2 / 0")