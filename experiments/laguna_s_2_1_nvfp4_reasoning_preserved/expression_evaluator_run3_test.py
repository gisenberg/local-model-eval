import pytest

class ExpressionEvaluator:
    """
    A recursive descent parser-based mathematical expression evaluator.

    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        self.tokens = []
        self.pos = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.

        Args:
            expr: Mathematical expression string

        Returns:
            Result of the expression as float

        Raises:
            ValueError: For invalid expressions, mismatched parentheses,
                       division by zero, or empty input
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        if not self.tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")

        return result

    def _tokenize(self, expr: str) -> list:
        """Convert expression string into list of tokens."""
        tokens = []
        i = 0

        while i < len(expr):
            char = expr[i]

            if char.isspace():
                i += 1
                continue
            elif char in '+-*/()':
                tokens.append(char)
                i += 1
            elif char.isdigit() or char == '.':
                # Parse number (integer or float)
                start = i
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                num_str = expr[start:i]
                try:
                    # Validate it's a proper number
                    float(num_str)
                    tokens.append(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")
            else:
                raise ValueError(f"Invalid character: {char}")

        return tokens

    def _parse_expression(self) -> float:
        """Parse addition and subtraction operations."""
        result = self._parse_term()

        while self.pos < len(self.tokens) and self.tokens[self.pos] in '+-':
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()

            if op == '+':
                result += right
            else:
                result -= right

        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division operations."""
        result = self._parse_factor()

        while self.pos < len(self.tokens) and self.tokens[self.pos] in '*/':
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()

            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right

        return result

    def _parse_factor(self) -> float:
        """Parse unary minus and primary expressions."""
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1
            return -self._parse_factor()

        if self.pos < len(self.tokens) and self.tokens[self.pos] == '+':
            self.pos += 1
            return self._parse_factor()

        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")

        token = self.tokens[self.pos]

        if token == '(':
            self.pos += 1
            result = self._parse_expression()

            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")

            self.pos += 1
            return result
        elif token == ')':
            raise ValueError("Mismatched parentheses")
        else:
            # Number
            try:
                self.pos += 1
                return float(token)
            except ValueError:
                raise ValueError(f"Invalid token: {token}")


# Tests
def test_basic_operations():
    """Test basic arithmetic operations with correct precedence."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("2 * 3 + 4") == 10.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0

def test_parentheses():
    """Test parentheses for grouping."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3))") == 5.0

def test_unary_minus():
    """Test unary minus operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_floating_point():
    """Test floating point number support."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14") == 3.14
    assert evaluator.evaluate("2.5 + 1.5") == 4.0
    assert evaluator.evaluate("10.0 / 4") == 2.5

def test_error_handling():
    """Test error conditions."""
    evaluator = ExpressionEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate("")

    with pytest.raises(ValueError):
        evaluator.evaluate("2 + ")

    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")

    with pytest.raises(ValueError):
        evaluator.evaluate("2 + )")

    with pytest.raises(ValueError):
        evaluator.evaluate("2 / 0")

    with pytest.raises(ValueError):
        evaluator.evaluate("2 + @")