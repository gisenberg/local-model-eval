import pytest

class ExpressionEvaluator:
    """Mathematical expression evaluator supporting basic arithmetic operations."""

    def __init__(self):
        self.tokens = []
        self.pos = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string.

        Args:
            expr: Mathematical expression string

        Returns:
            Result of the expression

        Raises:
            ValueError: For invalid expressions
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr.strip())
        self.pos = 0
        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError("Invalid expression")

        return result

    def _tokenize(self, expr: str) -> list:
        """Convert expression string to list of tokens."""
        tokens = []
        i = 0

        while i < len(expr):
            char = expr[i]

            if char.isspace():
                i += 1
            elif char.isdigit() or char == '.':
                # Parse number
                num_str = ""
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    num_str += expr[i]
                    i += 1
                tokens.append(float(num_str))
            elif char in '+-*/()':
                tokens.append(char)
                i += 1
            else:
                raise ValueError(f"Invalid character: {char}")

        return tokens

    def _parse_expression(self) -> float:
        """Parse addition and subtraction operations."""
        result = self._parse_term()

        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            result = result + right if op == '+' else result - right

        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division operations."""
        result = self._parse_factor()

        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
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
        """Parse numbers, unary operations, and parentheses."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")

        token = self.tokens[self.pos]

        # Handle unary minus
        if token == '-':
            self.pos += 1
            return -self._parse_factor()

        # Handle parentheses
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result

        # Handle number
        if isinstance(token, float):
            self.pos += 1
            return token

        raise ValueError(f"Invalid token: {token}")

# Tests
def test_basic_operations():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0

def test_operator_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14") == 3.14
    assert evaluator.evaluate("2.5 + 1.5") == 4.0
    assert evaluator.evaluate("1.0 / 3.0") == pytest.approx(0.3333333333333333)

def test_error_handling():
    evaluator = ExpressionEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate("")

    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")

    with pytest.raises(ValueError):
        evaluator.evaluate("2 +")

    with pytest.raises(ValueError):
        evaluator.evaluate("2 / 0")

    with pytest.raises(ValueError):
        evaluator.evaluate("2 + abc")