class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, / with correct precedence, parentheses, unary minus,
    and floating-point numbers.
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

        self.tokens = self._tokenize(expr)
        if not self.tokens:
            raise ValueError("Empty expression")

        self.pos = 0
        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")

        return result

    def _tokenize(self, expr: str) -> list:
        """Convert expression string into a list of tokens (numbers, operators, parentheses)."""
        tokens = []
        i = 0
        n = len(expr)
        while i < n:
            if expr[i].isspace():
                i += 1
                continue
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                num_str = expr[i:j]
                try:
                    tokens.append(float(num_str))
                except ValueError:
                    raise ValueError(f"Invalid number: '{num_str}'")
                i = j
                continue
            if expr[i] in '+-*/()':
                tokens.append(expr[i])
                i += 1
                continue
            raise ValueError(f"Invalid character: '{expr[i]}'")
        return tokens

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            if op == '+':
                left += right
            else:
                left -= right
        return left

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        left = self._parse_factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_factor(self) -> float:
        """Parse factors (delegates to unary operator handling)."""
        return self._parse_unary()

    def _parse_unary(self) -> float:
        """Parse unary minus operator."""
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse primary expressions: numbers and parenthesized sub-expressions."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")

        token = self.tokens[self.pos]
        if isinstance(token, float):
            self.pos += 1
            return token
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result
        raise ValueError(f"Unexpected token: {token}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / bind tighter than + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping and unary minus support."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("-3 + 2") == -1.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("---4") == -4.0

def test_floating_point_numbers(evaluator):
    """Test decimal number parsing and arithmetic."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("1.5 + 2.5") == 4.0
    assert evaluator.evaluate("10 / 3") == pytest.approx(3.3333333333333335)

def test_division_by_zero_and_mismatched_parens(evaluator):
    """Test error handling for division by zero and parentheses."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")

def test_invalid_tokens_and_empty(evaluator):
    """Test error handling for invalid characters and empty strings."""
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 & 3")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")