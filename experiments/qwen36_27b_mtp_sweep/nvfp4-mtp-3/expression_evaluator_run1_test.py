class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, / with correct precedence, parentheses, unary minus,
    and floating-point numbers. Raises ValueError for invalid inputs.
    """

    def __init__(self) -> None:
        self.tokens: list = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.
        
        Args:
            expr: A string containing a valid mathematical expression.
            
        Returns:
            The numerical result of the expression.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        result = self._parse_expression()

        if self.tokens[self.pos][0] != 'EOF':
            raise ValueError("Invalid expression: unexpected tokens after valid expression")

        return result

    def _tokenize(self, expr: str) -> list:
        """Convert expression string into a list of (type, value) tokens."""
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
                    tokens.append(('NUM', float(num_str)))
                except ValueError:
                    raise ValueError(f"Invalid number format: {num_str}")
                i = j
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: {expr[i]}")
        tokens.append(('EOF', None))
        return tokens

    def _current_token(self) -> tuple:
        """Return the current token without consuming it."""
        return self.tokens[self.pos]

    def _eat(self, expected: str = None) -> tuple:
        """Consume and return the current token, optionally checking its type."""
        token = self._current_token()
        if expected and token[0] != expected:
            raise ValueError(f"Expected {expected}, got {token[0]}")
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        while self._current_token()[0] in ('+', '-'):
            op = self._current_token()[0]
            self._eat()
            right = self._parse_term()
            if op == '+':
                left += right
            else:
                left -= right
        return left

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        left = self._parse_factor()
        while self._current_token()[0] in ('*', '/'):
            op = self._current_token()[0]
            self._eat()
            right = self._parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_factor(self) -> float:
        """Parse numbers, parentheses, and unary operators (highest precedence)."""
        token = self._current_token()
        if token[0] == '-':
            self._eat()
            return -self._parse_factor()
        elif token[0] == 'NUM':
            self._eat()
            return token[1]
        elif token[0] == '(':
            self._eat()
            result = self._parse_expression()
            if self._current_token()[0] != ')':
                raise ValueError("Mismatched parentheses")
            self._eat()
            return result
        else:
            raise ValueError(f"Unexpected token: {token[0]}")

import pytest

class TestExpressionEvaluator:
    @pytest.fixture
    def evaluator(self):
        return ExpressionEvaluator()

    def test_operator_precedence(self, evaluator):
        """Test that * and / bind tighter than + and -"""
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        assert evaluator.evaluate("10 / 2 - 3") == 2.0
        assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

    def test_parentheses_and_unary_minus(self, evaluator):
        """Test grouping and unary minus handling"""
        assert evaluator.evaluate("-(2 + 3)") == -5.0
        assert evaluator.evaluate("(-2) * 3") == -6.0
        assert evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert evaluator.evaluate("---3") == -3.0

    def test_floating_point_numbers(self, evaluator):
        """Test decimal number parsing and arithmetic"""
        assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
        assert evaluator.evaluate("1.5 + 2.5") == 4.0
        assert evaluator.evaluate(".5 * 4") == 2.0

    def test_division_by_zero(self, evaluator):
        """Test that division by zero raises ValueError"""
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("10 / 0")
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("0 / 0.0")

    def test_invalid_expressions(self, evaluator):
        """Test error handling for malformed inputs"""
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Invalid token"):
            evaluator.evaluate("2 + a")
        with pytest.raises(ValueError, match="Invalid expression"):
            evaluator.evaluate("2 + 3 )")