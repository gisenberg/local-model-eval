from typing import List, Tuple, Any

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, / with correct operator precedence, parentheses for grouping,
    unary minus, and floating point numbers.
    """

    def __init__(self) -> None:
        self.tokens: List[Tuple[str, Any]] = []
        self.pos: int = 0
        self.current_token: Tuple[str, Any] = ('EOF', None)

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The floating point result of the evaluation.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0
        self.current_token = self.tokens[0]

        result = self._parse_expression()

        if self.current_token[0] != 'EOF':
            raise ValueError("Mismatched parentheses or unexpected tokens")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, Any]]:
        """Convert expression string into a list of tokens."""
        tokens: List[Tuple[str, Any]] = []
        i = 0
        n = len(expr)
        while i < n:
            if expr[i].isspace():
                i += 1
                continue
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                dot_count = 0
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        dot_count += 1
                    j += 1
                if dot_count > 1:
                    raise ValueError(f"Invalid number format at position {i}")
                num_str = expr[i:j]
                if num_str == '.':
                    raise ValueError(f"Invalid number format at position {i}")
                tokens.append(('NUMBER', float(num_str)))
                i = j
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")
        tokens.append(('EOF', None))
        return tokens

    def _advance(self) -> None:
        """Move to the next token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = ('EOF', None)

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self.current_token[0] in ('+', '-'):
            op = self.current_token[0]
            self._advance()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self.current_token[0] in ('*', '/'):
            op = self.current_token[0]
            self._advance()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse numbers, parentheses, and unary operators (highest precedence)."""
        token = self.current_token
        if token[0] == '-':
            self._advance()
            return -self._parse_factor()
        if token[0] == '+':
            self._advance()
            return self._parse_factor()
        if token[0] == 'NUMBER':
            self._advance()
            return token[1]
        if token[0] == '(':
            self._advance()
            result = self._parse_expression()
            if self.current_token[0] != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        raise ValueError(f"Invalid token or unexpected end of expression: {token}")

import pytest

class TestExpressionEvaluator:
    @pytest.fixture
    def evaluator(self):
        return ExpressionEvaluator()

    def test_operator_precedence(self, evaluator):
        """Test correct precedence of +, -, *, /"""
        assert evaluator.evaluate("3 + 5 * 2") == 13.0
        assert evaluator.evaluate("10 - 2 * 3 + 4") == 8.0
        assert evaluator.evaluate("2 * 3 / 4") == 1.5

    def test_parentheses_and_unary(self, evaluator):
        """Test grouping and unary minus support"""
        assert evaluator.evaluate("-(2 + 1)") == -3.0
        assert evaluator.evaluate("2 * (-3 + 4)") == 2.0
        assert evaluator.evaluate("--5") == 5.0
        assert evaluator.evaluate("-(--3)") == -3.0

    def test_floating_point_numbers(self, evaluator):
        """Test floating point number parsing and arithmetic"""
        assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
        assert evaluator.evaluate("1.5 + 2.5") == 4.0
        assert evaluator.evaluate(".5 * 4") == 2.0

    def test_division_by_zero(self, evaluator):
        """Test division by zero raises ValueError"""
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("10 / 0")
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("5 / (2 - 2)")

    def test_error_handling(self, evaluator):
        """Test invalid inputs raise appropriate ValueErrors"""
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")
        with pytest.raises(ValueError, match="Invalid token"):
            evaluator.evaluate("3 + a")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("(3 + 4")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("3 + 4)")