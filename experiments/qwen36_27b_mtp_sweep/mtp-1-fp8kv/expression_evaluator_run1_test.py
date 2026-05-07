from __future__ import annotations

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, / with correct precedence, parentheses, unary minus,
    and floating point numbers.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0
        self._current_token = self._tokens[self._pos]

        result = self._parse_expr()

        if self._current_token[0] != 'EOF':
            raise ValueError(f"Unexpected token: {self._current_token}")

        return result

    def _tokenize(self, expr: str) -> list[tuple[str, float | str | None]]:
        """Convert expression string into a list of tokens."""
        tokens: list[tuple[str, float | str | None]] = []
        i = 0
        n = len(expr)
        while i < n:
            if expr[i].isspace():
                i += 1
                continue
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format")
                        has_dot = True
                    j += 1
                try:
                    num = float(expr[i:j])
                except ValueError:
                    raise ValueError(f"Invalid number: {expr[i:j]}")
                tokens.append(('NUM', num))
                i = j
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: {expr[i]}")
        tokens.append(('EOF', None))
        return tokens

    def _advance(self) -> None:
        """Move to the next token."""
        self._pos += 1
        if self._pos < len(self._tokens):
            self._current_token = self._tokens[self._pos]
        else:
            self._current_token = ('EOF', None)

    def _parse_expr(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token[0] in ('+', '-'):
            op = self._current_token[0]
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
        while self._current_token[0] in ('*', '/'):
            op = self._current_token[0]
            self._advance()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary operators, numbers, and parenthesized expressions."""
        token = self._current_token
        if token[0] == '+':
            self._advance()
            return self._parse_factor()
        elif token[0] == '-':
            self._advance()
            return -self._parse_factor()
        elif token[0] == 'NUM':
            self._advance()
            return float(token[1])
        elif token[0] == '(':
            self._advance()
            result = self._parse_expr()
            if self._current_token[0] != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        else:
            raise ValueError(f"Invalid token in expression: {token}")

import pytest

class TestExpressionEvaluator:
    @pytest.fixture(autouse=True)
    def evaluator(self):
        return ExpressionEvaluator()

    def test_operator_precedence(self, evaluator):
        """Test that * and / are evaluated before + and -."""
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        assert evaluator.evaluate("10 - 2 * 3 + 4") == 8.0

    def test_parentheses_and_unary_minus(self, evaluator):
        """Test grouping and unary minus handling."""
        assert evaluator.evaluate("-(2 + 3) * 2") == -10.0
        assert evaluator.evaluate("-3") == -3.0
        assert evaluator.evaluate("2 * -(3 + 4)") == -14.0

    def test_floating_point_numbers(self, evaluator):
        """Test support for decimal numbers."""
        assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
        assert evaluator.evaluate(".5 + 1.5") == pytest.approx(2.0)

    def test_division_by_zero(self, evaluator):
        """Test that division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("10 / 0")
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("5 / (2 - 2)")

    def test_error_handling(self, evaluator):
        """Test ValueError for empty, invalid, and mismatched expressions."""
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("   ")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Invalid token"):
            evaluator.evaluate("2 + a")
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluator.evaluate("2 + 3)")