from __future__ import annotations
from typing import Iterator, Tuple, Union

class ExpressionEvaluator:
    """A recursive descent parser for evaluating mathematical expressions.
    
    Supports +, -, *, / with standard precedence, parentheses, unary minus,
    and floating-point numbers. Raises ValueError for malformed input.
    """

    def __init__(self) -> None:
        pass

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or expr.strip() == '':
            raise ValueError("Empty expression")

        tokens = list(self._tokenize(expr))
        self._tokens: list[Tuple[str, Union[float, str, None]]] = tokens
        self._pos: int = 0

        result = self._parse_expr()

        if self._current_token()[0] != 'EOF':
            raise ValueError(f"Unexpected token: {self._current_token()}")

        return result

    @staticmethod
    def _tokenize(expr: str) -> Iterator[Tuple[str, Union[float, str, None]]]:
        """Convert expression string into a stream of tokens."""
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
                            raise ValueError(f"Invalid token: multiple decimal points at position {j}")
                        has_dot = True
                    j += 1
                num_str = expr[i:j]
                if num_str == '.':
                    raise ValueError(f"Invalid token: '.' at position {i}")
                yield ('NUM', float(num_str))
                i = j
            elif expr[i] == '+':
                yield ('ADD', '+')
                i += 1
            elif expr[i] == '-':
                yield ('SUB', '-')
                i += 1
            elif expr[i] == '*':
                yield ('MUL', '*')
                i += 1
            elif expr[i] == '/':
                yield ('DIV', '/')
                i += 1
            elif expr[i] == '(':
                yield ('LPAREN', '(')
                i += 1
            elif expr[i] == ')':
                yield ('RPAREN', ')')
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}' at position {i}")
        yield ('EOF', None)

    def _current_token(self) -> Tuple[str, Union[float, str, None]]:
        """Return the current token without consuming it."""
        return self._tokens[self._pos]

    def _eat(self, token_type: str) -> Union[float, str, None]:
        """Consume the current token if it matches the expected type."""
        token = self._current_token()
        if token[0] != token_type:
            raise ValueError(f"Expected {token_type}, got {token[0]}")
        self._pos += 1
        return token[1]

    def _parse_expr(self) -> float:
        """Parse additive expressions: term (('+' | '-') term)*"""
        result = self._parse_term()
        while self._current_token()[0] in ('ADD', 'SUB'):
            op = self._current_token()[0]
            self._pos += 1
            right = self._parse_term()
            if op == 'ADD':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplicative expressions: unary (('*' | '/') unary)*"""
        result = self._parse_unary()
        while self._current_token()[0] in ('MUL', 'DIV'):
            op = self._current_token()[0]
            self._pos += 1
            right = self._parse_unary()
            if op == 'MUL':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_unary(self) -> float:
        """Parse unary expressions: ('-')* primary"""
        if self._current_token()[0] == 'SUB':
            self._pos += 1
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse primary expressions: NUMBER | '(' expression ')'"""
        token = self._current_token()
        if token[0] == 'NUM':
            self._pos += 1
            return token[1]
        if token[0] == 'LPAREN':
            self._pos += 1
            result = self._parse_expr()
            if self._current_token()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._pos += 1
            return result
        raise ValueError(f"Unexpected token: {token[0]}")

import pytest

class TestExpressionEvaluator:
    @pytest.fixture
    def evaluator(self):
        return ExpressionEvaluator()

    def test_basic_operations_and_precedence(self, evaluator):
        """Test +, -, *, / with correct operator precedence."""
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        assert evaluator.evaluate("10 / 2 - 3") == 2.0
        assert evaluator.evaluate("1 + 2 + 3 + 4") == 10.0
        assert evaluator.evaluate("8 / 4 * 2") == 4.0

    def test_parentheses_and_unary_minus(self, evaluator):
        """Test grouping with parentheses and unary minus operator."""
        assert evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert evaluator.evaluate("-3") == -3.0
        assert evaluator.evaluate("-(2 + 1)") == -3.0
        assert evaluator.evaluate("---5") == 5.0
        assert evaluator.evaluate("3 * -2") == -6.0

    def test_floating_point_numbers(self, evaluator):
        """Test support for decimal numbers."""
        assert evaluator.evaluate("3.14") == 3.14
        assert evaluator.evaluate(".5 + .5") == 1.0
        assert evaluator.evaluate("2.5 * 4") == 10.0
        assert evaluator.evaluate("10 / 3.0") == pytest.approx(3.3333333333333335)

    def test_error_handling(self, evaluator):
        """Test ValueError raising for invalid inputs."""
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("(3 + 4")
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("10 / 0")
        with pytest.raises(ValueError, match="Invalid token"):
            evaluator.evaluate("3 & 4")
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluator.evaluate("3 +")

    def test_complex_nested_expressions(self, evaluator):
        """Test deeply nested and combined expressions."""
        assert evaluator.evaluate("((10 - 2) / 4) * (3 + 2.5)") == 13.5
        assert evaluator.evaluate("-2 * (3 + 4) / 2") == -7.0
        assert evaluator.evaluate("1 + 2 * (3 - 4) / (5 - 5.5)") == 5.0