from typing import List, Union
import re


class ExpressionEvaluator:
    """
    A recursive descent parser-based mathematical expression evaluator.

    Supports:
        - Basic arithmetic: +, -, *, /
        - Parentheses for grouping
        - Unary minus (e.g., "-3", "-(2+1)")
        - Floating point numbers (e.g., "3.14")

    Raises:
        ValueError: For mismatched parentheses, division by zero, invalid tokens, or empty expressions.
    """

    # Token types: 'NUMBER', 'PLUS', 'MINUS', 'MUL', 'DIV', 'LPAREN', 'RPAREN', 'EOF'
    _TOKEN_REGEX = re.compile(
        r"\s*("
        r"(?:\d+\.?\d*|\.\d+)"  # number (e.g., 3, 3.0, .5)
        r"|\+|-|\*|/|\(|\)"
        r")"
    )

    def __init__(self):
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.

        Args:
            expr (str): The expression string to evaluate.

        Returns:
            float: The numerical result.

        Raises:
            ValueError: On invalid input (mismatched parens, div by zero, invalid token, empty).
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        # Tokenize
        self._tokens = self._tokenize(expr)
        self._pos = 0

        try:
            result = self._parse_expression()
            if self._current_token() != 'EOF':
                raise ValueError("Unexpected token after expression")
            return result
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except (SyntaxError, IndexError) as e:
            raise ValueError(f"Parse error: {e}")

    def _tokenize(self, expr: str) -> List[str]:
        """Convert expression string to list of tokens."""
        tokens = []
        for match in self._TOKEN_REGEX.finditer(expr):
            token = match.group(1)
            tokens.append(token)
        tokens.append('EOF')  # Sentinel
        return tokens

    def _current_token(self) -> str:
        """Get current token (or 'EOF' if at end)."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return 'EOF'

    def _advance(self) -> str:
        """Advance token pointer and return previous token."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    # ——— Recursive descent grammar ———
    # expression → term (('+' | '-') term)*
    # term       → factor (('*' | '/') factor)*
    # factor     → ['+' | '-'] factor | number | '(' expression ')'

    def _parse_expression(self) -> float:
        result = self._parse_term()
        while self._current_token() in ('PLUS', 'MINUS') or self._current_token() in '+-':
            op = self._advance()
            if op == '+':
                result += self._parse_term()
            else:  # '-'
                result -= self._parse_term()
        return result

    def _parse_term(self) -> float:
        result = self._parse_factor()
        while self._current_token() in ('MUL', 'DIV') or self._current_token() in '*/':
            op = self._advance()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:  # '/'
                if right == 0:
                    raise ZeroDivisionError()
                result /= right
        return result

    def _parse_factor(self) -> float:
        # Handle unary +/-
        if self._current_token() == '+':
            self._advance()
            return self._parse_factor()
        elif self._current_token() == '-':
            self._advance()
            return -self._parse_factor()

        token = self._current_token()

        # Number
        if re.match(r'^-?\d+\.?\d*$|^\.\d+$', token):
            self._advance()
            return float(token)

        # Parenthesized expression
        if token == '(':
            self._advance()  # consume '('
            result = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses: expected ')'")
            self._advance()  # consume ')'
            return result

        # Invalid token
        raise ValueError(f"Invalid token: '{token}'")

import pytest


@pytest.fixture
def evaluator():
    return ExpressionEvaluator()


def test_basic_operations(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14
    assert evaluator.evaluate("10 - 3 + 2") == 9


def test_parentheses_and_unary_minus(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20
    assert evaluator.evaluate("-3") == -3
    assert evaluator.evaluate("-(2 + 3)") == -5
    assert evaluator.evaluate("- -(5)") == 5


def test_floating_point_numbers(evaluator):
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + .5") == 1.0


def test_error_cases(evaluator):
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")

    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + abc")

    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")


def test_complex_nested_expr(evaluator):
    assert evaluator.evaluate("((2 + 3) * (4 - 1)) / 5") == 3.0
    assert evaluator.evaluate("-(-3 + -(2 - 1))") == 2.0