from typing import List, Tuple, Any

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.

    Supports:
    - Operators: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary minus (and plus)
    - Floating point numbers
    """

    def __init__(self) -> None:
        self._tokens: List[Tuple[str, Any]] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

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

        result = self._parse_expression()

        if self._current_token()[0] != 'EOF':
            raise ValueError(f"Unexpected token: {self._current_token()[0]}")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, Any]]:
        """Convert the expression string into a list of tokens."""
        tokens: List[Tuple[str, Any]] = []
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
                    raise ValueError(f"Invalid number: '{num_str}'")
                i = j
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], None))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")

        tokens.append(('EOF', None))
        return tokens

    def _current_token(self) -> Tuple[str, Any]:
        """Return the current token or EOF."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return ('EOF', None)

    def _advance(self) -> None:
        """Move to the next token."""
        self._pos += 1

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        while self._current_token()[0] in ('+', '-'):
            op = self._current_token()[0]
            self._advance()
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
            self._advance()
            right = self._parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_factor(self) -> float:
        """Parse unary plus and minus."""
        token = self._current_token()
        if token[0] in ('+', '-'):
            self._advance()
            val = self._parse_factor()
            return val if token[0] == '+' else -val
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        token = self._current_token()
        if token[0] == 'NUM':
            self._advance()
            return token[1]
        elif token[0] == '(':
            self._advance()
            val = self._parse_expression()
            if self._current_token()[0] != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return val
        else:
            raise ValueError(f"Unexpected token: {token[0]}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / are evaluated before + and -."""
    assert evaluator.evaluate("2 + 3 * 4 - 10 / 2") == 7.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping and unary minus support."""
    assert evaluator.evaluate("-(2 + 3) * 2") == -10.0
    assert evaluator.evaluate("-3 + -(1 + 2)") == -6.0

def test_floating_point_numbers(evaluator):
    """Test support for decimal numbers."""
    assert evaluator.evaluate("3.14 * 2.0") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + .5") == pytest.approx(1.0)

def test_error_handling(evaluator):
    """Test ValueError for division by zero, mismatched parentheses, invalid tokens, and empty strings."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")

    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")

    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

def test_complex_nested_expression(evaluator):
    """Test a deeply nested expression with mixed operators."""
    # ((10 - 2) / (3 + 1)) * 5 - 2.5 = (8 / 4) * 5 - 2.5 = 2 * 5 - 2.5 = 7.5
    assert evaluator.evaluate("((10 - 2) / (3 + 1)) * 5 - 2.5") == pytest.approx(7.5)