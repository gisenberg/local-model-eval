from typing import List, Optional


class ExpressionEvaluator:
    """
    A recursive descent parser and evaluator for mathematical expressions.
    
    Grammar:
        expression -> term (('+' | '-') term)*
        term       -> factor (('*' | '/') factor)*
        factor     -> ('+' | '-') factor | primary
        primary    -> NUMBER | '(' expression ')'
    """

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.pos: int = 0

    def _tokenize(self, expr: str) -> None:
        """Converts the input string into a list of tokens."""
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokens: List[str] = []
        i = 0
        n = len(expr)

        while i < n:
            if expr[i].isspace():
                i += 1
                continue

            if expr[i].isdigit() or expr[i] == '.':
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or (expr[j] == '.' and not has_dot)):
                    if expr[j] == '.':
                        has_dot = True
                    j += 1
                tokens.append(expr[i:j])
                i = j
            elif expr[i] in '+-*/()':
                tokens.append(expr[i])
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")

        if not tokens:
            raise ValueError("Empty expression")

        self.tokens = tokens
        self.pos = 0

    def _current_token(self) -> Optional[str]:
        """Returns the current token or None if at end of input."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _parse_expression(self) -> float:
        """Parses addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token() in ('+', '-'):
            op = self._current_token()
            self.pos += 1
            right = self._parse_term()
            result = result + right if op == '+' else result - right
        return result

    def _parse_term(self) -> float:
        """Parses multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token() in ('*', '/'):
            op = self._current_token()
            self.pos += 1
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parses unary plus and minus."""
        if self._current_token() in ('+', '-'):
            op = self._current_token()
            self.pos += 1
            val = self._parse_factor()
            return val if op == '+' else -val
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parses numbers and parenthesized expressions (highest precedence)."""
        token = self._current_token()
        if token is None:
            raise ValueError("Unexpected end of expression")
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self.pos += 1
            return result
        if token == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid number: '{token}'")

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        self._tokenize(expr)
        result = self._parse_expression()
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: '{self._current_token()}'")
        return result

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence_and_parentheses(evaluator):
    """Tests correct precedence of * / over + - and grouping with parentheses."""
    assert evaluator.evaluate("3 + 4 * 2") == 14.0
    assert evaluator.evaluate("(3 + 4) * 2") == 14.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_unary_minus(evaluator):
    """Tests unary minus in various positions."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("- - 3.5") == 3.5
    assert evaluator.evaluate("10 + -4 * 2") == 2.0

def test_floating_point_numbers(evaluator):
    """Tests support for decimal numbers."""
    assert evaluator.evaluate("3.14 + 2.86") == pytest.approx(6.0)
    assert evaluator.evaluate(".5 * 4") == 2.0
    assert evaluator.evaluate("10.0 / 3.0") == pytest.approx(3.3333333333333335)
    assert evaluator.evaluate("1.5 * 2.5") == pytest.approx(3.75)

def test_error_handling(evaluator):
    """Tests ValueError raising for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 + a")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(3 + 4")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("3 + 4)")

def test_complex_nested_expressions(evaluator):
    """Tests deeply nested and complex combinations."""
    assert evaluator.evaluate("((2 + 3) * 4 - 1) / 5") == pytest.approx(3.8)
    assert evaluator.evaluate("-2 * (3 + -4)") == 2.0
    assert evaluator.evaluate("1 + 2 * (3 + 4) / 7 - 5") == pytest.approx(-2.0)
    assert evaluator.evaluate("-(--(--3))") == -3.0