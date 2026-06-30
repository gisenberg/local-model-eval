from typing import List, Tuple

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    Supports +, -, *, / with correct operator precedence, parentheses for grouping,
    unary minus, and floating-point numbers.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0
        self._current_token = self._tokens[0] if self._tokens else ('EOF', None)

        result = self._parse_expr()

        if self._current_token[0] != 'EOF':
            raise ValueError("Invalid expression: unexpected tokens after end")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, str]]:
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
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format")
                        has_dot = True
                    j += 1
                tokens.append(('NUMBER', expr[i:j]))
                i = j
                continue

            if expr[i] in '+-*/()':
                tokens.append((expr[i], expr[i]))
                i += 1
                continue

            raise ValueError(f"Invalid token: '{expr[i]}'")

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
        """Parse unary operators and primary expressions."""
        if self._current_token[0] == '-':
            self._advance()
            return -self._parse_factor()
        if self._current_token[0] == '+':
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        token_type, token_val = self._current_token
        if token_type == 'NUMBER':
            self._advance()
            return float(token_val)
        if token_type == '(':
            self._advance()
            result = self._parse_expr()
            if self._current_token[0] != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        raise ValueError(f"Unexpected token: {token_val}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / bind tighter than + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping and unary minus handling."""
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("(-2) * 3") == -6.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("- - 3") == 3.0

def test_floating_point_numbers(evaluator):
    """Test decimal number parsing and arithmetic."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("10 / 4") == 2.5
    assert evaluator.evaluate("0.1 + 0.2") == pytest.approx(0.3)

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("2 * (10 / 0)")

def test_invalid_expressions(evaluator):
    """Test error handling for malformed inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")