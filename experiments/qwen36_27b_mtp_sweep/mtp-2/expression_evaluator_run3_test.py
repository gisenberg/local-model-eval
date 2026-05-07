from typing import Any

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Grammar:
        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := ('+' | '-') factor | primary
        primary    := NUMBER | '(' expression ')'
    """

    def __init__(self) -> None:
        self.tokens: list[tuple[str, Any]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

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
            raise ValueError("Unexpected token after expression")

        return result

    def _tokenize(self, expr: str) -> list[tuple[str, Any]]:
        """Convert the expression string into a list of tokens."""
        tokens: list[tuple[str, Any]] = []
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
                            raise ValueError("Invalid number format: multiple decimal points")
                        has_dot = True
                    j += 1
                tokens.append(('NUMBER', float(expr[i:j])))
                i = j
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")

        tokens.append(('EOF', None))
        return tokens

    def _current_token(self) -> tuple[str, Any]:
        """Return the current token being processed."""
        return self.tokens[self.pos]

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token()[0] in ('+', '-'):
            op = self._current_token()[0]
            self.pos += 1
            right = self._parse_term()
            result = result + right if op == '+' else result - right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token()[0] in ('*', '/'):
            op = self._current_token()[0]
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
        """Parse unary plus and minus."""
        if self._current_token()[0] in ('+', '-'):
            op = self._current_token()[0]
            self.pos += 1
            val = self._parse_factor()
            return val if op == '+' else -val
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        token = self._current_token()
        if token[0] == 'NUMBER':
            self.pos += 1
            return token[1]
        if token[0] == '(':
            self.pos += 1
            result = self._parse_expression()
            if self._current_token()[0] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result
        raise ValueError(f"Unexpected token: {token[0]}")

import pytest

def test_basic_arithmetic_and_precedence():
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 - 2 / 2") == 9.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus():
    ev = ExpressionEvaluator()
    assert ev.evaluate("-(2 + 3)") == -5.0
    assert ev.evaluate("- - 3") == 3.0
    assert ev.evaluate("(-2) * 3") == -6.0
    assert ev.evaluate("2 * -3 + 4") == -2.0

def test_floating_point_numbers():
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate("1.5 + 2.5") == 4.0
    assert ev.evaluate(".5 * 4") == 2.0

def test_division_by_zero():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("(2 + 3) / 0.0")

def test_error_handling():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + a")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Unexpected token"):
        ev.evaluate("2 + 3)")