from typing import List, Tuple, Any

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.

    Supports:
    - Addition (+), Subtraction (-), Multiplication (*), Division (/)
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers (e.g., '3.14', '.5', '5.')

    Raises ValueError for:
    - Empty expressions
    - Invalid tokens
    - Mismatched parentheses
    - Division by zero
    """

    def __init__(self) -> None:
        self.tokens: List[Tuple[str, Any]] = []
        self.pos: int = 0

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

        self._tokenize(expr)
        self.pos = 0
        result = self._parse_expression()

        if self.pos < len(self.tokens) and self.tokens[self.pos][0] != 'EOF':
            raise ValueError("Mismatched parentheses or unexpected tokens")

        return result

    def _tokenize(self, expr: str) -> None:
        """Convert the expression string into a list of tokens."""
        self.tokens = []
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or ch == '.':
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format")
                        has_dot = True
                    j += 1
                num_str = expr[i:j]
                if num_str == '.':
                    raise ValueError("Invalid number format")
                self.tokens.append(('NUM', float(num_str)))
                i = j
            elif ch == '+':
                self.tokens.append(('PLUS', '+'))
                i += 1
            elif ch == '-':
                self.tokens.append(('MINUS', '-'))
                i += 1
            elif ch == '*':
                self.tokens.append(('MULT', '*'))
                i += 1
            elif ch == '/':
                self.tokens.append(('DIV', '/'))
                i += 1
            elif ch == '(':
                self.tokens.append(('LPAREN', '('))
                i += 1
            elif ch == ')':
                self.tokens.append(('RPAREN', ')'))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{ch}'")
        self.tokens.append(('EOF', None))

    def _current_token(self) -> Tuple[str, Any]:
        """Return the current token at the parser's position."""
        return self.tokens[self.pos]

    def _eat(self, token_type: str) -> None:
        """Consume the current token if it matches the expected type."""
        if self._current_token()[0] == token_type:
            self.pos += 1
        else:
            raise ValueError(f"Expected {token_type}, got {self._current_token()[0]}")

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token()[0] in ('PLUS', 'MINUS'):
            op = self._current_token()[0]
            self.pos += 1
            right = self._parse_term()
            if op == 'PLUS':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token()[0] in ('MULT', 'DIV'):
            op = self._current_token()[0]
            self.pos += 1
            right = self._parse_factor()
            if op == 'MULT':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary operators (highest precedence)."""
        if self._current_token()[0] == 'MINUS':
            self.pos += 1
            return -self._parse_factor()
        if self._current_token()[0] == 'PLUS':
            self.pos += 1
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        token = self._current_token()
        if token[0] == 'NUM':
            self.pos += 1
            return token[1]
        if token[0] == 'LPAREN':
            self.pos += 1
            result = self._parse_expression()
            if self._current_token()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result
        raise ValueError(f"Unexpected token: {token}")

import pytest

def test_basic_precedence():
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 - 2 / 2") == 9.0

def test_parentheses_and_unary():
    ev = ExpressionEvaluator()
    assert ev.evaluate("-(2 + 3)") == -5.0
    assert ev.evaluate("- - 3") == 3.0
    assert ev.evaluate("(2 + 3) * 4") == 20.0

def test_floating_point():
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate(".5 + .5") == 1.0

def test_division_by_zero():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("10 / 0")

def test_invalid_expressions():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError):
        ev.evaluate("")
    with pytest.raises(ValueError):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        ev.evaluate("2 + a")