"""Mathematical expression evaluator using a recursive descent parser."""


class _Parser:
    """Recursive descent parser for arithmetic expressions.

    Grammar:
        expr   → term (('+' | '-') term)*
        term   → factor (('*' | '/') factor)*
        factor → ('-' | '+') factor | primary
        primary → NUMBER | '(' expr ')'
        NUMBER  → digit+ ('.' digit+)?
    """

    def __init__(self, expression: str) -> None:
        self._expression: str = expression
        self._pos: int = 0
        self._current_char: str | None = (
            expression[0] if expression else None
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _advance(self) -> None:
        """Move the read head one character forward."""
        self._pos += 1
        self._current_char = (
            self._expression[self._pos]
            if self._pos < len(self._expression)
            else None
        )

    def _skip_whitespace(self) -> None:
        """Skip consecutive whitespace characters."""
        while self._current_char is not None and self._current_char.isspace():
            self._advance()

    def _error(self, message: str) -> None:
        """Raise a ValueError."""
        raise ValueError(message)

    # ------------------------------------------------------------------ #
    # Tokenisers
    # ------------------------------------------------------------------ #

    def _parse_number(self) -> float:
        """Consume and return a numeric literal (integer or float)."""
        start = self._pos

        while self._current_char is not None and self._current_char.isdigit():
            self._advance()

        if self._current_char == '.':
            self._advance()
            if (
                self._current_char is None
                or not self._current_char.isdigit()
            ):
                self._error(
                    f"Expected digit after decimal point at position {self._pos}"
                )
            while self._current_char is not None and self._current_char.isdigit():
                self._advance()
        elif start == self._pos:
            # No digit and no decimal point → not a number
            self._error(f"Expected a number at position {self._pos}")

        token = self._expression[start : self._pos]
        try:
            return float(token)
        except ValueError:
            self._error(f"Invalid number: '{token}'")

    # ------------------------------------------------------------------ #
    # Recursive-descent rules  (top-down, same precedence as grammar)
    # ------------------------------------------------------------------ #

    def _parse_primary(self) -> float:
        """primary → NUMBER | '(' expr ')'"""
        self._skip_whitespace()

        if self._current_char is None:
            self._error("Unexpected end of expression")

        if self._current_char == '(':
            self._advance()  # consume '('
            result = self._parse_expr()
            self._skip_whitespace()
            if self._current_char != ')':
                self._error("Mismatched parentheses: missing closing ')'")
            self._advance()  # consume ')'
            return result

        if self._current_char and (
            self._current_char.isdigit() or self._current_char == '.'
        ):
            return self._parse_number()

        self._error(f"Invalid token: '{self._current_char}'")

    def _parse_factor(self) -> float:
        """factor → ('-' | '+') factor | primary"""
        self._skip_whitespace()

        if self._current_char == '-':
            self._advance()
            return -self._parse_factor()

        if self._current_char == '+':
            self._advance()
            return self._parse_factor()

        return self._parse_primary()

    def _parse_term(self) -> float:
        """term → factor (('*' | '/') factor)*"""
        result = self._parse_factor()

        while True:
            self._skip_whitespace()
            if self._current_char == '*':
                self._advance()
                result *= self._parse_factor()
            elif self._current_char == '/':
                self._advance()
                right = self._parse_factor()
                if right == 0:
                    self._error("Division by zero")
                result /= right
            else:
                break

        return result

    def _parse_expr(self) -> float:
        """expr → term (('+' | '-') term)*"""
        result = self._parse_term()

        while True:
            self._skip_whitespace()
            if self._current_char == '+':
                self._advance()
                result += self._parse_term()
            elif self._current_char == '-':
                self._advance()
                result -= self._parse_term()
            else:
                break

        return result


class ExpressionEvaluator:
    """Evaluate arithmetic expressions with correct operator precedence.

    Supports:
    - Binary operators: +, -, *, /
    - Parentheses for grouping
    - Unary plus and minus (e.g. ``-3``, ``-(2+1)``)
    - Floating-point literals (e.g. ``3.14``)

    Raises ``ValueError`` for empty expressions, mismatched parentheses,
    division by zero, or invalid tokens.
    """

    def evaluate(self, expr: str) -> float:
        """Evaluate *expr* and return the numeric result.

        Args:
            expr: An arithmetic expression string.

        Returns:
            The evaluated result as a ``float``.

        Raises:
            ValueError: If the expression is empty, contains mismatched
                parentheses, involves division by zero, or contains
                invalid tokens.

        Examples:
            >>> ev = ExpressionEvaluator()
            >>> ev.evaluate("3 + 4 * 2")
            11.0
            >>> ev.evaluate("-(2 + 3)")
            -5.0
        """
        if not expr or expr.isspace():
            raise ValueError("Empty expression")

        parser = _Parser(expr)
        result = parser._parse_expr()

        parser._skip_whitespace()
        if parser._current_char is not None:
            parser._error(
                f"Unexpected character: '{parser._current_char}' "
                f"at position {parser._pos}"
            )

        return result

"""Tests for ExpressionEvaluator."""

import pytest


@pytest.fixture
def ev() -> ExpressionEvaluator:
    return ExpressionEvaluator()

class TestBasicOperationsAndPrecedence:
    """Test binary operators and precedence rules."""

    def test_addition(self, ev: ExpressionEvaluator) -> None:
        assert ev.evaluate("2 + 3") == 5.0
        assert ev.evaluate("1 + 2 + 3 + 4") == 10.0

    def test_multiplication_and_precedence(self, ev: ExpressionEvaluator) -> None:
        assert ev.evaluate("3 + 4 * 2") == 11.0       # * before +
        assert ev.evaluate("6 / 3 + 2") == 4.0         # / before +
        assert ev.evaluate("2 * 3 + 4 * 5") == 26.0   # left-to-right *

    def test_left_to_right_associativity(self, ev: ExpressionEvaluator) -> None:
        assert ev.evaluate("10 - 3 - 2") == 5.0
        assert ev.evaluate("10 / 5 / 2") == pytest.approx(1.0)


class TestParentheses:
    """Test grouping with parentheses."""

    def test_basic_parens(self, ev: ExpressionEvaluator) -> None:
        assert ev.evaluate("(3 + 4) * 2") == 14.0
        assert ev.evaluate("2 * (3 + 4)") == 14.0

    def test_nested_parens(self, ev: ExpressionEvaluator) -> None:
        assert ev.evaluate("((2 + 3) * 4)") == 20.0
        assert ev.evaluate("(2 + (3 * 4))") == 14.0


class TestUnaryMinusAndFloats:
    """Test unary operators and floating-point literals."""

    def test_unary_minus(self, ev: ExpressionEvaluator) -> None:
        assert ev.evaluate("-3") == -3.0
        assert ev.evaluate("-(2 + 3)") == -5.0
        assert ev.evaluate("--4") == 4.0

    def test_unary_plus(self, ev: ExpressionEvaluator) -> None:
        assert ev.evaluate("+5") == 5.0
        assert ev.evaluate("+-3") == -3.0

    def test_floating_point(self, ev: ExpressionEvaluator) -> None:
        assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
        assert ev.evaluate("-0.5 + 1.5") == pytest.approx(1.0)
        assert ev.evaluate(".5 + .5") == pytest.approx(1.0)


class TestErrorHandling:
    """Test that ValueError is raised for invalid input."""

    def test_empty_expression(self, ev: ExpressionEvaluator) -> None:
        with pytest.raises(ValueError, match="Empty expression"):
            ev.evaluate("")
        with pytest.raises(ValueError, match="Empty expression"):
            ev.evaluate("   ")

    def test_division_by_zero(self, ev: ExpressionEvaluator) -> None:
        with pytest.raises(ValueError, match="Division by zero"):
            ev.evaluate("5 / 0")

    def test_mismatched_parentheses(self, ev: ExpressionEvaluator) -> None:
        with pytest.raises(ValueError, match="Mismatched"):
            ev.evaluate("(2 + 3")

    def test_invalid_tokens(self, ev: ExpressionEvaluator) -> None:
        with pytest.raises(ValueError, match="Invalid token"):
            ev.evaluate("2 & 3")
        with pytest.raises(ValueError):
            ev.evaluate("2 + +")           # trailing unary op, no operand

    def test_trailing_garbage(self, ev: ExpressionEvaluator) -> None:
        with pytest.raises(ValueError, match="Unexpected character"):
            ev.evaluate("2 + 3 a")