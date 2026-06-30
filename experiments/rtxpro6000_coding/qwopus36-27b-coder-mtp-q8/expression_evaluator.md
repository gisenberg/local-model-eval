```python
"""
Mathematical expression evaluator using recursive descent parsing.

Supports +, -, *, / with correct operator precedence, parentheses,
unary minus, and floating point numbers.
"""

from typing import Tuple


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.

    Grammar:
        expr   -> term (('+' | '-') term)*
        term   -> factor (('*' | '/') factor)*
        factor -> ('+' | '-') factor | primary
        primary -> NUMBER | '(' expr ')'
    """

    def __init__(self) -> None:
        """Initialize the evaluator."""
        pass

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The floating point result of the expression.

        Raises:
            ValueError: If the expression is invalid, empty, has mismatched
                parentheses, division by zero, or contains invalid tokens.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._expr = expr.strip()
        self._pos = 0
        self._length = len(self._expr)

        result = self._parse_expr()

        if self._pos < self._length:
            raise ValueError(f"Invalid token at position {self._pos}: '{self._expr[self._pos]}'")

        return result

    def _parse_expr(self) -> float:
        """
        Parse an expression: term (('+' | '-') term)*

        Returns:
            The evaluated result of the expression.
        """
        result = self._parse_term()

        while self._pos < self._length and self._expr[self._pos] in ('+', '-'):
            op = self._expr[self._pos]
            self._pos += 1
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right

        return result

    def _parse_term(self) -> float:
        """
        Parse a term: factor (('*' | '/') factor)*

        Returns:
            The evaluated result of the term.
        """
        result = self._parse_factor()

        while self._pos < self._length and self._expr[self._pos] in ('*', '/'):
            op = self._expr[self._pos]
            self._pos += 1
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right

        return result

    def _parse_factor(self) -> float:
        """
        Parse a factor: ('+' | '-') factor | primary

        Handles unary plus and minus.

        Returns:
            The evaluated result of the factor.
        """
        if self._pos < self._length and self._expr[self._pos] in ('+', '-'):
            op = self._expr[self._pos]
            self._pos += 1
            operand = self._parse_factor()
            if op == '-':
                return -operand
            else:
                return operand

        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parse a primary: NUMBER | '(' expr ')'

        Returns:
            The evaluated result of the primary.

        Raises:
            ValueError: If there is a mismatched parenthesis or invalid token.
        """
        if self._pos >= self._length:
            raise ValueError("Unexpected end of expression")

        ch = self._expr[self._pos]

        if ch == '(':
            self._pos += 1
            result = self._parse_expr()
            if self._pos >= self._length or self._expr[self._pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._pos += 1
            return result

        if ch.isdigit() or ch == '.':
            return self._parse_number()

        raise ValueError(f"Invalid token at position {self._pos}: '{ch}'")

    def _parse_number(self) -> float:
        """
        Parse a floating point number.

        Returns:
            The parsed number as a float.

        Raises:
            ValueError: If the number format is invalid.
        """
        start = self._pos
        has_dot = False

        while self._pos < self._length:
            ch = self._expr[self._pos]
            if ch.isdigit():
                self._pos += 1
            elif ch == '.':
                if has_dot:
                    raise ValueError(f"Invalid number format at position {start}")
                has_dot = True
                self._pos += 1
            else:
                break

        if self._pos == start:
            raise ValueError(f"Invalid token at position {start}")

        num_str = self._expr[start:self._pos]

        # Validate that we have at least one digit
        if not any(c.isdigit() for c in num_str):
            raise ValueError(f"Invalid number format: '{num_str}'")

        try:
            return float(num_str)
        except ValueError:
            raise ValueError(f"Invalid number format: '{num_str}'")
```

```python
"""
Tests for ExpressionEvaluator.
"""

import pytest
from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Create an ExpressionEvaluator instance."""
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator: ExpressionEvaluator) -> None:
    """Test basic arithmetic operations."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # precedence
    assert evaluator.evaluate("2.5 + 3.5") == 6.0
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_precedence(evaluator: ExpressionEvaluator) -> None:
    """Test operator precedence: * and / before + and -."""
    assert evaluator.evaluate("2 + 3 * 4 - 1") == 13.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("1 + 2 + 3 + 4") == 10.0
    assert evaluator.evaluate("10 / 2 / 5") == 1.0


def test_parentheses(evaluator: ExpressionEvaluator) -> None:
    """Test parentheses for grouping."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0
    assert evaluator.evaluate("(10 - 2) / (4 - 2)") == 4.0
    assert evaluator.evaluate("1 + (2 * (3 + 4))") == 15.0


def test_unary_minus(evaluator: ExpressionEvaluator) -> None:
    """Test unary minus operator."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("5 - -3") == 8.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--3") == 3.0
    assert evaluator.evaluate("-2 * 3") == -6.0
    assert evaluator.evaluate("2 * -3") == -6.0
    assert evaluator.evaluate("-3.14") == pytest.approx(-3.14)


def test_error_cases(evaluator: ExpressionEvaluator) -> None:
    """Test error handling for invalid expressions."""
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")

    # Whitespace only
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")

    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

    # Invalid token
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")

    # Invalid number format
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("2..5")
```