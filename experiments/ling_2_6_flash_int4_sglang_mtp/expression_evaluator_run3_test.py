from typing import Optional


class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.

    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus
    - Floating point numbers

    Raises:
        ValueError: For invalid expressions, division by zero, etc.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate the mathematical expression `expr`.

        Args:
            expr (str): The expression to evaluate.

        Returns:
            float: The result of evaluating the expression.

        Raises:
            ValueError: If the expression is invalid.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression is empty.")

        self.expr = expr.replace(" ", "")
        self.pos = 0
        self.len = len(self.expr)
        result = self.parse_expression()
        if self.pos < self.len:
            self._error("Unexpected token at position {}".format(self.pos))
        return result

    def parse_expression(self) -> float:
        """Parse addition and subtraction."""
        return self.parse_term()

    def parse_term(self) -> float:
        """Parse multiplication and division."""
        left = self.parse_factor()
        while self.pos < self.len and self.expr[self.pos] in ('*', '/'):
            op = self.expr[self.pos]
            self.pos += 1
            right = self.parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0:
                    self._error("Division by zero.")
                left /= right
        return left

    def parse_factor(self) -> float:
        """Parse numbers, parentheses, and unary minus."""
        if self.pos >= self.len:
            self._error("Unexpected end of expression.")

        if self.expr[self.pos] == '-':
            self.pos += 1
            return -self.parse_factor()

        if self.expr[self.pos] == '(':
            self.pos += 1
            result = self.parse_expression()
            if self.pos >= self.len or self.expr[self.pos] != ')':
                self._error("Mismatched parentheses.")
            self.pos += 1  # Skip ')'
            return result

        return self.parse_number()

    def parse_number(self) -> float:
        """Parse a floating point number."""
        start = self.pos
        while self.pos < self.len and (self.expr[self.pos].isdigit() or self.expr[self.pos] == '.'):
            self.pos += 1
        if start == self.pos:
            self._error("Invalid number at position {}".format(self.pos))
        try:
            return float(self.expr[start:self.pos])
        except ValueError:
            self._error("Invalid number at position {}".format(start))

    def _error(self, msg: str) -> None:
        raise ValueError(msg)

import pytest


def test_simple_addition():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 2") == 3.0


def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2+1)") == -3.0


def test_parentheses_and_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0


def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero."):
        evaluator.evaluate("5 / 0")


def test_mismatched_parentheses():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Mismatched parentheses."):
        evaluator.evaluate("(1 + 2")