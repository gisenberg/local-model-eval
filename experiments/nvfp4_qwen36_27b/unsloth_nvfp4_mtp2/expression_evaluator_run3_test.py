class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.

    Supports:
    - Addition (+), subtraction (-), multiplication (*), division (/)
    - Correct operator precedence
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers (e.g., '3.14')

    Raises ValueError for:
    - Mismatched parentheses
    - Division by zero
    - Invalid tokens
    - Empty expressions
    """

    def __init__(self) -> None:
        self._expr: str = ""
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.

        Args:
            expr: The mathematical expression to evaluate.

        Returns:
            The numerical result of the expression.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._expr = expr
        self._pos = 0

        result = self._parse_expression()

        self._skip_whitespace()
        if self._pos < len(self._expr):
            raise ValueError(f"Invalid token at position {self._pos}: '{self._expr[self._pos]}'")

        return result

    def _skip_whitespace(self) -> None:
        """Skips whitespace characters in the expression."""
        while self._pos < len(self._expr) and self._expr[self._pos].isspace():
            self._pos += 1

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Grammar: expression -> term (('+' | '-') term)*
        """
        left = self._parse_term()
        while self._pos < len(self._expr):
            op = self._expr[self._pos]
            if op == '+':
                self._pos += 1
                left += self._parse_term()
            elif op == '-':
                self._pos += 1
                left -= self._parse_term()
            else:
                break
        return left

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Grammar: term -> factor (('*' | '/') factor)*
        """
        left = self._parse_factor()
        while self._pos < len(self._expr):
            op = self._expr[self._pos]
            if op == '*':
                self._pos += 1
                left *= self._parse_factor()
            elif op == '/':
                self._pos += 1
                right = self._parse_factor()
                if right == 0.0:
                    raise ValueError("Division by zero")
                left /= right
            else:
                break
        return left

    def _parse_factor(self) -> float:
        """
        Parses unary operators and primary expressions.
        Grammar: factor -> ('+' | '-') factor | primary
        """
        self._skip_whitespace()
        if self._pos < len(self._expr) and self._expr[self._pos] in ('+', '-'):
            sign = 1.0 if self._expr[self._pos] == '+' else -1.0
            self._pos += 1
            return sign * self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parses numbers and parenthesized expressions.
        Grammar: primary -> NUMBER | '(' expression ')'
        """
        self._skip_whitespace()
        if self._pos >= len(self._expr):
            raise ValueError("Unexpected end of expression")

        char = self._expr[self._pos]
        if char == '(':
            self._pos += 1
            result = self._parse_expression()
            self._skip_whitespace()
            if self._pos >= len(self._expr) or self._expr[self._pos] != ')':
                raise ValueError("Mismatched parentheses")
            self._pos += 1
            return result
        elif char.isdigit() or char == '.':
            return self._parse_number()
        else:
            raise ValueError(f"Invalid token: '{char}'")

    def _parse_number(self) -> float:
        """
        Parses a numeric literal (integer or float).
        """
        start = self._pos
        has_dot = False
        while self._pos < len(self._expr) and (self._expr[self._pos].isdigit() or self._expr[self._pos] == '.'):
            if self._expr[self._pos] == '.':
                if has_dot:
                    raise ValueError("Invalid number format: multiple decimal points")
                has_dot = True
            self._pos += 1

        if self._pos == start:
            raise ValueError("Expected number")

        try:
            return float(self._expr[start:self._pos])
        except ValueError:
            raise ValueError(f"Invalid number format: '{self._expr[start:self._pos]}'")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic_and_precedence(evaluator):
    """Tests correct operator precedence for +, -, *, /"""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    assert evaluator.evaluate("100 / 10 / 2") == 5.0

def test_parentheses_and_unary_minus(evaluator):
    """Tests grouping with parentheses and unary minus handling"""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-(--2 + 3)") == -5.0

def test_floating_point_numbers(evaluator):
    """Tests parsing and arithmetic with floating point literals"""
    assert evaluator.evaluate("3.14") == 3.14
    assert evaluator.evaluate("1.5 * 2.0") == 3.0
    assert evaluator.evaluate(".5 + 1.5") == 2.0
    assert evaluator.evaluate("10 / 4.0") == 2.5

def test_division_by_zero_error(evaluator):
    """Tests that division by zero raises ValueError"""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (2 - 2)")

def test_invalid_and_empty_expressions(evaluator):
    """Tests error handling for malformed or empty inputs"""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + abc")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + 3)")