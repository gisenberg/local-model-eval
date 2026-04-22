from typing import List, Tuple, Any

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Operators: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary plus and minus (e.g., '-3', '-(2+1)', '--5')
    - Floating-point numbers (e.g., '3.14', '.5', '3.')
    
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
        Evaluates a mathematical expression string and returns the result.

        Args:
            expr: The mathematical expression to evaluate.

        Returns:
            The numerical result as a float.

        Raises:
            ValueError: If the expression is invalid, empty, or contains errors.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        result = self._parse_expression()

        if self._current_token()[0] != 'EOF':
            raise ValueError("Invalid expression or mismatched parentheses")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, Any]]:
        """Converts the input string into a list of (type, value) tokens."""
        tokens: List[Tuple[str, Any]] = []
        i, n = 0, len(expr)

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
                    raise ValueError(f"Invalid number: {num_str}")
                i = j
                continue

            if expr[i] in '+-*/()':
                tokens.append(('OP', expr[i]))
                i += 1
                continue

            raise ValueError(f"Invalid token: '{expr[i]}'")

        tokens.append(('EOF', None))
        return tokens

    def _current_token(self) -> Tuple[str, Any]:
        """Returns the token at the current position."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ('EOF', None)

    def _eat(self, expected_type: str, expected_value: str = None) -> None:
        """Consumes the current token if it matches expectations."""
        token = self._current_token()
        if token[0] != expected_type or (expected_value is not None and token[1] != expected_value):
            raise ValueError(f"Expected {expected_type} {expected_value}, got {token}")
        self.pos += 1

    def _parse_expression(self) -> float:
        """Parses addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token()[0] == 'OP' and self._current_token()[1] in ('+', '-'):
            op = self._current_token()[1]
            self.pos += 1
            right = self._parse_term()
            result = result + right if op == '+' else result - right
        return result

    def _parse_term(self) -> float:
        """Parses multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token()[0] == 'OP' and self._current_token()[1] in ('*', '/'):
            op = self._current_token()[1]
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
        """Parses unary plus and minus operators."""
        token = self._current_token()
        if token[0] == 'OP' and token[1] in ('+', '-'):
            op = token[1]
            self.pos += 1
            val = self._parse_factor()
            return val if op == '+' else -val
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parses atomic values: numbers and parenthesized expressions."""
        token = self._current_token()
        if token[0] == 'NUM':
            self.pos += 1
            return token[1]

        if token[0] == 'OP' and token[1] == '(':
            self.pos += 1
            result = self._parse_expression()
            if self._current_token()[0] != 'OP' or self._current_token()[1] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result

        raise ValueError(f"Unexpected token: {token}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Tests correct precedence of +, -, *, /"""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus(evaluator):
    """Tests grouping with parentheses and unary minus handling"""
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0
    assert evaluator.evaluate("-(2 * 3) + 10") == 4.0

def test_floating_point_numbers(evaluator):
    """Tests support for decimal numbers"""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("10.0 / 4.0") == pytest.approx(2.5)

def test_division_by_zero(evaluator):
    """Tests that division by zero raises ValueError"""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (2 - 2)")

def test_invalid_inputs(evaluator):
    """Tests error handling for empty, invalid, and mismatched expressions"""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 & 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")