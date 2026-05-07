from typing import List, Optional


class Token:
    """Represents a lexical token in the expression."""
    def __init__(self, type: str, value: Optional[float] = None) -> None:
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value})"


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.

    Supports:
    - Addition (+), Subtraction (-), Multiplication (*), Division (/)
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14)

    Raises ValueError for mismatched parentheses, division by zero,
    invalid tokens, and empty expressions.
    """
    def __init__(self) -> None:
        self.tokens: List[Token] = []
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

        self.tokens = self._tokenize(expr)
        self.pos = 0

        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")

        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Convert the expression string into a list of tokens."""
        tokens: List[Token] = []
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
                if expr[i] == '.' and j == i + 1:
                    raise ValueError("Invalid token: '.'")
                tokens.append(Token('NUMBER', float(expr[i:j])))
                i = j
            elif expr[i] in '+-*/()':
                tokens.append(Token(expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")

        return tokens

    def _current_token(self) -> Token:
        """Return the current token or an EOF token if at the end."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token('EOF')

    def _advance(self) -> Token:
        """Consume the current token and return it."""
        token = self._current_token()
        if self.pos < len(self.tokens):
            self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token().type in ('+', '-'):
            op = self._advance().type
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token().type in ('*', '/'):
            op = self._advance().type
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary operators, numbers, and parentheses (highest precedence)."""
        token = self._current_token()

        if token.type == '-':
            self._advance()
            return -self._parse_factor()
        if token.type == '+':
            self._advance()
            return self._parse_factor()
        if token.type == 'NUMBER':
            self._advance()
            return token.value
        if token.type == '(':
            self._advance()
            result = self._parse_expression()
            if self._current_token().type != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        if token.type == ')':
            raise ValueError("Mismatched parentheses")
        raise ValueError(f"Invalid token: {token}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / are evaluated before + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3 + 4") == 8.0
    assert evaluator.evaluate("20 / 4 + 5") == 10.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping with parentheses and unary minus operator."""
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("-3 * -2") == 6.0
    assert evaluator.evaluate("-(2 + 1) * 4") == -12.0
    assert evaluator.evaluate("(-5)") == -5.0

def test_floating_point_numbers(evaluator):
    """Test support for decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("1.5 + 2.5") == 4.0
    assert evaluator.evaluate("10.0 / 4.0") == pytest.approx(2.5)

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("(2 - 2) / 5")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0.0")

def test_error_handling(evaluator):
    """Test ValueError for empty, invalid, and mismatched expressions."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 & 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + )")