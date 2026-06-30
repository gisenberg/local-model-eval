from __future__ import annotations
from typing import Optional


class Token:
    """Represents a lexical token in the expression."""
    def __init__(self, type: str, value: Optional[float]) -> None:
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


class ExpressionEvaluator:
    """
    A recursive descent parser and evaluator for mathematical expressions.
    
    Supports:
    - Binary operators: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary plus/minus
    - Floating-point numbers
    """

    def __init__(self) -> None:
        self.tokens: list[Token] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        result = self._parse_expression()

        if self.current_token.type != 'EOF':
            raise ValueError("Unexpected token after expression")

        return result

    @property
    def current_token(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> None:
        self.pos += 1

    def _tokenize(self, expr: str) -> list[Token]:
        """Convert expression string into a list of tokens."""
        tokens: list[Token] = []
        i = 0
        n = len(expr)

        while i < n:
            c = expr[i]
            if c.isspace():
                i += 1
                continue

            if c.isdigit() or c == '.':
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format: multiple decimal points")
                        has_dot = True
                    j += 1
                num_str = expr[i:j]
                if num_str == '.':
                    raise ValueError("Invalid number format: lone decimal point")
                tokens.append(Token('NUMBER', float(num_str)))
                i = j
                continue

            if c == '+':
                tokens.append(Token('PLUS', None))
            elif c == '-':
                tokens.append(Token('MINUS', None))
            elif c == '*':
                tokens.append(Token('MULT', None))
            elif c == '/':
                tokens.append(Token('DIV', None))
            elif c == '(':
                tokens.append(Token('LPAREN', None))
            elif c == ')':
                tokens.append(Token('RPAREN', None))
            else:
                raise ValueError(f"Invalid character: {c!r}")

            i += 1

        tokens.append(Token('EOF', None))
        return tokens

    def _parse_expression(self) -> float:
        """Handle addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self.current_token.type in ('PLUS', 'MINUS'):
            op = self.current_token.type
            self._advance()
            right = self._parse_term()
            result = result + right if op == 'PLUS' else result - right
        return result

    def _parse_term(self) -> float:
        """Handle multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self.current_token.type in ('MULT', 'DIV'):
            op = self.current_token.type
            self._advance()
            right = self._parse_factor()
            if op == 'MULT':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Handle unary plus/minus."""
        if self.current_token.type in ('PLUS', 'MINUS'):
            op = self.current_token.type
            self._advance()
            value = self._parse_factor()
            return value if op == 'PLUS' else -value
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handle numbers and parenthesized expressions."""
        token = self.current_token
        if token.type == 'NUMBER':
            self._advance()
            return float(token.value)
        elif token.type == 'LPAREN':
            self._advance()
            result = self._parse_expression()
            if self.current_token.type != 'RPAREN':
                raise ValueError("Mismatched parentheses: missing closing parenthesis")
            self._advance()
            return result
        else:
            raise ValueError(f"Unexpected token: {token.type}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic_and_precedence(evaluator):
    """Test standard operators and correct precedence."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_unary_minus_support(evaluator):
    """Test unary minus in various contexts."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("- - 3.5") == 3.5
    assert evaluator.evaluate("2 * -3") == -6.0

def test_floating_point_numbers(evaluator):
    """Test parsing and arithmetic with floats."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("1.5 + 2.5") == 4.0
    assert evaluator.evaluate("10 / 3") == pytest.approx(3.3333333333333335)
    assert evaluator.evaluate(".5 + 5.") == 5.5

def test_error_handling(evaluator):
    """Test ValueError raising for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3 )")
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + a")

def test_complex_nested_expressions(evaluator):
    """Test deeply nested and mixed expressions."""
    assert evaluator.evaluate("((10 - 2) / 4) * (3 + 1.5)") == pytest.approx(9.0)
    assert evaluator.evaluate("-2 * (3 + 4) / 2") == -7.0
    assert evaluator.evaluate("1 + 2 * (3 - 4) / (5 + 5)") == pytest.approx(0.8)
    assert evaluator.evaluate("-( -( -( 2 ) ) )") == -2.0