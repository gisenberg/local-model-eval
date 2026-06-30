from __future__ import annotations
from typing import List, Optional

class Token:
    """Represents a lexical token in the expression."""
    def __init__(self, type: str, value: Optional[float]) -> None:
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


class ExpressionEvaluator:
    """
    A recursive descent parser that evaluates mathematical expressions.
    
    Supports:
    - Addition (+), Subtraction (-), Multiplication (*), Division (/)
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary plus and minus operators
    - Floating point numbers
    """
    def __init__(self) -> None:
        self.tokens: List[Token] = []
        self.pos: int = 0
        self.current_token: Optional[Token] = None

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0
        self.current_token = self.tokens[0]

        result = self._parse_expression()

        if self.current_token and self.current_token.type != 'EOF':
            raise ValueError("Unexpected tokens after expression")

        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Converts a string expression into a list of tokens."""
        tokens: List[Token] = []
        i = 0
        n = len(expr)

        while i < n:
            char = expr[i]

            if char.isspace():
                i += 1
                continue

            if char.isdigit() or char == '.':
                start = i
                has_dot = False
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format: multiple decimal points")
                        has_dot = True
                    i += 1
                num_str = expr[start:i]
                if num_str == '.':
                    raise ValueError("Invalid number format: lone decimal point")
                tokens.append(Token('NUMBER', float(num_str)))
            elif char == '+':
                tokens.append(Token('PLUS', char))
                i += 1
            elif char == '-':
                tokens.append(Token('MINUS', char))
                i += 1
            elif char == '*':
                tokens.append(Token('MULT', char))
                i += 1
            elif char == '/':
                tokens.append(Token('DIV', char))
                i += 1
            elif char == '(':
                tokens.append(Token('LPAREN', char))
                i += 1
            elif char == ')':
                tokens.append(Token('RPAREN', char))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{char}'")

        tokens.append(Token('EOF', None))
        return tokens

    def _eat(self, token_type: str) -> Token:
        """Consumes the current token if it matches the expected type."""
        if self.current_token and self.current_token.type == token_type:
            token = self.current_token
            self.pos += 1
            self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None
            return token
        raise ValueError(f"Unexpected token: {self.current_token}")

    def _parse_expression(self) -> float:
        """Handles addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self.current_token and self.current_token.type in ('PLUS', 'MINUS'):
            op = self.current_token.type
            self._eat(op)
            right = self._parse_term()
            if op == 'PLUS':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Handles multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self.current_token and self.current_token.type in ('MULT', 'DIV'):
            op = self.current_token.type
            self._eat(op)
            right = self._parse_factor()
            if op == 'MULT':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Handles unary plus and minus operators."""
        if self.current_token and self.current_token.type in ('PLUS', 'MINUS'):
            op = self.current_token.type
            self._eat(op)
            operand = self._parse_factor()  # Recursive call supports chained unary ops like --3
            return operand if op == 'PLUS' else -operand
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and parenthesized expressions."""
        if not self.current_token:
            raise ValueError("Unexpected end of expression")

        if self.current_token.type == 'NUMBER':
            token = self._eat('NUMBER')
            return token.value
        elif self.current_token.type == 'LPAREN':
            self._eat('LPAREN')
            result = self._parse_expression()
            if not self.current_token or self.current_token.type != 'RPAREN':
                raise ValueError("Mismatched parentheses: missing closing parenthesis")
            self._eat('RPAREN')
            return result
        else:
            raise ValueError(f"Invalid token in expression: {self.current_token}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence_and_parentheses(evaluator):
    """Tests correct precedence of * and / over + and -, and grouping with ()"""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    assert evaluator.evaluate("((2 + 3) * (4 - 1)) / 5") == pytest.approx(3.0)

def test_unary_minus_and_plus(evaluator):
    """Tests unary operators including chained and grouped cases"""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("- - 3.5") == 3.5
    assert evaluator.evaluate("+10") == 10.0

def test_floating_point_numbers(evaluator):
    """Tests parsing and arithmetic with decimal numbers"""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("1.5 + 2.5") == 4.0
    assert evaluator.evaluate("10.0 / 3.0") == pytest.approx(3.3333333333333335)
    assert evaluator.evaluate(".5 + .5") == 1.0

def test_error_handling(evaluator):
    """Tests ValueError raising for invalid inputs"""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError, match="Unexpected tokens"):
        evaluator.evaluate("2 + 3 )")

def test_complex_nested_expressions(evaluator):
    """Tests deeply nested and mixed operator expressions"""
    assert evaluator.evaluate("-2 * (3 + 4) / 2") == -7.0
    assert evaluator.evaluate("1 + 2 * 3 - 4 / 2") == 5.0
    assert evaluator.evaluate("-( -( -( 5 ) ) )") == -5.0
    assert evaluator.evaluate("10 / (2 + 3) * 4") == pytest.approx(8.0)