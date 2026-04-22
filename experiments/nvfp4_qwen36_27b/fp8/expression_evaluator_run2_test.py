import re
from typing import List, Optional, Union

class Token:
    """Represents a lexical token in the expression."""
    def __init__(self, type: str, value: Union[float, str, None]):
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


class Tokenizer:
    """Converts a string expression into a list of tokens."""
    def __init__(self, text: str):
        self.tokens: List[Token] = self._tokenize(text)

    def _tokenize(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        # Matches: numbers (int/float), operators, parentheses, or any non-whitespace char
        pattern = re.compile(r'[0-9]+(?:\.[0-9]+)?|[+\-*/()]|\S')
        for match in re.finditer(pattern, text):
            val = match.group()
            if val in '+-*/()':
                tokens.append(Token(val, val))
            elif val.replace('.', '', 1).isdigit() and val != '.':
                tokens.append(Token('NUMBER', float(val)))
            else:
                raise ValueError(f"Invalid token: {val}")
        tokens.append(Token('EOF', None))
        return tokens


class ExpressionEvaluator:
    """
    A recursive descent parser and evaluator for mathematical expressions.
    
    Supports:
    - Operators: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary minus (and plus)
    - Floating point numbers
    """
    def __init__(self):
        self.tokens: List[Token] = []
        self.pos: int = 0
        self.current_token: Optional[Token] = None

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The numerical result of the expression.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokenizer = Tokenizer(expr)
        self.tokens = tokenizer.tokens
        self.pos = 0
        self.current_token = self.tokens[0]

        result = self._parse_expression()

        if self.current_token.type != 'EOF':
            raise ValueError(f"Unexpected token: {self.current_token.type}")

        return result

    def _advance(self) -> None:
        """Moves to the next token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = Token('EOF', None)

    def _parse_expression(self) -> float:
        """Parses addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self.current_token.type in ('+', '-'):
            op = self.current_token.type
            self._advance()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parses multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self.current_token.type in ('*', '/'):
            op = self.current_token.type
            self._advance()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parses unary operators and delegates to primary."""
        if self.current_token.type == '-':
            self._advance()
            return -self._parse_factor()
        if self.current_token.type == '+':
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parses numbers and parenthesized expressions."""
        token = self.current_token
        if token.type == 'NUMBER':
            self._advance()
            return float(token.value)
        if token.type == '(':
            self._advance()
            result = self._parse_expression()
            if self.current_token.type != ')':
                raise ValueError("Mismatched parentheses: missing closing parenthesis")
            self._advance()
            return result
        raise ValueError(f"Unexpected token: {token.type}")

import pytest

def test_basic_precedence():
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 / 2 - 3") == 2.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary():
    ev = ExpressionEvaluator()
    assert ev.evaluate("-(2 + 3) * 4") == -20.0
    assert ev.evaluate("-3 + -2") == -5.0
    assert ev.evaluate("-( -5 )") == 5.0
    assert ev.evaluate("2 * -(3 + 4)") == -14.0

def test_floating_point():
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate(".5 + .5") == 1.0
    assert ev.evaluate("10.0 / 4.0") == pytest.approx(2.5)

def test_division_by_zero():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("(2 + 3) / 0.0")

def test_invalid_expressions():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + abc")
    with pytest.raises(ValueError, match="Unexpected token"):
        ev.evaluate("2 + * 3")