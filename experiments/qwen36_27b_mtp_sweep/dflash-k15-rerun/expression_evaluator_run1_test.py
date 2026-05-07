from enum import Enum, auto
from typing import List


class TokenType(Enum):
    """Enumeration of all possible token types in the expression grammar."""
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MULT = auto()
    DIV = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


class Token:
    """Represents a lexical token with its type and string value."""
    def __init__(self, type: TokenType, value: str) -> None:
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Addition (+), Subtraction (-), Multiplication (*), Division (/)
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14, .5, 2.)
    
    Raises ValueError for invalid syntax, mismatched parentheses, 
    division by zero, and empty expressions.
    """
    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The numerical result of the evaluation.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token: {self._tokens[self._pos]}")

        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Converts the input string into a list of tokens."""
        tokens: List[Token] = []
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
                            raise ValueError(f"Invalid number format at position {j}")
                        has_dot = True
                    j += 1
                
                num_str = expr[i:j]
                if not any(c.isdigit() for c in num_str):
                    raise ValueError(f"Invalid number: {num_str}")
                
                tokens.append(Token(TokenType.NUMBER, num_str))
                i = j
                continue

            if ch == '+':
                tokens.append(Token(TokenType.PLUS, '+'))
            elif ch == '-':
                tokens.append(Token(TokenType.MINUS, '-'))
            elif ch == '*':
                tokens.append(Token(TokenType.MULT, '*'))
            elif ch == '/':
                tokens.append(Token(TokenType.DIV, '/'))
            elif ch == '(':
                tokens.append(Token(TokenType.LPAREN, '('))
            elif ch == ')':
                tokens.append(Token(TokenType.RPAREN, ')'))
            else:
                raise ValueError(f"Invalid token: '{ch}'")
            
            i += 1

        tokens.append(Token(TokenType.EOF, ''))
        return tokens

    def _current_token(self) -> Token:
        """Returns the token at the current parsing position."""
        return self._tokens[self._pos]

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Grammar: expression := term (('+' | '-') term)*
        """
        result = self._parse_term()
        while self._current_token().type in (TokenType.PLUS, TokenType.MINUS):
            op = self._current_token().type
            self._pos += 1
            right = self._parse_term()
            if op == TokenType.PLUS:
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Grammar: term := factor (('*' | '/') factor)*
        """
        result = self._parse_factor()
        while self._current_token().type in (TokenType.MULT, TokenType.DIV):
            op = self._current_token().type
            self._pos += 1
            right = self._parse_factor()
            if op == TokenType.MULT:
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """
        Parses unary plus and minus.
        Grammar: factor := ('+' | '-') factor | primary
        """
        token = self._current_token()
        if token.type == TokenType.PLUS:
            self._pos += 1
            return self._parse_factor()
        elif token.type == TokenType.MINUS:
            self._pos += 1
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parses numbers and parenthesized expressions.
        Grammar: primary := NUMBER | '(' expression ')'
        """
        token = self._current_token()
        if token.type == TokenType.NUMBER:
            self._pos += 1
            return float(token.value)
        elif token.type == TokenType.LPAREN:
            self._pos += 1
            result = self._parse_expression()
            if self._current_token().type != TokenType.RPAREN:
                raise ValueError("Mismatched parentheses")
            self._pos += 1
            return result
        else:
            raise ValueError(f"Unexpected token: {token}")

import pytest


@pytest.fixture
def evaluator():
    return ExpressionEvaluator()


def test_operator_precedence(evaluator: ExpressionEvaluator) -> None:
    """Test that * and / are evaluated before + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0


def test_parentheses_grouping(evaluator: ExpressionEvaluator) -> None:
    """Test that parentheses correctly override default precedence."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert evaluator.evaluate("10 - (2 + 3) * 2") == 0.0


def test_unary_minus(evaluator: ExpressionEvaluator) -> None:
    """Test unary minus handling in various contexts."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 + -2 * 3") == -1.0
    assert evaluator.evaluate("--5") == 5.0


def test_floating_point_numbers(evaluator: ExpressionEvaluator) -> None:
    """Test support for decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("1.5 + 2.5") == 4.0
    assert evaluator.evaluate(".5 * 4") == 2.0
    assert evaluator.evaluate("2. / 4.") == 0.5


def test_error_handling(evaluator: ExpressionEvaluator) -> None:
    """Test that appropriate ValueErrors are raised for invalid inputs."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")

    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")

    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3 )")