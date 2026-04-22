from enum import Enum
from typing import Iterator, List, Optional

class TokenType(Enum):
    """Enumeration of all possible token types in the grammar."""
    NUMBER = 'NUMBER'
    PLUS = 'PLUS'
    MINUS = 'MINUS'
    MULT = 'MULT'
    DIV = 'DIV'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    EOF = 'EOF'

class Token:
    """Represents a single lexical token."""
    def __init__(self, type: TokenType, value: Optional[float] = None) -> None:
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f'Token({self.type.name}, {self.value!r})'


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Grammar:
        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := ['-'] (NUMBER | '(' expression ')')
    """

    def __init__(self) -> None:
        self.tokens: List[Token] = []
        self.pos: int = 0
        self.current_token: Optional[Token] = None

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

        self.tokens = list(self._tokenize(expr))
        self.pos = 0
        self.current_token = self.tokens[0]

        result = self._parse_expression()

        if self.current_token.type != TokenType.EOF:
            raise ValueError(f"Unexpected token: {self.current_token}")

        return result

    def _tokenize(self, expr: str) -> Iterator[Token]:
        """Convert the expression string into an iterator of tokens."""
        i = 0
        n = len(expr)
        while i < n:
            if expr[i].isspace():
                i += 1
                continue
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                num_str = expr[i:j]
                if num_str.count('.') > 1:
                    raise ValueError(f"Invalid number: {num_str}")
                if num_str == '.':
                    raise ValueError(f"Invalid number: {num_str}")
                yield Token(TokenType.NUMBER, float(num_str))
                i = j
            elif expr[i] == '+':
                yield Token(TokenType.PLUS)
                i += 1
            elif expr[i] == '-':
                yield Token(TokenType.MINUS)
                i += 1
            elif expr[i] == '*':
                yield Token(TokenType.MULT)
                i += 1
            elif expr[i] == '/':
                yield Token(TokenType.DIV)
                i += 1
            elif expr[i] == '(':
                yield Token(TokenType.LPAREN)
                i += 1
            elif expr[i] == ')':
                yield Token(TokenType.RPAREN)
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")
        yield Token(TokenType.EOF)

    def _advance(self) -> None:
        """Move the parser to the next token."""
        self.pos += 1
        self.current_token = self.tokens[self.pos]

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token.type
            self._advance()
            right = self._parse_term()
            if op == TokenType.PLUS:
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self.current_token.type in (TokenType.MULT, TokenType.DIV):
            op = self.current_token.type
            self._advance()
            right = self._parse_factor()
            if op == TokenType.MULT:
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse numbers, unary minus, and parenthesized expressions (highest precedence)."""
        token = self.current_token
        if token.type == TokenType.NUMBER:
            self._advance()
            return token.value
        elif token.type == TokenType.MINUS:
            self._advance()
            return -self._parse_factor()
        elif token.type == TokenType.LPAREN:
            self._advance()
            result = self._parse_expression()
            if self.current_token.type != TokenType.RPAREN:
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        else:
            raise ValueError(f"Invalid token in factor: {token}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / bind tighter than + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 22.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping and unary negation."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("- - - 3") == -3.0

def test_floating_point_numbers(evaluator):
    """Test decimal number parsing and arithmetic."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("1.5 + 2.5") == 4.0
    assert evaluator.evaluate(".5 * 4") == 2.0
    assert evaluator.evaluate("10.0 / 3.0") == pytest.approx(3.3333333333333335)

def test_error_handling(evaluator):
    """Test ValueError raising for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("2.3.4")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / (2 - 2)")