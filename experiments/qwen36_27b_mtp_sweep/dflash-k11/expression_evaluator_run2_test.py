from typing import Optional

class Token:
    """Represents a lexical token."""
    def __init__(self, type: str, value: Optional[float]):
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value})"


class Lexer:
    """Converts an expression string into a stream of tokens."""
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if text else None

    def advance(self) -> None:
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def skip_whitespace(self) -> None:
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self) -> float:
        """Parse an integer or floating point number."""
        result = ''
        has_digit = False
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if '.' in result:
                    raise ValueError("Invalid number format")
            else:
                has_digit = True
            result += self.current_char
            self.advance()
        if not has_digit:
            raise ValueError("Invalid number format")
        return float(result)

    def get_next_token(self) -> Token:
        """Return the next token from the input string."""
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            if self.current_char.isdigit() or self.current_char == '.':
                return Token('NUMBER', self.number())
            if self.current_char == '+':
                self.advance()
                return Token('PLUS', '+')
            if self.current_char == '-':
                self.advance()
                return Token('MINUS', '-')
            if self.current_char == '*':
                self.advance()
                return Token('MULT', '*')
            if self.current_char == '/':
                self.advance()
                return Token('DIV', '/')
            if self.current_char == '(':
                self.advance()
                return Token('LPAREN', '(')
            if self.current_char == ')':
                self.advance()
                return Token('RPAREN', ')')
            raise ValueError(f"Invalid token: '{self.current_char}'")
        return Token('EOF', None)


class Parser:
    """Recursive descent parser for mathematical expressions."""
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def eat(self, token_type: str) -> None:
        """Consume the current token if it matches the expected type."""
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            if token_type == 'RPAREN':
                raise ValueError("Mismatched parentheses")
            raise ValueError(f"Expected {token_type}, got {self.current_token.type}")

    def parse(self) -> float:
        """Parse the entire expression and return the result."""
        result = self.expr()
        if self.current_token.type != 'EOF':
            if self.current_token.type == 'RPAREN':
                raise ValueError("Mismatched parentheses")
            raise ValueError(f"Unexpected token after expression: {self.current_token}")
        return result

    def expr(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        node = self.term()
        while self.current_token.type in ('PLUS', 'MINUS'):
            token = self.current_token
            if token.type == 'PLUS':
                self.eat('PLUS')
                node = node + self.term()
            elif token.type == 'MINUS':
                self.eat('MINUS')
                node = node - self.term()
        return node

    def term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        node = self.factor()
        while self.current_token.type in ('MULT', 'DIV'):
            token = self.current_token
            if token.type == 'MULT':
                self.eat('MULT')
                node = node * self.factor()
            elif token.type == 'DIV':
                self.eat('DIV')
                divisor = self.factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                node = node / divisor
        return node

    def factor(self) -> float:
        """Parse unary operators, numbers, and parenthesized expressions."""
        token = self.current_token
        if token.type == 'PLUS':
            self.eat('PLUS')
            return self.factor()
        if token.type == 'MINUS':
            self.eat('MINUS')
            return -self.factor()
        if token.type == 'NUMBER':
            self.eat('NUMBER')
            return token.value
        if token.type == 'LPAREN':
            self.eat('LPAREN')
            node = self.expr()
            self.eat('RPAREN')
            return node
        if token.type == 'EOF':
            raise ValueError("Unexpected end of expression")
        raise ValueError(f"Unexpected token: {token}")


class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    Supports +, -, *, / with correct operator precedence, parentheses,
    unary minus, and floating point numbers.
    """
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
        lexer = Lexer(expr)
        parser = Parser(lexer)
        return parser.parse()

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / are evaluated before + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping with parentheses and unary minus operator."""
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("-3 * (4 - 1)") == -9.0
    assert evaluator.evaluate("(-2) + 5") == 3.0
    assert evaluator.evaluate("--5") == 5.0

def test_floating_point_numbers(evaluator):
    """Test support for decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("10.0 / 4.0") == pytest.approx(2.5)

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (2 - 2)")

def test_error_handling(evaluator):
    """Test ValueError for empty, mismatched parentheses, and invalid tokens."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(3 + 4")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("3 + 4)")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 & 4")