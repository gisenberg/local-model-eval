from typing import List, Union, Iterator
from enum import Enum, auto

class TokenType(Enum):
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()

class Token:
    def __init__(self, type_: TokenType, value: Union[str, float]):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"

class Tokenizer:
    """Tokenizes the input string into a list of Token objects."""
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if self.text else None

    def error(self, msg: str):
        raise ValueError(f"Invalid token at position {self.pos}: {msg}")

    def advance(self):
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self) -> Token:
        """Parse a floating point number."""
        result = ""
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            result += self.current_char
            self.advance()
        
        if result.count('.') > 1:
            self.error("Invalid number format: multiple decimal points")
        
        if not result:
            self.error("Expected a number")
            
        return Token(TokenType.NUMBER, float(result))

    def get_next_token(self) -> Token:
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            if self.current_char == '+':
                self.advance()
                return Token(TokenType.PLUS, '+')
            
            if self.current_char == '-':
                self.advance()
                return Token(TokenType.MINUS, '-')
            
            if self.current_char == '*':
                self.advance()
                return Token(TokenType.MUL, '*')
            
            if self.current_char == '/':
                self.advance()
                return Token(TokenType.DIV, '/')
            
            if self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN, '(')
            
            if self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN, ')')
            
            if self.current_char.isdigit() or (self.current_char == '.' and self.pos + 1 < len(self.text) and self.text[self.pos+1].isdigit()):
                return self.number()
            
            self.error(f"Unknown character: '{self.current_char}'")
        
        return Token(TokenType.EOF, None)

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers
    
    Raises:
        ValueError: For mismatched parentheses, division by zero, or invalid syntax.
    """

    def __init__(self):
        self.lexer = None
        self.current_token = None

    def evaluate(self, expr: str) -> float:
        """
        Public method to evaluate a mathematical expression string.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")
        
        self.lexer = Tokenizer(expr)
        self.current_token = self.lexer.get_next_token()
        
        result = self._parse_expression()
        
        if self.current_token.type != TokenType.EOF:
            raise ValueError(f"Unexpected token after expression: {self.current_token}")
            
        return result

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Grammar: expression -> term (('+' | '-') term)*
        """
        node = self._parse_term()
        
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token
            self.advance()
            right = self._parse_term()
            
            if op.type == TokenType.PLUS:
                node = node + right
            else:
                node = node - right
                
        return node

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Grammar: term -> factor (('*' | '/') factor)*
        """
        node = self._parse_factor()
        
        while self.current_token.type in (TokenType.MUL, TokenType.DIV):
            op = self.current_token
            self.advance()
            right = self._parse_factor()
            
            if op.type == TokenType.MUL:
                node = node * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                node = node / right
                
        return node

    def _parse_factor(self) -> float:
        """
        Parses numbers, parentheses, and unary operators.
        Grammar: factor -> NUMBER | '(' expression ')' | '-' factor | '+' factor
        """
        token = self.current_token
        
        if token.type == TokenType.NUMBER:
            self.advance()
            return token.value
        
        if token.type == TokenType.PLUS:
            self.advance()
            return self._parse_factor()
        
        if token.type == TokenType.MINUS:
            self.advance()
            return -self._parse_factor()
        
        if token.type == TokenType.LPAREN:
            self.advance()
            result = self._parse_expression()
            
            if self.current_token.type != TokenType.RPAREN:
                raise ValueError("Mismatched parentheses: expected ')'")
            
            self.advance()
            return result
        
        raise ValueError(f"Unexpected token in factor: {token}")

    def advance(self):
        """Moves to the next token in the stream."""
        self.current_token = self.lexer.get_next_token()

# --- Test Suite ---
if __name__ == "__main__":
    import pytest

    def test_basic_arithmetic():
        ev = ExpressionEvaluator()
        assert ev.evaluate("2 + 3") == 5.0
        assert ev.evaluate("10 - 4") == 6.0
        assert ev.evaluate("3 * 4") == 12.0
        assert ev.evaluate("10 / 2") == 5.0
        assert ev.evaluate("1.5 + 2.5") == 4.0

    def test_precedence():
        ev = ExpressionEvaluator()
        # Multiplication before addition
        assert ev.evaluate("2 + 3 * 4") == 14.0
        # Division before subtraction
        assert ev.evaluate("10 - 6 / 2") == 7.0
        # Left to right for same precedence
        assert ev.evaluate("10 - 2 - 3") == 5.0
        assert ev.evaluate("100 / 2 / 5") == 10.0

    def test_parentheses():
        ev = ExpressionEvaluator()
        # Override precedence
        assert ev.evaluate("(2 + 3) * 4") == 20.0
        assert ev.evaluate("2 * (3 + 4)") == 14.0
        # Nested parentheses
        assert ev.evaluate("((2 + 3) * 4) - 1") == 19.0

    def test_unary_minus():
        ev = ExpressionEvaluator()
        assert ev.evaluate("-5") == -5.0
        assert ev.evaluate("3 + -2") == 1.0
        assert ev.evaluate("-(2 + 1)") == -3.0
        assert ev.evaluate("-(-5)") == 5.0
        assert ev.evaluate("2 * -3") == -6.0

    def test_error_cases():
        ev = ExpressionEvaluator()
        
        # Empty expression
        try:
            ev.evaluate("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "empty" in str(e).lower()
            
        # Mismatched parentheses
        try:
            ev.evaluate("(2 + 3")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "mismatched" in str(e).lower()
            
        # Division by zero
        try:
            ev.evaluate("5 / 0")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "division by zero" in str(e).lower()
            
        # Invalid token
        try:
            ev.evaluate("2 + x")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unknown" in str(e).lower() or "invalid" in str(e).lower()

    # Run tests if executed directly
    # pytest tests -v
    print("Running tests...")
    test_basic_arithmetic()
    test_precedence()
    test_parentheses()
    test_unary_minus()
    test_error_cases()
    print("All tests passed!")