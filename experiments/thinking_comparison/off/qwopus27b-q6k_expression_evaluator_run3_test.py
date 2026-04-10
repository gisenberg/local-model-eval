from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Union

class TokenType(Enum):
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()

@dataclass
class Token:
    type: TokenType
    value: Union[float, str]

class Tokenizer:
    """Tokenizer for mathematical expressions."""
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if text else None
    
    def error(self, message: str) -> None:
        """Raise a ValueError with the given message."""
        raise ValueError(message)
    
    def advance(self) -> None:
        """Move to the next character."""
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None
    
    def skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
    
    def number(self) -> Token:
        """Parse a number (integer or float)."""
        result = ""
        has_dot = False
        
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if has_dot:
                    self.error("Invalid number format: multiple decimal points")
                has_dot = True
            result += self.current_char
            self.advance()
        
        return Token(TokenType.NUMBER, float(result))
    
    def get_next_token(self) -> Token:
        """Get the next token from the input."""
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            if self.current_char.isdigit() or self.current_char == '.':
                return self.number()
            
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
            
            self.error(f"Invalid character: '{self.current_char}'")
        
        return Token(TokenType.EOF, None)

class ExpressionEvaluator:
    """
    A recursive descent parser for mathematical expressions.
    
    Supports +, -, *, / with correct precedence, parentheses, unary minus,
    and floating point numbers.
    """
    
    def __init__(self):
        self.tokenizer: Tokenizer = None
        self.current_token: Token = None
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result as a float.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The result of the evaluation as a float
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        self.tokenizer = Tokenizer(expr)
        self.current_token = self.tokenizer.get_next_token()
        
        result = self.parse_expression()
        
        if self.current_token.type != TokenType.EOF:
            self.tokenizer.error(f"Unexpected token: '{self.current_token.value}'")
        
        return result
    
    def eat(self, token_type: TokenType) -> None:
        """
        Consume the current token if it matches the expected type.
        
        Args:
            token_type: The expected token type
            
        Raises:
            ValueError: If the current token doesn't match the expected type
        """
        if self.current_token.type == token_type:
            self.current_token = self.tokenizer.get_next_token()
        else:
            self.tokenizer.error(f"Expected {token_type.name}, got {self.current_token.type.name}")
    
    def parse_expression(self) -> float:
        """
        Parse an expression: Term (('+' | '-') Term)*
        Handles addition and subtraction with left associativity.
        """
        result = self.parse_term()
        
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            if self.current_token.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)
                result += self.parse_term()
            elif self.current_token.type == TokenType.MINUS:
                self.eat(TokenType.MINUS)
                result -= self.parse_term()
        
        return result
    
    def parse_term(self) -> float:
        """
        Parse a term: Factor (('*' | '/') Factor)*
        Handles multiplication and division with left associativity.
        """
        result = self.parse_factor()
        
        while self.current_token.type in (TokenType.MUL, TokenType.DIV):
            if self.current_token.type == TokenType.MUL:
                self.eat(TokenType.MUL)
                result *= self.parse_factor()
            elif self.current_token.type == TokenType.DIV:
                self.eat(TokenType.DIV)
                divisor = self.parse_factor()
                if divisor == 0:
                    self.tokenizer.error("Division by zero")
                result /= divisor
        
        return result
    
    def parse_factor(self) -> float:
        """
        Parse a factor: Number | '(' Expression ')' | '-' Factor
        Handles numbers, parentheses, and unary minus.
        """
        if self.current_token.type == TokenType.NUMBER:
            value = self.current_token.value
            self.eat(TokenType.NUMBER)
            return value
        
        elif self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            result = self.parse_expression()
            if self.current_token.type != TokenType.RPAREN:
                self.tokenizer.error("Expected closing parenthesis")
            self.eat(TokenType.RPAREN)
            return result
        
        elif self.current_token.type == TokenType.MINUS:
            # Unary minus
            self.eat(TokenType.MINUS)
            return -self.parse_factor()
        
        else:
            self.tokenizer.error(f"Unexpected token: {self.current_token.type.name}")


# Pytest tests
import pytest

class TestExpressionEvaluator:
    def setup_method(self):
        self.evaluator = ExpressionEvaluator()
    
    def test_basic_arithmetic(self):
        """Test basic addition, subtraction, multiplication, and division."""
        assert self.evaluator.evaluate("2 + 3") == 5
        assert self.evaluator.evaluate("10 - 4") == 6
        assert self.evaluator.evaluate("3 * 4") == 12
        assert self.evaluator.evaluate("15 / 3") == 5.0
        assert self.evaluator.evaluate("3.14 * 2") == 6.28
    
    def test_operator_precedence(self):
        """Test that multiplication/division have higher precedence than addition/subtraction."""
        assert self.evaluator.evaluate("2 + 3 * 4") == 14  # 2 + 12
        assert self.evaluator.evaluate("2 * 3 + 4") == 10  # 6 + 4
        assert self.evaluator.evaluate("10 / 2 + 3") == 8.0  # 5 + 3
        assert self.evaluator.evaluate("2 + 3 * 4 - 5 / 5") == 13.0  # 2 + 12 - 1
    
    def test_parentheses(self):
        """Test parentheses for grouping and overriding precedence."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20
        assert self.evaluator.evaluate("2 * (3 + 4)") == 14
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20
        assert self.evaluator.evaluate("(2 + 3) * (4 - 1)") == 15
    
    def test_unary_minus(self):
        """Test unary minus for negative numbers and negation of expressions."""
        assert self.evaluator.evaluate("-3") == -3
        assert self.evaluator.evaluate("-3 + 5") == 2
        assert self.evaluator.evaluate("5 - -3") == 8
        assert self.evaluator.evaluate("-(2 + 3)") == -5
        assert self.evaluator.evaluate("-(-3)") == 3
        assert self.evaluator.evaluate("-2 * 3") == -6
    
    def test_error_cases(self):
        """Test various error conditions."""
        # Empty expression
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("")
        
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("   ")
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Expected closing parenthesis"):
            self.evaluator.evaluate("(2 + 3")
        
        with pytest.raises(ValueError, match="Unexpected token"):
            self.evaluator.evaluate("2 + 3)")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("5 / 0")
        
        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 + a")
        
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 + * 3")