from typing import Optional
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Enumeration of token types for the expression parser."""
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


@dataclass
class Token:
    """Represents a lexical token with a type and optional numeric value."""
    type: TokenType
    value: Optional[float] = None


class Tokenizer:
    """
    Lexical analyzer that converts input string into a stream of tokens.
    """
    
    def __init__(self, text: str) -> None:
        """
        Initialize the tokenizer with input text.
        
        Args:
            text: The input string to tokenize
        """
        self.text: str = text
        self.pos: int = 0
        self.current_char: Optional[str] = self.text[0] if text else None
    
    def error(self, message: str) -> None:
        """
        Raise a ValueError with the given message.
        
        Args:
            message: The error message to include
        """
        raise ValueError(message)
    
    def advance(self) -> None:
        """Move to the next character in the input."""
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None
    
    def skip_whitespace(self) -> None:
        """Skip over whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
    
    def number(self) -> Token:
        """
        Parse a numeric literal (integer or floating point).
        
        Returns:
            A Token of type NUMBER with the parsed float value
            
        Raises:
            ValueError: If the number format is invalid
        """
        result = ''
        has_dot = False
        
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if has_dot:
                    self.error("Invalid number format: multiple decimal points")
                has_dot = True
            result += self.current_char
            self.advance()
        
        if not result or result == '.':
            self.error("Invalid number format")
            
        try:
            value = float(result)
        except ValueError:
            self.error(f"Invalid number format: '{result}'")
            
        return Token(TokenType.NUMBER, value)
    
    def get_next_token(self) -> Token:
        """
        Return the next token from the input stream.
        
        Returns:
            The next Token, or EOF when input is exhausted
        """
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            if self.current_char.isdigit():
                return self.number()
            
            if self.current_char == '+':
                self.advance()
                return Token(TokenType.PLUS)
            
            if self.current_char == '-':
                self.advance()
                return Token(TokenType.MINUS)
            
            if self.current_char == '*':
                self.advance()
                return Token(TokenType.MULTIPLY)
            
            if self.current_char == '/':
                self.advance()
                return Token(TokenType.DIVIDE)
            
            if self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN)
            
            if self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN)
            
            self.error(f"Invalid character: '{self.current_char}'")
        
        return Token(TokenType.EOF)


class ExpressionEvaluator:
    """
    Recursive descent parser for mathematical expressions.
    
    Supports: +, -, *, / with correct precedence, parentheses, unary minus,
    and floating point numbers.
    """
    
    def __init__(self) -> None:
        """Initialize the evaluator."""
        self.tokenizer: Optional[Tokenizer] = None
        self.current_token: Optional[Token] = None
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The result of the evaluation as a float
            
        Raises:
            ValueError: If the expression is empty, invalid, contains mismatched
                       parentheses, division by zero, or other errors
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        self.tokenizer = Tokenizer(expr)
        self.current_token = self.tokenizer.get_next_token()
        
        result = self.parse_expression()
        
        if self.current_token.type != TokenType.EOF:
            raise ValueError(f"Unexpected token after expression: {self.current_token.type}")
        
        return result
    
    def parse_expression(self) -> float:
        """
        Parse an expression (handles + and - operators with lowest precedence).
        
        Grammar: expression := term (('+' | '-') term)*
        
        Returns:
            The evaluated value of the expression
        """
        result = self.parse_term()
        
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token
            self.current_token = self.tokenizer.get_next_token()
            
            right = self.parse_term()
            
            if op.type == TokenType.PLUS:
                result += right
            else:
                result -= right
        
        return result
    
    def parse_term(self) -> float:
        """
        Parse a term (handles * and / operators with medium precedence).
        
        Grammar: term := factor (('*' | '/') factor)*
        
        Returns:
            The evaluated value of the term
        """
        result = self.parse_factor()
        
        while self.current_token.type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            op = self.current_token
            self.current_token = self.tokenizer.get_next_token()
            
            right = self.parse_factor()
            
            if op.type == TokenType.MULTIPLY:
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        
        return result
    
    def parse_factor(self) -> float:
        """
        Parse a factor (handles unary operators, numbers, and parentheses).
        
        Grammar: factor := ('+' | '-') factor | NUMBER | '(' expression ')'
        
        Returns:
            The evaluated value of the factor
            
        Raises:
            ValueError: If parentheses are mismatched or unexpected token is found
        """
        if self.current_token.type == TokenType.PLUS:
            self.current_token = self.tokenizer.get_next_token()
            return self.parse_factor()
        
        if self.current_token.type == TokenType.MINUS:
            self.current_token = self.tokenizer.get_next_token()
            return -self.parse_factor()
        
        if self.current_token.type == TokenType.NUMBER:
            value = self.current_token.value
            self.current_token = self.tokenizer.get_next_token()
            return value
        
        if self.current_token.type == TokenType.LPAREN:
            self.current_token = self.tokenizer.get_next_token()
            result = self.parse_expression()
            
            if self.current_token.type != TokenType.RPAREN:
                raise ValueError("Mismatched parentheses: expected ')'")
            
            self.current_token = self.tokenizer.get_next_token()
            return result
        
        raise ValueError(f"Unexpected token: {self.current_token.type}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("15 / 4") == 3.75

def test_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0

def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_errors(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3")