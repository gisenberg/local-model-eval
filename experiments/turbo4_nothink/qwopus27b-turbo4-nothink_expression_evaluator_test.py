from typing import List, Union, Optional
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
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
    type: TokenType
    value: Optional[Union[float, str]] = None


class Tokenizer:
    """Tokenizes mathematical expressions into tokens."""
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if text else None
    
    def error(self, message: str) -> None:
        """Raise a ValueError with the given message."""
        raise ValueError(message)
    
    def advance(self) -> None:
        """Advance to the next character."""
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
        """Parse a number (integer or floating point)."""
        result = ""
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
    A recursive descent parser for mathematical expressions.
    
    Supports: +, -, *, / with correct precedence, parentheses, unary minus,
    and floating point numbers.
    """
    
    def __init__(self):
        self.tokenizer: Optional[Tokenizer] = None
        self.current_token: Optional[Token] = None
    
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
        
        result = self._parse_expression()
        
        if self.current_token.type != TokenType.EOF:
            self.tokenizer.error(f"Unexpected token: {self.current_token.type}")
        
        return result
    
    def _parse_expression(self) -> float:
        """
        Parse an expression: Term (('+' | '-') Term)*
        Handles addition and subtraction with lowest precedence.
        """
        result = self._parse_term()
        
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token
            self._advance()
            right = self._parse_term()
            
            if op.type == TokenType.PLUS:
                result += right
            else:
                result -= right
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse a term: Factor (('*' | '/') Factor)*
        Handles multiplication and division with higher precedence than +/-
        """
        result = self._parse_factor()
        
        while self.current_token.type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            op = self.current_token
            self._advance()
            right = self._parse_factor()
            
            if op.type == TokenType.MULTIPLY:
                result *= right
            else:
                if right == 0:
                    self.tokenizer.error("Division by zero")
                result /= right
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parse a factor: Number | '(' Expression ')' | '-' Factor
        Handles numbers, parentheses, and unary minus with highest precedence.
        """
        token = self.current_token
        
        if token.type == TokenType.NUMBER:
            self._advance()
            return token.value  # type: ignore
        
        if token.type == TokenType.LPAREN:
            self._advance()
            result = self._parse_expression()
            
            if self.current_token.type != TokenType.RPAREN:
                self.tokenizer.error("Mismatched parentheses: expected ')'")
            
            self._advance()
            return result
        
        if token.type == TokenType.MINUS:
            self._advance()
            return -self._parse_factor()
        
        self.tokenizer.error(f"Unexpected token: {token.type}")
    
    def _advance(self) -> None:
        """Advance to the next token."""
        self.current_token = self.tokenizer.get_next_token()


# Pytest tests
import pytest


class TestExpressionEvaluator:
    """Test suite for ExpressionEvaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ExpressionEvaluator()
    
    def test_basic_arithmetic(self):
        """Test basic addition, subtraction, multiplication, and division."""
        assert self.evaluator.evaluate("2 + 3") == 5.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("3 * 4") == 12.0
        assert self.evaluator.evaluate("15 / 3") == 5.0
        assert self.evaluator.evaluate("2.5 + 3.5") == 6.0
    
    def test_precedence(self):
        """Test operator precedence: * and / before + and -."""
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
        assert self.evaluator.evaluate("2 * 3 + 4") == 10.0
        assert self.evaluator.evaluate("10 - 2 * 3") == 4.0
        assert self.evaluator.evaluate("10 / 2 + 3") == 8.0
        assert self.evaluator.evaluate("2 + 3 * 4 - 5") == 9.0
        assert self.evaluator.evaluate("10 / 2 * 3") == 15.0
    
    def test_parentheses(self):
        """Test parentheses for grouping and overriding precedence."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert self.evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20.0
        assert self.evaluator.evaluate("(2 + 3) * (4 + 5)") == 45.0
        assert self.evaluator.evaluate("2 + (3 * (4 + 5))") == 29.0
    
    def test_unary_minus(self):
        """Test unary minus operator with various precedences."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-(2 + 1)") == -3.0
        assert self.evaluator.evaluate("2 * -3") == -6.0
        assert self.evaluator.evaluate("-2 * 3") == -6.0
        assert self.evaluator.evaluate("--3") == 3.0
        assert self.evaluator.evaluate("2 + -3") == -1.0
        assert self.evaluator.evaluate("2 - -3") == 5.0
        assert self.evaluator.evaluate("-2 * -3") == 6.0
    
    def test_error_cases(self):
        """Test error handling for invalid inputs."""
        # Empty expression
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("")
        
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("   ")
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("(2 + 3")
        
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("2 + 3)")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("1 / 0")
        
        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 + a")
        
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 + * 3")
        
        # Unexpected token (trailing operator)
        with pytest.raises(ValueError, match="Unexpected token"):
            self.evaluator.evaluate("2 + 3 +")