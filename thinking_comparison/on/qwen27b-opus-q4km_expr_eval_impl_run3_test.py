from __future__ import annotations
from typing import List, Tuple, Optional
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


class Token:
    """Represents a single token in the expression."""
    
    def __init__(self, token_type: TokenType, value: Optional[float] = None):
        self.type = token_type
        self.value = value
    
    def __repr__(self) -> str:
        if self.type == TokenType.NUMBER:
            return f"Token(NUMBER, {self.value})"
        return f"Token({self.type.name})"


class Tokenizer:
    """Converts an expression string into a list of tokens."""
    
    def __init__(self, expression: str):
        self.expression = expression
        self.pos = 0
        self.length = len(expression)
    
    def _peek(self) -> str:
        """Returns the current character without advancing position."""
        if self.pos >= self.length:
            return '\0'
        return self.expression[self.pos]
    
    def _advance(self) -> str:
        """Returns the current character and advances position."""
        char = self._peek()
        self.pos += 1
        return char
    
    def _skip_whitespace(self) -> None:
        """Skips over any whitespace characters."""
        while self.pos < self.length and self.expression[self.pos].isspace():
            self.pos += 1
    
    def _read_number(self) -> Token:
        """Reads a number (integer or floating point) from the expression."""
        start_pos = self.pos
        has_decimal = False
        
        while self.pos < self.length:
            char = self.expression[self.pos]
            if char.isdigit():
                self.pos += 1
            elif char == '.' and not has_decimal:
                has_decimal = True
                self.pos += 1
            else:
                break
        
        number_str = self.expression[start_pos:self.pos]
        return Token(TokenType.NUMBER, float(number_str))
    
    def tokenize(self) -> List[Token]:
        """Converts the expression string into a list of tokens."""
        tokens: List[Token] = []
        
        while True:
            self._skip_whitespace()
            
            if self.pos >= self.length:
                break
            
            char = self._peek()
            
            if char.isdigit() or char == '.':
                tokens.append(self._read_number())
            elif char == '+':
                self._advance()
                tokens.append(Token(TokenType.PLUS))
            elif char == '-':
                self._advance()
                tokens.append(Token(TokenType.MINUS))
            elif char == '*':
                self._advance()
                tokens.append(Token(TokenType.MULTIPLY))
            elif char == '/':
                self._advance()
                tokens.append(Token(TokenType.DIVIDE))
            elif char == '(':
                self._advance()
                tokens.append(Token(TokenType.LPAREN))
            elif char == ')':
                self._advance()
                tokens.append(Token(TokenType.RPAREN))
            else:
                raise ValueError(f"Invalid character '{char}' at position {self.pos}")
        
        tokens.append(Token(TokenType.EOF))
        return tokens


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus
    - Floating point numbers
    
    Grammar:
        Expression -> Term (('+' | '-') Term)*
        Term -> Factor (('*' | '/') Factor)*
        Factor -> Number | '(' Expression ')' | '-' Factor
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.pos: int = 0
    
    def _current_token(self) -> Token:
        """Returns the current token being examined."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF)
    
    def _advance(self) -> Token:
        """Returns the current token and advances to the next one."""
        token = self._current_token()
        if self.pos < len(self.tokens):
            self.pos += 1
        return token
    
    def _error_expected(self, expected: str) -> None:
        """Raises an error when an unexpected token is found."""
        current = self._current_token()
        if current.type == TokenType.EOF:
            raise ValueError(f"Unexpected end of expression, expected {expected}")
        raise ValueError(f"Unexpected token '{current}', expected {expected}")
    
    def _parse_expression(self) -> float:
        """
        Parses an expression (handles + and - operators).
        
        Expression -> Term (('+' | '-') Term)*
        """
        result = self._parse_term()
        
        while True:
            token = self._current_token()
            if token.type == TokenType.PLUS:
                self._advance()
                right = self._parse_term()
                result += right
            elif token.type == TokenType.MINUS:
                self._advance()
                right = self._parse_term()
                result -= right
            else:
                break
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parses a term (handles * and / operators).
        
        Term -> Factor (('*' | '/') Factor)*
        """
        result = self._parse_factor()
        
        while True:
            token = self._current_token()
            if token.type == TokenType.MULTIPLY:
                self._advance()
                right = self._parse_factor()
                result *= right
            elif token.type == TokenType.DIVIDE:
                self._advance()
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
            else:
                break
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parses a factor (numbers, parentheses, unary minus).
        
        Factor -> Number | '(' Expression ')' | '-' Factor
        """
        token = self._current_token()
        
        # Handle unary minus
        if token.type == TokenType.MINUS:
            self._advance()
            return -self._parse_factor()
        
        # Handle numbers
        if token.type == TokenType.NUMBER:
            self._advance()
            return token.value  # type: ignore
        
        # Handle parenthesized expressions
        if token.type == TokenType.LPAREN:
            self._advance()
            result = self._parse_expression()
            
            if self._current_token().type != TokenType.RPAREN:
                raise ValueError("Expected ')' to close parenthesis")
            
            self._advance()
            return result
        
        # If we get here, we have an unexpected token
        if token.type == TokenType.EOF:
            raise ValueError("Unexpected end of expression")
        elif token.type == TokenType.RPAREN:
            raise ValueError("Unexpected ')' - possible mismatched parentheses")
        else:
            raise ValueError(f"Unexpected token '{token}'")
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The floating point result of evaluating the expression
            
        Raises:
            ValueError: For invalid expressions, mismatched parentheses,
                       division by zero, or empty expressions
        """
        # Check for empty or whitespace-only expression
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        # Tokenize the expression
        tokenizer = Tokenizer(expr)
        self.tokens = tokenizer.tokenize()
        self.pos = 0
        
        # Parse and evaluate
        result = self._parse_expression()
        
        # Verify we consumed all tokens (except EOF)
        if self._current_token().type != TokenType.EOF:
            token = self._current_token()
            raise ValueError(f"Unexpected token '{token}' after expression")
        
        return result

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