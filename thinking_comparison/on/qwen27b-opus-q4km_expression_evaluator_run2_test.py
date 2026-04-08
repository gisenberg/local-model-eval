from typing import List, Iterator, Tuple, Optional
import re
from enum import Enum, auto


class TokenType(Enum):
    """Enumeration of token types."""
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
    
    def __init__(self, type_: TokenType, value: Optional[float] = None):
        self.type = type_
        self.value = value
    
    def __repr__(self) -> str:
        if self.type == TokenType.NUMBER:
            return f"Token(NUMBER, {self.value})"
        return f"Token({self.type.name})"


class Tokenizer:
    """Converts an expression string into a list of tokens."""
    
    def __init__(self, expr: str):
        self.expr = expr
        self.pos = 0
        self.length = len(expr)
    
    def _current_char(self) -> str:
        """Returns the current character or empty string if at end."""
        if self.pos < self.length:
            return self.expr[self.pos]
        return ""
    
    def _advance(self) -> None:
        """Moves to the next character."""
        self.pos += 1
    
    def _skip_whitespace(self) -> None:
        """Skips over whitespace characters."""
        while self._current_char().isspace():
            self._advance()
    
    def _read_number(self) -> float:
        """Reads a number (integer or float) from the expression."""
        result = ""
        has_dot = False
        
        while self._current_char():
            char = self._current_char()
            if char.isdigit():
                result += char
                self._advance()
            elif char == '.' and not has_dot:
                has_dot = True
                result += char
                self._advance()
            else:
                break
        
        return float(result)
    
    def tokenize(self) -> List[Token]:
        """Converts the expression string into a list of tokens."""
        tokens: List[Token] = []
        
        while self.pos < self.length:
            self._skip_whitespace()
            
            if self.pos >= self.length:
                break
            
            char = self._current_char()
            
            if char.isdigit() or char == '.':
                # Read a number
                value = self._read_number()
                tokens.append(Token(TokenType.NUMBER, value))
            
            elif char == '+':
                tokens.append(Token(TokenType.PLUS))
                self._advance()
            
            elif char == '-':
                tokens.append(Token(TokenType.MINUS))
                self._advance()
            
            elif char == '*':
                tokens.append(Token(TokenType.MULTIPLY))
                self._advance()
            
            elif char == '/':
                tokens.append(Token(TokenType.DIVIDE))
                self._advance()
            
            elif char == '(':
                tokens.append(Token(TokenType.LPAREN))
                self._advance()
            
            elif char == ')':
                tokens.append(Token(TokenType.RPAREN))
                self._advance()
            
            else:
                raise ValueError(f"Invalid character: '{char}'")
        
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
    
    Example:
        >>> evaluator = ExpressionEvaluator()
        >>> evaluator.evaluate("2 + 3 * 4")
        14.0
        >>> evaluator.evaluate("(2 + 3) * 4")
        20.0
        >>> evaluator.evaluate("-3 + 5")
        2.0
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.pos = 0
    
    def _current_token(self) -> Token:
        """Returns the current token being processed."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF)
    
    def _advance(self) -> None:
        """Moves to the next token."""
        self.pos += 1
    
    def _eat(self, token_type: TokenType) -> None:
        """
        Consumes the current token if it matches the expected type.
        Raises ValueError if the token doesn't match.
        """
        token = self._current_token()
        if token.type == token_type:
            self._advance()
        else:
            raise ValueError(
                f"Expected {token_type.name}, got {token.type.name}"
            )
    
    def _parse_expression(self) -> float:
        """
        Parses an expression: handles + and - operators (lowest precedence).
        
        Grammar: expression → term (('+' | '-') term)*
        """
        result = self._parse_term()
        
        while True:
            token = self._current_token()
            if token.type == TokenType.PLUS:
                self._advance()
                result += self._parse_term()
            elif token.type == TokenType.MINUS:
                self._advance()
                result -= self._parse_term()
            else:
                break
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parses a term: handles * and / operators (higher precedence).
        
        Grammar: term → factor (('*' | '/') factor)*
        """
        result = self._parse_factor()
        
        while True:
            token = self._current_token()
            if token.type == TokenType.MULTIPLY:
                self._advance()
                result *= self._parse_factor()
            elif token.type == TokenType.DIVIDE:
                self._advance()
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                result /= divisor
            else:
                break
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parses a factor: handles unary minus, parentheses, and numbers.
        
        Grammar: factor → '-' factor | '(' expression ')' | number
        """
        token = self._current_token()
        
        # Handle unary minus
        if token.type == TokenType.MINUS:
            self._advance()
            return -self._parse_factor()
        
        # Handle parentheses
        if token.type == TokenType.LPAREN:
            self._advance()
            result = self._parse_expression()
            self._eat(TokenType.RPAREN)
            return result
        
        # Handle numbers
        if token.type == TokenType.NUMBER:
            self._advance()
            return token.value  # type: ignore
        
        # Unexpected token
        raise ValueError(f"Unexpected token: {token.type.name}")
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression and returns the result.
        
        Args:
            expr: A string containing a mathematical expression.
            
        Returns:
            The result of evaluating the expression as a float.
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains
                       mismatched parentheses, division by zero, or
                       invalid tokens.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        # Tokenize the expression
        tokenizer = Tokenizer(expr)
        self.tokens = tokenizer.tokenize()
        self.pos = 0
        
        # Parse and evaluate
        result = self._parse_expression()
        
        # Check for extra tokens (unmatched closing parentheses, etc.)
        if self._current_token().type != TokenType.EOF:
            raise ValueError(
                f"Unexpected token after expression: {self._current_token().type.name}"
            )
        
        return result


# Test cases
if __name__ == "__main__":
    evaluator = ExpressionEvaluator()
    
    # Basic arithmetic
    print(f"2 + 3 = {evaluator.evaluate('2 + 3')}")  # 5.0
    print(f"10 - 4 = {evaluator.evaluate('10 - 4')}")  # 6.0
    print(f"3 * 4 = {evaluator.evaluate('3 * 4')}")  # 12.0
    print(f"15 / 3 = {evaluator.evaluate('15 / 3')}")  # 5.0
    
    # Operator precedence
    print(f"2 + 3 * 4 = {evaluator.evaluate('2 + 3 * 4')}")  # 14.0
    print(f"2 * 3 + 4 * 5 = {evaluator.evaluate('2 * 3 + 4 * 5')}")  # 26.0
    
    # Parentheses
    print(f"(2 + 3) * 4 = {evaluator.evaluate('(2 + 3) * 4')}")  # 20.0
    print(f"((2 + 3) * 4) / 2 = {evaluator.evaluate('((2 + 3) * 4) / 2')}")  # 10.0
    
    # Unary minus
    print(f"-3 = {evaluator.evaluate('-3')}")  # -3.0
    print(f"-3 + 5 = {evaluator.evaluate('-3 + 5')}")  # 2.0
    print(f"-(2 + 1) = {evaluator.evaluate('-(2 + 1)')}")  # -3.0
    print(f"-(-5) = {evaluator.evaluate('-(-5)')}")  # 5.0
    
    # Floating point
    print(f"3.14 + 2.86 = {evaluator.evaluate('3.14 + 2.86')}")  # 6.0
    print(f"1.5 * 2 = {evaluator.evaluate('1.5 * 2')}")  # 3.0
    
    # Complex expression
    print(f"2 + 3 * (4 - 1) / 3 = {evaluator.evaluate('2 + 3 * (4 - 1) / 3')}")  # 5.0

import pytest



class TestExpressionEvaluator:
    """Test suite for ExpressionEvaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ExpressionEvaluator()
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert self.evaluator.evaluate("2 + 3") == 5.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("3 * 4") == 12.0
        assert self.evaluator.evaluate("15 / 3") == 5.0
        assert self.evaluator.evaluate("2.5 + 3.5") == 6.0
    
    def test_operator_precedence(self):
        """Test that operator precedence is correctly handled."""
        # Multiplication before addition
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
        # Multiplication before subtraction
        assert self.evaluator.evaluate("10 - 2 * 3") == 4.0
        # Division before addition
        assert self.evaluator.evaluate("10 + 20 / 4") == 15.0
        # Complex precedence
        assert self.evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
        assert self.evaluator.evaluate("2 + 3 * 4 - 5 / 5") == 13.0
    
    def test_parentheses(self):
        """Test parentheses for grouping."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20.0
        assert self.evaluator.evaluate("(2 + 3) * (4 + 5)") == 45.0
        assert self.evaluator.evaluate("((2 + 3) * (4 + 5)) / 5") == 9.0
        assert self.evaluator.evaluate("1 + (2 * (3 + (4 * 5)))") == 43.0
    
    def test_unary_minus(self):
        """Test unary minus operator."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-3 + 5") == 2.0
        assert self.evaluator.evaluate("5 + -3") == 2.0
        assert self.evaluator.evaluate("-(2 + 1)") == -3.0
        assert self.evaluator.evaluate("-(-5)") == 5.0
        assert self.evaluator.evaluate("--3") == 3.0
        assert self.evaluator.evaluate("-2 * 3") == -6.0
        assert self.evaluator.evaluate("2 * -3") == -6.0
        assert self.evaluator.evaluate("-2 * -3") == 6.0
    
    def test_error_cases(self):
        """Test error handling for invalid inputs."""
        # Empty expression
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("")
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("   ")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("5 / 0")
        
        # Mismatched parentheses - extra closing
        with pytest.raises(ValueError, match="Unexpected"):
            self.evaluator.evaluate("(2 + 3))")
        
        # Mismatched parentheses - extra opening
        with pytest.raises(ValueError, match="Expected"):
            self.evaluator.evaluate("(2 + 3")
        
        # Invalid token
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 + a")
        
        # Invalid token - special character
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 @ 3")