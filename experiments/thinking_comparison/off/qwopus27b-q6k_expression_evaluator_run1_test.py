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
    value: Union[float, str]

class ExpressionEvaluator:
    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.
        
        Args:
            expr: A string containing a mathematical expression with +, -, *, /, 
                  parentheses, and unary minus.
                  
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is invalid, empty, has mismatched 
                       parentheses, or contains division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")
            
        self._tokens = self._tokenize(expr)
        if not self._tokens:
            raise ValueError("Expression cannot be empty")
            
        self._pos = 0
        result = self._parse_expression()
        
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token '{self._tokens[self._pos].value}' at position {self._pos}")
            
        return result
    
    def _tokenize(self, expr: str) -> List[Token]:
        """
        Convert expression string into a list of tokens.
        
        Args:
            expr: The input expression string.
            
        Returns:
            A list of Token objects representing the tokens in the expression.
            
        Raises:
            ValueError: If the expression contains invalid characters or number formats.
        """
        tokens: List[Token] = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
                
            if char.isdigit() or char == '.':
                # Parse number
                start = i
                has_dot = False
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format: multiple decimal points at position {i}")
                        has_dot = True
                    i += 1
                
                num_str = expr[start:i]
                if num_str == '.':
                    raise ValueError(f"Invalid number format: '.' at position {start}")
                if num_str.startswith('.') and len(num_str) == 1:
                    raise ValueError(f"Invalid number format: '.' at position {start}")
                    
                try:
                    value = float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number format: '{num_str}' at position {start}")
                    
                tokens.append(Token(TokenType.NUMBER, value))
                continue
                
            if char == '+':
                tokens.append(Token(TokenType.PLUS, '+'))
                i += 1
            elif char == '-':
                tokens.append(Token(TokenType.MINUS, '-'))
                i += 1
            elif char == '*':
                tokens.append(Token(TokenType.MULTIPLY, '*'))
                i += 1
            elif char == '/':
                tokens.append(Token(TokenType.DIVIDE, '/'))
                i += 1
            elif char == '(':
                tokens.append(Token(TokenType.LPAREN, '('))
                i += 1
            elif char == ')':
                tokens.append(Token(TokenType.RPAREN, ')'))
                i += 1
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")
                
        tokens.append(Token(TokenType.EOF, ''))
        return tokens
    
    def _current_token(self) -> Token:
        """Get the current token at position _pos."""
        if self._pos >= len(self._tokens):
            return self._tokens[-1]  # EOF
        return self._tokens[self._pos]
    
    def _consume(self, expected_type: Optional[TokenType] = None) -> Token:
        """
        Consume the current token and advance position.
        
        Args:
            expected_type: Optional expected token type for validation.
            
        Returns:
            The consumed token.
            
        Raises:
            ValueError: If expected_type is provided and current token doesn't match.
        """
        token = self._current_token()
        if expected_type and token.type != expected_type:
            raise ValueError(f"Expected {expected_type.name} but got {token.type.name} '{token.value}'")
        self._pos += 1
        return token
    
    def _parse_expression(self) -> float:
        """
        Parse addition and subtraction (lowest precedence).
        Expression := Term (('+' | '-') Term)*
        """
        left = self._parse_term()
        
        while self._current_token().type in (TokenType.PLUS, TokenType.MINUS):
            op = self._consume()
            right = self._parse_term()
            if op.type == TokenType.PLUS:
                left = left + right
            else:
                left = left - right
                
        return left
    
    def _parse_term(self) -> float:
        """
        Parse multiplication and division (higher precedence).
        Term := Factor (('*' | '/') Factor)*
        """
        left = self._parse_factor()
        
        while self._current_token().type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            op = self._consume()
            right = self._parse_factor()
            if op.type == TokenType.MULTIPLY:
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
                
        return left
    
    def _parse_factor(self) -> float:
        """
        Parse numbers, parentheses, and unary minus (highest precedence).
        Factor := Number | '(' Expression ')' | '-' Factor
        """
        token = self._current_token()
        
        if token.type == TokenType.NUMBER:
            self._consume()
            return token.value
            
        if token.type == TokenType.LPAREN:
            self._consume()
            result = self._parse_expression()
            if self._current_token().type != TokenType.RPAREN:
                raise ValueError("Missing closing parenthesis ')'")
            self._consume()
            return result
            
        if token.type == TokenType.MINUS:
            self._consume()
            # Unary minus: parse factor recursively
            return -self._parse_factor()
            
        raise ValueError(f"Unexpected token '{token.value}' at position {self._pos}")


# Pytest tests
import pytest

class TestExpressionEvaluator:
    def setup_method(self):
        self.evaluator = ExpressionEvaluator()
    
    def test_basic_arithmetic(self):
        """Test basic +, -, *, / operations."""
        assert self.evaluator.evaluate("2 + 3") == 5.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("3 * 4") == 12.0
        assert self.evaluator.evaluate("15 / 3") == 5.0
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0  # precedence
    
    def test_precedence(self):
        """Test operator precedence: * and / before + and -."""
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
        assert self.evaluator.evaluate("2 * 3 + 4") == 10.0
        assert self.evaluator.evaluate("10 / 2 + 3") == 8.0
        assert self.evaluator.evaluate("2 + 3 * 4 - 5") == 9.0
        assert self.evaluator.evaluate("10 - 2 * 3 + 4") == 6.0
    
    def test_parentheses(self):
        """Test parentheses for grouping."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert self.evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert self.evaluator.evaluate("(2 + 3) * (4 + 5)") == 45.0
        assert self.evaluator.evaluate("((2 + 3))") == 5.0
        assert self.evaluator.evaluate("-(2 + 3)") == -5.0
    
    def test_unary_minus(self):
        """Test unary minus operator."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("--3") == 3.0
        assert self.evaluator.evaluate("-3 * 4") == -12.0
        assert self.evaluator.evaluate("3 * -4") == -12.0
        assert self.evaluator.evaluate("-3 * -4") == 12.0
        assert self.evaluator.evaluate("-(2 + 3)") == -5.0
        assert self.evaluator.evaluate("-(-3)") == 3.0
    
    def test_error_cases(self):
        """Test error handling."""
        # Empty expression
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate("")
        
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate("   ")
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="parenthes"):
            self.evaluator.evaluate("(2 + 3")
        
        with pytest.raises(ValueError, match="parenthes"):
            self.evaluator.evaluate("2 + 3)")
        
        # Division by zero
        with pytest.raises(ValueError, match="zero"):
            self.evaluator.evaluate("1 / 0")
        
        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("2 + a")
        
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("2++3")