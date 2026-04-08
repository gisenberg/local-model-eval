from typing import List, Union, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import pytest


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


class Tokenizer:
    """Converts a string expression into a list of tokens."""
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)
    
    def peek(self) -> str:
        """Return the current character without advancing."""
        if self.pos >= self.length:
            return '\0'
        return self.text[self.pos]
    
    def advance(self) -> str:
        """Advance to the next character and return the current one."""
        char = self.peek()
        self.pos += 1
        return char
    
    def skip_whitespace(self) -> None:
        """Skip over whitespace characters."""
        while self.pos < self.length and self.text[self.pos].isspace():
            self.pos += 1
    
    def read_number(self) -> Token:
        """Read a number (integer or float) from the input."""
        start = self.pos
        has_dot = False
        
        while self.pos < self.length:
            char = self.text[self.pos]
            if char.isdigit():
                self.pos += 1
            elif char == '.' and not has_dot:
                has_dot = True
                self.pos += 1
            else:
                break
        
        num_str = self.text[start:self.pos]
        return Token(TokenType.NUMBER, float(num_str))
    
    def tokenize(self) -> List[Token]:
        """Convert the input string into a list of tokens."""
        tokens = []
        
        while self.pos < self.length:
            self.skip_whitespace()
            
            if self.pos >= self.length:
                break
                
            char = self.peek()
            
            if char.isdigit() or (char == '.' and self.pos + 1 < self.length and self.text[self.pos + 1].isdigit()):
                tokens.append(self.read_number())
            elif char == '+':
                tokens.append(Token(TokenType.PLUS, '+'))
                self.advance()
            elif char == '-':
                tokens.append(Token(TokenType.MINUS, '-'))
                self.advance()
            elif char == '*':
                tokens.append(Token(TokenType.MULTIPLY, '*'))
                self.advance()
            elif char == '/':
                tokens.append(Token(TokenType.DIVIDE, '/'))
                self.advance()
            elif char == '(':
                tokens.append(Token(TokenType.LPAREN, '('))
                self.advance()
            elif char == ')':
                tokens.append(Token(TokenType.RPAREN, ')'))
                self.advance()
            else:
                raise ValueError(f"Invalid character: '{char}' at position {self.pos}")
        
        tokens.append(Token(TokenType.EOF, None))
        return tokens


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.pos = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result as a float.
        
        Args:
            expr: A string containing a mathematical expression with numbers,
                  operators (+, -, *, /), parentheses, and optional whitespace.
        
        Returns:
            float: The result of evaluating the expression.
        
        Raises:
            ValueError: If the expression is invalid, empty, has mismatched parentheses,
                       or contains division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        tokenizer = Tokenizer(expr)
        self.tokens = tokenizer.tokenize()
        self.pos = 0
        
        result = self._parse_expression()
        
        if self._current_token().type != TokenType.EOF:
            raise ValueError(f"Unexpected token after expression: {self._current_token().value}")
        
        return result
    
    def _current_token(self) -> Token:
        """Return the current token without advancing."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF
    
    def _advance(self) -> Token:
        """Advance to the next token and return the current one."""
        token = self._current_token()
        self.pos += 1
        return token
    
    def _parse_expression(self) -> float:
        """
        Parse an expression (handles + and - operators).
        Grammar: expression = term (('+' | '-') term)*
        """
        result = self._parse_term()
        
        while self._current_token().type in (TokenType.PLUS, TokenType.MINUS):
            op = self._advance()
            right = self._parse_term()
            
            if op.type == TokenType.PLUS:
                result += right
            else:
                result -= right
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse a term (handles * and / operators).
        Grammar: term = factor (('*' | '/') factor)*
        """
        result = self._parse_factor()
        
        while self._current_token().type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            op = self._advance()
            right = self._parse_factor()
            
            if op.type == TokenType.MULTIPLY:
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parse a factor (handles unary operators, parentheses, and numbers).
        Grammar: factor = unary_factor | number
        """
        token = self._current_token()
        
        if token.type == TokenType.NUMBER:
            self._advance()
            return token.value
        elif token.type == TokenType.LPAREN:
            self._advance()  # consume '('
            result = self._parse_expression()
            if self._current_token().type != TokenType.RPAREN:
                raise ValueError("Mismatched parentheses: expected ')'")
            self._advance()  # consume ')'
            return result
        elif token.type in (TokenType.PLUS, TokenType.MINUS):
            return self._parse_unary()
        else:
            raise ValueError(f"Unexpected token: {token.value}")
    
    def _parse_unary(self) -> float:
        """
        Parse unary operators (+ or -).
        Grammar: unary_factor = ('-' | '+') factor | '(' expression ')'
        """
        op = self._advance()
        operand = self._parse_factor()
        
        if op.type == TokenType.MINUS:
            return -operand
        else:
            return operand


# Test Suite
def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0
    assert evaluator.evaluate("2.5 + 3.5") == 6.0


def test_precedence():
    """Test operator precedence (* and / before + and -)."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + 12 = 14
    assert evaluator.evaluate("2 * 3 + 4") == 10.0  # 6 + 4 = 10
    assert evaluator.evaluate("10 / 2 * 3") == 15.0  # (10/2)*3 = 15
    assert evaluator.evaluate("10 / (2 * 3)") == pytest.approx(1.6666666666666667)
    assert evaluator.evaluate("2 + 3 * 4 - 5") == 9.0  # 2 + 12 - 5 = 9


def test_parentheses():
    """Test parentheses for grouping."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0
    assert evaluator.evaluate("1 + (2 * (3 + 4))") == 15.0
    assert evaluator.evaluate("(1 + 2) * (3 + 4)") == 21.0


def test_unary_minus():
    """Test unary minus operator."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 - -3") == 8.0
    assert evaluator.evaluate("-5 * 2") == -10.0
    assert evaluator.evaluate("-(5 * 2)") == -10.0
    assert evaluator.evaluate("--3") == 3.0  # double negative
    assert evaluator.evaluate("3 * -4") == -12.0


def test_error_cases():
    """Test error handling for invalid inputs."""
    evaluator = ExpressionEvaluator()
    
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (2 - 2)")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + a")
    
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3 +")
    
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("()")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])