from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

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
    value: float | str

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        parser = _Parser(tokens)
        result = parser.parse()
        
        # Ensure all tokens were consumed
        if parser.current_token.type != TokenType.EOF:
            raise ValueError("Invalid expression: unexpected tokens at end")
            
        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """
        Converts the input string into a list of tokens.
        
        Args:
            expr: The raw expression string.
            
        Returns:
            A list of Token objects.
            
        Raises:
            ValueError: If an invalid character is encountered.
        """
        tokens: List[Token] = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
            
            # Handle numbers (including floats)
            if char.isdigit() or char == '.':
                start = i
                has_dot = False
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid token: multiple decimal points at position {i}")
                        has_dot = True
                    i += 1
                num_str = expr[start:i]
                try:
                    value = float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number format: {num_str}")
                tokens.append(Token(TokenType.NUMBER, value))
                continue
            
            # Handle operators and parentheses
            if char == '+':
                tokens.append(Token(TokenType.PLUS, '+'))
            elif char == '-':
                tokens.append(Token(TokenType.MINUS, '-'))
            elif char == '*':
                tokens.append(Token(TokenType.MULTIPLY, '*'))
            elif char == '/':
                tokens.append(Token(TokenType.DIVIDE, '/'))
            elif char == '(':
                tokens.append(Token(TokenType.LPAREN, '('))
            elif char == ')':
                tokens.append(Token(TokenType.RPAREN, ')'))
            else:
                raise ValueError(f"Invalid token: '{char}' at position {i}")
            
            i += 1
            
        tokens.append(Token(TokenType.EOF, None))
        return tokens


class _Parser:
    """
    Internal recursive descent parser class.
    Grammar:
        Expression -> Term { ('+' | '-') Term }
        Term       -> Factor { ('*' | '/') Factor }
        Factor     -> Number | '(' Expression ')' | UnaryOp Factor
        UnaryOp    -> '+' | '-'
    """
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    @property
    def current_token(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        token = self.current_token
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def parse(self) -> float:
        """
        Entry point for parsing.
        """
        result = self.parse_expression()
        return result

    def parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        """
        left = self.parse_term()
        
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.advance()
            right = self.parse_term()
            if op.type == TokenType.PLUS:
                left += right
            else:
                left -= right
                
        return left

    def parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        """
        left = self.parse_factor()
        
        while self.current_token.type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            op = self.advance()
            right = self.parse_factor()
            if op.type == TokenType.MULTIPLY:
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
                
        return left

    def parse_factor(self) -> float:
        """
        Parses numbers, parentheses, and unary operators.
        """
        token = self.current_token
        
        # Handle Unary Minus (and Unary Plus for completeness)
        if token.type == TokenType.MINUS:
            self.advance()
            return -self.parse_factor()
        elif token.type == TokenType.PLUS:
            self.advance()
            return self.parse_factor()
            
        # Handle Parentheses
        if token.type == TokenType.LPAREN:
            self.advance()
            value = self.parse_expression()
            if self.current_token.type != TokenType.RPAREN:
                raise ValueError("Mismatched parentheses: expected ')'")
            self.advance()
            return value
            
        # Handle Numbers
        if token.type == TokenType.NUMBER:
            self.advance()
            return token.value
            
        # If none of the above, it's an error
        raise ValueError(f"Invalid token: '{token.value}' at position {self.pos}")

