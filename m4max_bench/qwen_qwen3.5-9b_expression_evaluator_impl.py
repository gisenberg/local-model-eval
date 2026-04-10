from enum import Enum, auto
from typing import List, Tuple, Optional

class TokenType(Enum):
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()

class Token:
    def __init__(self, type: TokenType, value: Optional[str] = None):
        self.type = type
        self.value = value
    
    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"

class ExpressionEvaluator:
    """
    A recursive descent parser for mathematical expressions.
    
    Supports +, -, *, / with correct precedence, parentheses, unary minus, 
    and floating point numbers. Raises ValueError for invalid inputs.
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: A string representing a valid mathematical expression.
            
        Returns:
            The evaluated result as a float.
            
        Raises:
            ValueError: If the expression is empty, has invalid tokens, 
                       mismatched parentheses, or division by zero.
        """
        if not expr:
            raise ValueError("Empty expression")
        
        # Tokenize the input string
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        try:
            result = self._parse_expression()
            # Ensure we consumed all tokens (no trailing garbage)
            if self.pos < len(self.tokens):
                raise ValueError(f"Unexpected token after expression: {self.tokens[self.pos]}")
            return result
        except ZeroDivisionError:
            raise ValueError("Division by zero")

    def _tokenize(self, expr: str) -> List[Token]:
        """
        Converts the input string into a list of tokens.
        
        Args:
            expr: The raw expression string.
            
        Returns:
            A list of Token objects.
            
        Raises:
            ValueError: If invalid characters are found in the expression.
        """
        tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
            
            # Handle Numbers (including floats)
            if char.isdigit() or char == '.':
                j = i
                has_digit = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        # Check for multiple dots in a row (e.g., "3..5")
                        if j > i and expr[j-1] == '.':
                            raise ValueError(f"Invalid token: multiple consecutive dots at index {j}")
                    has_digit = True
                    j += 1
                
                num_str = expr[i:j]
                
                # Validate number format (cannot be just a dot)
                if not has_digit:
                    raise ValueError(f"Invalid token: '{num_str}'")
                
                tokens.append(Token(TokenType.NUMBER, num_str))
                i = j
                continue
            
            # Handle Operators and Parentheses
            if char == '+':
                tokens.append(Token(TokenType.PLUS, "+"))
                i += 1
            elif char == '-':
                tokens.append(Token(TokenType.MINUS, "-"))
                i += 1
            elif char == '*':
                tokens.append(Token(TokenType.MULTIPLY, "*"))
                i += 1
            elif char == '/':
                tokens.append(Token(TokenType.DIVIDE, "/"))
                i += 1
            elif char == '(':
                tokens.append(Token(TokenType.LPAREN, "("))
                i += 1
            elif char == ')':
                tokens.append(Token(TokenType.RPAREN, ")"))
                i += 1
            else:
                raise ValueError(f"Invalid character in expression: '{char}'")
        
        # Add EOF token to signal end of input
        tokens.append(Token(TokenType.EOF, None))
        return tokens
    
    def _current_token(self) -> Token:
        """Returns the current token or EOF if at end."""
        if self.pos >= len(self.tokens):
            return self.tokens[-1] # Return EOF
        return self.tokens[self.pos]
    
    def _consume(self) -> Token:
        """Consumes and returns the current token, advancing position."""
        token = self._current_token()
        self.pos += 1
        return token
    
    def _parse_expression(self) -> float:
        """
        Parses an expression (handles + and -).
        
        Grammar: Expression -> Term { (+|-) Term }
        """
        left = self._parse_term()
        
        while True:
            token = self._current_token()
            if token.type == TokenType.PLUS:
                self._consume()
                right = self._parse_term()
                left += right
            elif token.type == TokenType.MINUS:
                self._consume()
                right = self._parse_term()
                left -= right
            else:
                break
        
        return left
    
    def _parse_term(self) -> float:
        """
        Parses a term (handles * and /).
        
        Grammar: Term -> Factor { (*|/) Factor }
        """
        left = self._parse_factor()
        
        while True:
            token = self._current_token()
            if token.type == TokenType.MULTIPLY:
                self._consume()
                right = self._parse_factor()
                left *= right
            elif token.type == TokenType.DIVIDE:
                self._consume()
                right = self._parse_factor()
                # Check for division by zero here to raise ValueError
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                left /= right
            else:
                break
        
        return left
    
    def _parse_factor(self) -> float:
        """
        Parses a factor (handles numbers, parentheses, and unary minus).
        
        Grammar: Factor -> Number | ( Expression ) | - Factor | + Factor
        """
        token = self._current_token()
        
        if token.type == TokenType.MINUS:
            self._consume()
            value = self._parse_factor()
            return -value
        
        elif token.type == TokenType.PLUS:
            self._consume()
            value = self._parse_factor()
            return +value
        
        elif token.type == TokenType.LPAREN:
            self._consume() # Consume '('
            value = self._parse_expression()
            if self._current_token().type != TokenType.RPAREN:
                raise ValueError("Mismatched parentheses")
            self._consume() # Consume ')'
            return value
        
        elif token.type == TokenType.NUMBER:
            self._consume()
            try:
                return float(token.value)
            except ValueError:
                raise ValueError(f"Invalid number format: {token.value}")
        
        else:
            raise ValueError(f"Unexpected token in factor: {token}")

# --- Pytest Tests ---

