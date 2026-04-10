"""
Mathematical Expression Evaluator using Recursive Descent Parsing.

Supports:
- Basic arithmetic: +, -, *, /
- Operator precedence (* and / before + and -)
- Parentheses for grouping
- Unary minus (e.g., -3, -(2+1))
- Floating point numbers
"""

from typing import List, Tuple, Any
from enum import Enum, auto


class TokenType(Enum):
    """Token types for the lexer."""
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


class Token:
    """Represents a single token from the input expression."""
    
    def __init__(self, token_type: TokenType, value: Any):
        self.type = token_type
        self.value = value
    
    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


class Lexer:
    """Tokenizes the input expression string."""
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if text else None
    
    def advance(self) -> None:
        """Move to the next character."""
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None
    
    def peek(self) -> str | None:
        """Look at the next character without advancing."""
        return self.text[self.pos + 1] if self.pos + 1 < len(self.text) else None
    
    def skip_whitespace(self) -> None:
        """Skip over whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
    
    def read_number(self) -> float:
        """Read a number (integer or float) from the input."""
        result = ""
        
        # Read integer part
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        
        # Read decimal part if present
        if self.current_char == '.':
            result += '.'
            self.advance()
            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()
        
        return float(result)
    
    def get_next_token(self) -> Token:
        """Get the next token from the input."""
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            if self.current_char.isdigit() or (self.current_char == '.' and 
                self.peek() is not None and self.peek().isdigit()):
                return Token(TokenType.NUMBER, self.read_number())
            
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
            
            # Unknown character
            raise ValueError(f"Invalid character: '{self.current_char}' at position {self.pos}")
        
        return Token(TokenType.EOF, None)
    
    def tokenize(self) -> List[Token]:
        """Tokenize the entire input string."""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Grammar:
        expression → term (('+' | '-') term)*
        term       → factor (('*' | '/') factor)*
        factor     → unary
        unary      → ('-' | '+') unary | primary
        primary    → number | '(' expression ')'
    """
    
    def __init__(self, expr: str):
        """Initialize the evaluator with an expression string."""
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        self.lexer = Lexer(expr)
        self.tokens = self.lexer.tokenize()
        self.pos = 0
        self.current_token = self.tokens[0] if self.tokens else Token(TokenType.EOF, None)
    
    def eat(self, token_type: TokenType) -> Token:
        """
        Consume the current token if it matches the expected type.
        
        Args:
            token_type: The expected token type.
            
        Returns:
            The consumed token.
            
        Raises:
            ValueError: If the token type doesn't match.
        """
        if self.current_token.type == token_type:
            token = self.current_token
            if self.pos + 1 < len(self.tokens):
                self.pos += 1
                self.current_token = self.tokens[self.pos]
            else:
                self.current_token = Token(TokenType.EOF, None)
            return token
        else:
            raise ValueError(
                f"Expected {token_type.name}, got {self.current_token.type.name}"
            )
    
    def expression(self) -> float:
        """
        Parse an expression: term (('+' | '-') term)*
        
        This is the entry point for parsing, handling addition and subtraction.
        """
        result = self.term()
        
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            if self.current_token.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)
                result = result + self.term()
            else:
                self.eat(TokenType.MINUS)
                result = result - self.term()
        
        return result
    
    def term(self) -> float:
        """
        Parse a term: factor (('*' | '/') factor)*
        
        Handles multiplication and division with higher precedence than
        addition and subtraction.
        """
        result = self.factor()
        
        while self.current_token.type in (TokenType.MUL, TokenType.DIV):
            if self.current_token.type == TokenType.MUL:
                self.eat(TokenType.MUL)
                result = result * self.factor()
            else:
                self.eat(TokenType.DIV)
                divisor = self.factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                result = result / divisor
        
        return result
    
    def factor(self) -> float:
        """
        Parse a factor: unary
        
        Delegates to unary for handling unary operators.
        """
        return self.unary()
    
    def unary(self) -> float:
        """
        Parse a unary expression: ('-' | '+') unary | primary
        
        Handles unary minus and plus operators.
        """
        if self.current_token.type == TokenType.MINUS:
            self.eat(TokenType.MINUS)
            return -self.unary()
        
        if self.current_token.type == TokenType.PLUS:
            self.eat(TokenType.PLUS)
            return self.unary()
        
        return self.primary()
    
    def primary(self) -> float:
        """
        Parse a primary: number | '(' expression ')'
        
        Handles numbers and parenthesized expressions.
        """
        if self.current_token.type == TokenType.NUMBER:
            value = self.current_token.value
            self.eat(TokenType.NUMBER)
            return value
        
        if self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            result = self.expression()
            
            if self.current_token.type != TokenType.RPAREN:
                raise ValueError("Missing closing parenthesis")
            
            self.eat(TokenType.RPAREN)
            return result
        
        # Unexpected token
        raise ValueError(
            f"Unexpected token: {self.current_token.type.name}"
        )
    
    def evaluate(self) -> float:
        """
        Evaluate the expression and return the result.
        
        Returns:
            The result of the expression as a float.
            
        Raises:
            ValueError: For various error conditions including:
                - Empty expression
                - Invalid characters
                - Mismatched parentheses
                - Division by zero
                - Invalid tokens
        """
        result = self.expression()
        
        # Check for extra tokens (should be only EOF)
        if self.current_token.type != TokenType.EOF:
            raise ValueError(
                f"Unexpected token after expression: {self.current_token.type.name}"
            )
        
        return result


# Convenience function for one-off evaluations
def evaluate_expression(expr: str) -> float:
    """
    Evaluate a mathematical expression string.
    
    Args:
        expr: The expression to evaluate.
        
    Returns:
        The result as a float.
        
    Raises:
        ValueError: For invalid expressions.
    """
    evaluator = ExpressionEvaluator(expr)
    return evaluator.evaluate()


# For pytest compatibility - the evaluate method should accept expr as parameter
class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Grammar:
        expression → term (('+' | '-') term)*
        term       → factor (('*' | '/') factor)*
        factor     → unary
        unary      → ('-' | '+') unary | primary
        primary    → number | '(' expression ')'
    """
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: The expression to evaluate.
            
        Returns:
            The result of the expression as a float.
            
        Raises:
            ValueError: For various error conditions including:
                - Empty expression
                - Invalid characters
                - Mismatched parentheses
                - Division by zero
                - Invalid tokens
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        # Create lexer and tokenize
        lexer = Lexer(expr)
        tokens = lexer.tokenize()
        
        # Create parser state
        pos = 0
        current_token = tokens[0] if tokens else Token(TokenType.EOF, None)
        
        def eat(token_type: TokenType) -> Token:
            if current_token.type == token_type:
                token = current_token
                nonlocal pos, current_token
                if pos + 1 < len(tokens):
                    pos += 1
                    current_token = tokens[pos]
                else:
                    current_token = Token(TokenType.EOF, None)
                return token
            else:
                raise ValueError(
                    f"Expected {token_type.name}, got {current_token.type.name}"
                )
        
        def expression() -> float:
            result = term()
            while current_token.type in (TokenType.PLUS, TokenType.MINUS):
                if current_token.type == TokenType.PLUS:
                    eat(TokenType.PLUS)
                    result = result + term()
                else:
                    eat(TokenType.MINUS)
                    result = result - term()
            return result
        
        def term() -> float:
            result = factor()
            while current_token.type in (TokenType.MUL, TokenType.DIV):
                if current_token.type == TokenType.MUL:
                    eat(TokenType.MUL)
                    result = result * factor()
                else:
                    eat(TokenType.DIV)
                    divisor = factor()
                    if divisor == 0:
                        raise ValueError("Division by zero")
                    result = result / divisor
            return result
        
        def factor() -> float:
            return unary()
        
        def unary() -> float:
            if current_token.type == TokenType.MINUS:
                eat(TokenType.MINUS)
                return -unary()
            if current_token.type == TokenType.PLUS:
                eat(TokenType.PLUS)
                return unary()
            return primary()
        
        def primary() -> float:
            if current_token.type == TokenType.NUMBER:
                value = current_token.value
                eat(TokenType.NUMBER)
                return value
            if current_token.type == TokenType.LPAREN:
                eat(TokenType.LPAREN)
                result = expression()
                if current_token.type != TokenType.RPAREN:
                    raise ValueError("Missing closing parenthesis")
                eat(TokenType.RPAREN)
                return result
            raise ValueError(f"Unexpected token: {current_token.type.name}")
        
        result = expression()
        
        if current_token.type != TokenType.EOF:
            raise ValueError(
                f"Unexpected token after expression: {current_token.type.name}"
            )
        
        return result

"""
Pytest tests for the ExpressionEvaluator.
"""

import pytest



class TestExpressionEvaluator:
    """Test suite for ExpressionEvaluator."""
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        evaluator = ExpressionEvaluator()
        
        # Addition
        assert evaluator.evaluate("1 + 2") == 3.0
        assert evaluator.evaluate("10 + 20 + 30") == 60.0
        
        # Subtraction
        assert evaluator.evaluate("10 - 3") == 7.0
        assert evaluator.evaluate("10 - 3 - 2") == 5.0
        
        # Multiplication
        assert evaluator.evaluate("3 * 4") == 12.0
        assert evaluator.evaluate("2 * 3 * 4") == 24.0
        
        # Division
        assert evaluator.evaluate("20 / 4") == 5.0
        assert evaluator.evaluate("100 / 10 / 2") == 5.0
        
        # Mixed operations
        assert evaluator.evaluate("1 + 2 * 3") == 7.0  # 1 + (2*3) = 7
        assert evaluator.evaluate("10 / 2 + 3") == 8.0  # (10/2) + 3 = 8
        
        # Floating point numbers
        assert evaluator.evaluate("3.14 + 2.86") == 6.0
        assert evaluator.evaluate("0.1 + 0.2") == pytest.approx(0.3)
    
    def test_operator_precedence(self):
        """Test that operator precedence is correctly handled."""
        evaluator = ExpressionEvaluator()
        
        # Multiplication before addition
        assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + 12 = 14
        
        # Division before addition
        assert evaluator.evaluate("10 / 2 + 3") == 8.0  # 5 + 3 = 8
        
        # Multiplication before subtraction
        assert evaluator.evaluate("10 - 2 * 3") == 4.0  # 10 - 6 = 4
        
        # Multiplication and division have same precedence (left to right)
        assert evaluator.evaluate("12 / 3 * 2") == 8.0  # (12/3)*2 = 8
        assert evaluator.evaluate("12 * 2 / 3") == 8.0  # (12*2)/3 = 8
        
        # Complex expression
        assert evaluator.evaluate("1 + 2 * 3 - 4 / 2") == 5.0  # 1 + 6 - 2 = 5
    
    def test_parentheses(self):
        """Test parentheses for grouping and overriding precedence."""
        evaluator = ExpressionEvaluator()
        
        # Basic parentheses
        assert evaluator.evaluate("(1 + 2) * 3") == 9.0  # 3 * 3 = 9
        assert evaluator.evaluate("10 / (2 + 3)") == 2.0  # 10 / 5 = 2
        
        # Nested parentheses
        assert evaluator.evaluate("((2 + 3) * 4)") == 20.0
        assert evaluator.evaluate("(((1 + 2)))") == 3.0
        
        # Complex nested expression
        assert evaluator.evaluate("(1 + (2 * (3 + 4)))") == 15.0  # 1 + (2 * 7) = 15
        
        # Parentheses with unary minus
        assert evaluator.evaluate("-(1 + 2)") == -3.0
        assert evaluator.evaluate("-(-5)") == 5.0
    
    def test_unary_minus(self):
        """Test unary minus operator."""
        evaluator = ExpressionEvaluator()
        
        # Simple unary minus
        assert evaluator.evaluate("-5") == -5.0
        assert evaluator.evaluate("-3.14") == -3.14
        
        # Unary minus with operations
        assert evaluator.evaluate("-3 + 2") == -1.0
        assert evaluator.evaluate("5 + -3") == 2.0
        assert evaluator.evaluate("-3 * 4") == -12.0
        assert evaluator.evaluate("-10 / 2") == -5.0
        
        # Double negation
        assert evaluator.evaluate("--5") == 5.0
        assert evaluator.evaluate("---5") == -5.0
        
        # Unary minus with parentheses
        assert evaluator.evaluate("-(2 + 3)") == -5.0
        assert evaluator.evaluate("-(2 * 3)") == -6.0
        
        # Unary minus precedence
        assert evaluator.evaluate("-3 * 2") == -6.0  # (-3) * 2
        assert evaluator.evaluate("2 * -3") == -6.0  # 2 * (-3)
    
    def test_error_cases(self):
        """Test various error conditions."""
        evaluator = ExpressionEvaluator()
        
        # Empty expression
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")
        
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("   ")
        
        # Invalid characters
        with pytest.raises(ValueError, match="Invalid character"):
            evaluator.evaluate("1 + a")
        
        with pytest.raises(ValueError, match="Invalid character"):
            evaluator.evaluate("1 @ 2")
        
        # Mismatched parentheses - missing closing
        with pytest.raises(ValueError, match="Missing closing parenthesis"):
            evaluator.evaluate("(1 + 2")
        
        with pytest.raises(ValueError, match="Missing closing parenthesis"):
            evaluator.evaluate("((1 + 2)")
        
        # Mismatched parentheses - extra closing
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluator.evaluate("1 + 2)")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("1 / 0")
        
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("10 / (2 - 2)")
        
        # Invalid token in expression
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluator.evaluate("1 + + 2")
        
        # Trailing operator
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluator.evaluate("1 +")