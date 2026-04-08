from typing import List, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Enum representing different token types in the expression."""
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
    """Represents a single token in the expression."""
    type: TokenType
    value: Union[float, str]
    position: int  # Character position in original string for error reporting


class Tokenizer:
    """Converts an expression string into a list of tokens."""
    
    def __init__(self, expression: str):
        self.expression = expression
        self.pos = 0
        self.length = len(expression)
    
    def _peek_char(self) -> str:
        """Return the current character without advancing."""
        if self.pos >= self.length:
            return '\0'
        return self.expression[self.pos]
    
    def _advance(self) -> str:
        """Return the current character and advance position."""
        char = self._peek_char()
        self.pos += 1
        return char
    
    def _skip_whitespace(self) -> None:
        """Skip any whitespace characters."""
        while self.pos < self.length and self.expression[self.pos].isspace():
            self.pos += 1
    
    def _read_number(self) -> Token:
        """Read a number (integer or floating point) from the expression."""
        start_pos = self.pos
        has_dot = False
        
        while self.pos < self.length:
            char = self.expression[self.pos]
            if char.isdigit():
                self.pos += 1
            elif char == '.' and not has_dot:
                has_dot = True
                self.pos += 1
            else:
                break
        
        num_str = self.expression[start_pos:self.pos]
        return Token(TokenType.NUMBER, float(num_str), start_pos)
    
    def tokenize(self) -> List[Token]:
        """Convert the entire expression into a list of tokens."""
        tokens: List[Token] = []
        
        while self.pos < self.length:
            self._skip_whitespace()
            
            if self.pos >= self.length:
                break
            
            char = self._peek_char()
            start_pos = self.pos
            
            if char.isdigit() or char == '.':
                tokens.append(self._read_number())
            elif char == '+':
                self._advance()
                tokens.append(Token(TokenType.PLUS, '+', start_pos))
            elif char == '-':
                self._advance()
                tokens.append(Token(TokenType.MINUS, '-', start_pos))
            elif char == '*':
                self._advance()
                tokens.append(Token(TokenType.MULTIPLY, '*', start_pos))
            elif char == '/':
                self._advance()
                tokens.append(Token(TokenType.DIVIDE, '/', start_pos))
            elif char == '(':
                self._advance()
                tokens.append(Token(TokenType.LPAREN, '(', start_pos))
            elif char == ')':
                self._advance()
                tokens.append(Token(TokenType.RPAREN, ')', start_pos))
            else:
                raise ValueError(
                    f"Invalid character '{char}' at position {start_pos}"
                )
        
        tokens.append(Token(TokenType.EOF, '', self.length))
        return tokens


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus operator
    - Floating point numbers
    
    Grammar:
        expression -> term (('+' | '-') term)*
        term       -> factor (('*' | '/') factor)*
        factor     -> number | '(' expression ')' | '-' factor
    """
    
    def __init__(self, expression: str):
        """Initialize the evaluator with an expression string."""
        if not expression or expression.strip() == '':
            raise ValueError("Empty expression")
        
        self.tokenizer = Tokenizer(expression)
        self.tokens = self.tokenizer.tokenize()
        self.current_pos = 0
    
    def _current_token(self) -> Token:
        """Return the current token being processed."""
        if self.current_pos < len(self.tokens):
            return self.tokens[self.current_pos]
        return self.tokens[-1]  # EOF token
    
    def _advance(self) -> Token:
        """Return current token and advance to the next one."""
        token = self._current_token()
        if self.current_pos < len(self.tokens) - 1:
            self.current_pos += 1
        return token
    
    def _match(self, token_type: TokenType) -> bool:
        """Check if current token matches the expected type."""
        return self._current_token().type == token_type
    
    def _expect(self, token_type: TokenType, context: str = "") -> Token:
        """
        Expect a specific token type, raise error if not found.
        
        Args:
            token_type: The expected token type
            context: Additional context for the error message
            
        Returns:
            The matched token
            
        Raises:
            ValueError: If the expected token is not found
        """
        token = self._current_token()
        if token.type != token_type:
            expected_name = token_type.name
            actual_name = token.type.name
            if context:
                raise ValueError(
                    f"Expected {expected_name} {context}, got {actual_name} "
                    f"at position {token.position}"
                )
            raise ValueError(
                f"Expected {expected_name}, got {actual_name} "
                f"at position {token.position}"
            )
        return self._advance()
    
    def _parse_expression(self) -> float:
        """
        Parse an expression (handles + and - operators).
        
        Grammar: expression -> term (('+' | '-') term)*
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
        Parse a term (handles * and / operators).
        
        Grammar: term -> factor (('*' | '/') factor)*
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
        Parse a factor (numbers, parentheses, unary minus).
        
        Grammar: factor -> number | '(' expression ')' | '-' factor
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
            self._expect(TokenType.RPAREN, "after expression")
            return result
        
        # Invalid token encountered
        raise ValueError(
            f"Unexpected token {token.type.name} at position {token.position}"
        )
    
    def evaluate(self) -> float:
        """
        Evaluate the expression and return the result.
        
        Returns:
            The numerical result of the expression
            
        Raises:
            ValueError: If the expression is invalid or cannot be evaluated
        """
        result = self._parse_expression()
        
        # Check for extra tokens (mismatched parentheses, extra operators)
        if self._current_token().type != TokenType.EOF:
            token = self._current_token()
            raise ValueError(
                f"Unexpected token {token.type.name} at position {token.position}"
            )
        
        return result


# Convenience function for standalone use
def evaluate_expression(expr: str) -> float:
    """
    Evaluate a mathematical expression string.
    
    Args:
        expr: A string containing a mathematical expression
        
    Returns:
        The numerical result of the expression
        
    Raises:
        ValueError: If the expression is invalid
    """
    evaluator = ExpressionEvaluator(expr)
    return evaluator.evaluate()

import pytest


class TestExpressionEvaluator:
    """Test suite for the ExpressionEvaluator class."""
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        evaluator = ExpressionEvaluator("2 + 3 * 4")
        assert evaluator.evaluate() == 14.0
        
        evaluator = ExpressionEvaluator("10 - 2")
        assert evaluator.evaluate() == 8.0
        
        evaluator = ExpressionEvaluator("6 / 2")
        assert evaluator.evaluate() == 3.0
        
        evaluator = ExpressionEvaluator("2 * 3 * 4")
        assert evaluator.evaluate() == 24.0
    
    def test_operator_precedence(self):
        """Test that operator precedence is correctly handled."""
        # Multiplication before addition
        evaluator = ExpressionEvaluator("2 + 3 * 4")
        assert evaluator.evaluate() == 14.0  # 2 + 12 = 14, not 20
        
        # Multiplication before subtraction
        evaluator = ExpressionEvaluator("10 - 2 * 3")
        assert evaluator.evaluate() == 4.0  # 10 - 6 = 4, not 24
        
        # Division before addition
        evaluator = ExpressionEvaluator("10 + 20 / 4")
        assert evaluator.evaluate() == 15.0  # 10 + 5 = 15
        
        # Complex precedence
        evaluator = ExpressionEvaluator("2 + 3 * 4 - 8 / 2")
        assert evaluator.evaluate() == 10.0  # 2 + 12 - 4 = 10
    
    def test_parentheses(self):
        """Test parentheses for grouping."""
        # Basic parentheses
        evaluator = ExpressionEvaluator("(2 + 3) * 4")
        assert evaluator.evaluate() == 20.0
        
        # Nested parentheses
        evaluator = ExpressionEvaluator("((2 + 3) * 4)")
        assert evaluator.evaluate() == 20.0
        
        # Multiple parentheses
        evaluator = ExpressionEvaluator("(2 + 3) * (4 + 5)")
        assert evaluator.evaluate() == 45.0
        
        # Deep nesting
        evaluator = ExpressionEvaluator("((((1 + 2))))")
        assert evaluator.evaluate() == 3.0
    
    def test_unary_minus(self):
        """Test unary minus operator."""
        # Simple unary minus
        evaluator = ExpressionEvaluator("-3")
        assert evaluator.evaluate() == -3.0
        
        # Unary minus with parentheses
        evaluator = ExpressionEvaluator("-(2 + 3)")
        assert evaluator.evaluate() == -5.0
        
        # Unary minus in expression
        evaluator = ExpressionEvaluator("5 + -3")
        assert evaluator.evaluate() == 2.0
        
        # Multiple unary minus
        evaluator = ExpressionEvaluator("--3")
        assert evaluator.evaluate() == 3.0
        
        # Unary minus with multiplication
        evaluator = ExpressionEvaluator("-2 * 3")
        assert evaluator.evaluate() == -6.0
        
        # Complex unary minus
        evaluator = ExpressionEvaluator("-(-2 + 3) * 4")
        assert evaluator.evaluate() == -4.0
    
    def test_error_cases(self):
        """Test various error conditions."""
        # Empty expression
        with pytest.raises(ValueError, match="Empty expression"):
            ExpressionEvaluator("").evaluate()
        
        # Whitespace only
        with pytest.raises(ValueError, match="Empty expression"):
            ExpressionEvaluator("   ").evaluate()
        
        # Mismatched parentheses - missing closing
        with pytest.raises(ValueError, match="RPAREN"):
            ExpressionEvaluator("(2 + 3").evaluate()
        
        # Mismatched parentheses - extra closing
        with pytest.raises(ValueError, match="RPAREN"):
            ExpressionEvaluator("2 + 3)").evaluate()
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            ExpressionEvaluator("10 / 0").evaluate()
        
        # Invalid character
        with pytest.raises(ValueError, match="Invalid character"):
            ExpressionEvaluator("2 + a").evaluate()
        
        # Extra operator
        with pytest.raises(ValueError, match="Unexpected token"):
            ExpressionEvaluator("2 + 3 +").evaluate()
        
        # Invalid number format
        with pytest.raises(ValueError, match="Invalid character"):
            ExpressionEvaluator("2..5").evaluate()


# Additional test for floating point numbers
def test_floating_point_numbers():
    """Test floating point number parsing."""
    evaluator = ExpressionEvaluator("3.14 + 2.5")
    assert evaluator.evaluate() == 5.64
    
    evaluator = ExpressionEvaluator("1.5 * 2.0")
    assert evaluator.evaluate() == 3.0
    
    evaluator = ExpressionEvaluator("10.0 / 4.0")
    assert evaluator.evaluate() == 2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])