"""
Mathematical Expression Evaluator using Recursive Descent Parsing.

Supports: +, -, *, /, parentheses, unary minus, and floating-point numbers.
"""

import re
from typing import Iterator, List, Tuple, Union


class Token:
    """Represents a single token in the expression."""
    
    NUMBER = 'NUMBER'
    PLUS = 'PLUS'
    MINUS = 'MINUS'
    MULTIPLY = 'MULTIPLY'
    DIVIDE = 'DIVIDE'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    EOF = 'EOF'
    
    def __init__(self, type_: str, value: Union[str, float]) -> None:
        self.type = type_
        self.value = value
    
    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


class Tokenizer:
    """Converts an expression string into a list of tokens."""
    
    TOKEN_PATTERNS: List[Tuple[str, str]] = [
        (r'\s+', None),  # Skip whitespace
        (r'\d+\.\d+|\d+\.\d*|\.\d+', Token.NUMBER),  # Floats: 3.14, 3., .5
        (r'\d+', Token.NUMBER),  # Integers
        (r'\+', Token.PLUS),
        (r'-', Token.MINUS),
        (r'\*', Token.MULTIPLY),
        (r'/', Token.DIVIDE),
        (r'\(', Token.LPAREN),
        (r'\)', Token.RPAREN),
    ]
    
    def __init__(self, expression: str) -> None:
        self.expression = expression
        self.pos = 0
    
    def tokenize(self) -> List[Token]:
        """Convert the expression string into a list of tokens."""
        tokens: List[Token] = []
        combined_pattern = '|'.join(f'(?P<{name}>{pat})' 
                                   for pat, name in self.TOKEN_PATTERNS if name)
        regex = re.compile(combined_pattern)
        
        for match in regex.finditer(self.expression):
            token_type = match.lastgroup
            if token_type is None:
                continue  # Skip whitespace
            value = match.group(token_type)
            if token_type == Token.NUMBER:
                value = float(value)
            tokens.append(Token(token_type, value))
        
        tokens.append(Token(Token.EOF, ''))
        return tokens


class ExpressionEvaluator:
    """
    Evaluates mathematical expressions using recursive descent parsing.
    
    Supports:
    - Operators: +, -, *, / with correct precedence
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating-point numbers
    
    Raises ValueError for invalid expressions.
    """
    
    def __init__(self) -> None:
        self.tokens: List[Token] = []
        self.pos = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: The expression to evaluate (e.g., "2 + 3 * (4 - 1)")
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is invalid.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        tokenizer = Tokenizer(expr)
        self.tokens = tokenizer.tokenize()
        self.pos = 0
        
        if len(self.tokens) == 2:  # Only EOF token
            raise ValueError("Empty expression")
        
        result = self._parse_expression()
        
        if self._current_token().type != Token.EOF:
            raise ValueError(f"Unexpected token: {self._current_token()}")
        
        return result
    
    def _current_token(self) -> Token:
        """Return the current token without advancing."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(Token.EOF, '')
    
    def _eat(self, expected_type: str) -> Token:
        """
        Consume the current token if it matches the expected type.
        
        Args:
            expected_type: The expected token type.
            
        Returns:
            The consumed token.
            
        Raises:
            ValueError: If the token doesn't match.
        """
        token = self._current_token()
        if token.type == expected_type:
            self.pos += 1
            return token
        raise ValueError(f"Expected {expected_type}, got {token.type}")
    
    def _parse_expression(self) -> float:
        """
        Parse an expression: Term (('+' | '-') Term)*
        
        This handles addition and subtraction with left-to-right associativity.
        """
        value = self._parse_term()
        
        while True:
            token = self._current_token()
            if token.type == Token.PLUS:
                self._eat(Token.PLUS)
                value = value + self._parse_term()
            elif token.type == Token.MINUS:
                self._eat(Token.MINUS)
                value = value - self._parse_term()
            else:
                break
        
        return value
    
    def _parse_term(self) -> float:
        """
        Parse a term: Factor (('*' | '/') Factor)*
        
        This handles multiplication and division with left-to-right associativity.
        """
        value = self._parse_factor()
        
        while True:
            token = self._current_token()
            if token.type == Token.MULTIPLY:
                self._eat(Token.MULTIPLY)
                value = value * self._parse_factor()
            elif token.type == Token.DIVIDE:
                self._eat(Token.DIVIDE)
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                value = value / divisor
            else:
                break
        
        return value
    
    def _parse_factor(self) -> float:
        """
        Parse a factor: Unary | '(' Expression ')'
        
        A factor is either a unary expression or a parenthesized expression.
        """
        token = self._current_token()
        
        if token.type == Token.LPAREN:
            self._eat(Token.LPAREN)
            value = self._parse_expression()
            if self._current_token().type != Token.RPAREN:
                raise ValueError("Missing closing parenthesis")
            self._eat(Token.RPAREN)
            return value
        
        return self._parse_unary()
    
    def _parse_unary(self) -> float:
        """
        Parse a unary expression: ('-' | '+')? Factor | Number
        
        Handles unary plus and minus operators.
        """
        token = self._current_token()
        
        if token.type == Token.MINUS:
            self._eat(Token.MINUS)
            return -self._parse_unary()
        
        if token.type == Token.PLUS:
            self._eat(Token.PLUS)
            return self._parse_unary()
        
        if token.type == Token.NUMBER:
            self._eat(Token.NUMBER)
            return token.value
        
        raise ValueError(f"Unexpected token: {token}")


# Test suite
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

# tests/test_expression_evaluator.py
import pytest



class TestExpressionEvaluator:
    """Test suite for ExpressionEvaluator."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.evaluator = ExpressionEvaluator()
    
    def test_basic_arithmetic(self) -> None:
        """Test basic arithmetic operations."""
        assert self.evaluator.evaluate("2 + 3") == 5.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("6 * 7") == 42.0
        assert self.evaluator.evaluate("20 / 4") == 5.0
        assert self.evaluator.evaluate("3.14 + 2.86") == 6.0
    
    def test_operator_precedence(self) -> None:
        """Test that multiplication/division have higher precedence than addition/subtraction."""
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + 12 = 14
        assert self.evaluator.evaluate("2 * 3 + 4 * 5") == 26.0  # 6 + 20 = 26
        assert self.evaluator.evaluate("10 - 2 * 3") == 4.0  # 10 - 6 = 4
        assert self.evaluator.evaluate("1 + 2 * 3 - 4 / 2") == 5.0  # 1 + 6 - 2 = 5
    
    def test_parentheses(self) -> None:
        """Test parentheses for grouping."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20.0
        assert self.evaluator.evaluate("2 * (3 + 4) * (5 - 1)") == 48.0
        assert self.evaluator.evaluate("((1 + 2) * (3 + 4)) / 7") == 3.0
    
    def test_unary_minus(self) -> None:
        """Test unary minus operator."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-3 + 5") == 2.0
        assert self.evaluator.evaluate("5 - -3") == 8.0
        assert self.evaluator.evaluate("-(2 + 3)") == -5.0
        assert self.evaluator.evaluate("-(-5)") == 5.0
        assert self.evaluator.evaluate("-2 * -3") == 6.0
        assert self.evaluator.evaluate("+-3") == -3.0
        assert self.evaluator.evaluate("-+3") == -3.0
    
    def test_error_cases(self) -> None:
        """Test error handling for invalid expressions."""
        # Empty expression
        with pytest.raises(ValueError, match="Empty"):
            self.evaluator.evaluate("")
        
        with pytest.raises(ValueError, match="Empty"):
            self.evaluator.evaluate("   ")
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="parenthes"):
            self.evaluator.evaluate("(2 + 3")
        
        with pytest.raises(ValueError, match="Unexpected"):
            self.evaluator.evaluate("2 + 3)")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("10 / 0")
        
        # Invalid token
        with pytest.raises(ValueError, match="Unexpected"):
            self.evaluator.evaluate("2 + a")
        
        with pytest.raises(ValueError, match="Unexpected"):
            self.evaluator.evaluate("2 @ 3")