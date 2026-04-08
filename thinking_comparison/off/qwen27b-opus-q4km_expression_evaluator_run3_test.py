import re
from typing import List, Tuple, Optional, Iterator

class Token:
    """Represents a single token in the expression."""
    
    def __init__(self, type_: str, value: Optional[float] = None, char: Optional[str] = None):
        self.type = type_
        self.value = value
        self.char = char
    
    def __repr__(self) -> str:
        if self.value is not None:
            return f"Token({self.type}, {self.value})"
        return f"Token({self.type}, '{self.char}')"


class Tokenizer:
    """Converts an expression string into a list of tokens."""
    
    def __init__(self, expression: str):
        self.expression = expression
        self.pos = 0
    
    def tokenize(self) -> List[Token]:
        """Tokenize the entire expression."""
        tokens: List[Token] = []
        
        while self.pos < len(self.expression):
            char = self.expression[self.pos]
            
            # Skip whitespace
            if char.isspace():
                self.pos += 1
                continue
            
            # Number (integer or float)
            if char.isdigit() or char == '.':
                token = self._read_number()
                tokens.append(token)
                continue
            
            # Operators and parentheses
            if char in '+-*/()':
                tokens.append(Token(char, char=char))
                self.pos += 1
                continue
            
            # Unknown character
            raise ValueError(f"Invalid character '{char}' at position {self.pos}")
        
        return tokens
    
    def _read_number(self) -> Token:
        """Read a number (integer or float) starting at current position."""
        start = self.pos
        has_dot = False
        
        while self.pos < len(self.expression):
            char = self.expression[self.pos]
            
            if char.isdigit():
                self.pos += 1
            elif char == '.' and not has_dot:
                has_dot = True
                self.pos += 1
            else:
                break
        
        num_str = self.expression[start:self.pos]
        
        if not num_str or num_str == '.':
            raise ValueError(f"Invalid number format at position {start}")
        
        return Token('NUMBER', value=float(num_str))


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus
    - Floating point numbers
    
    Grammar:
        expression -> term (('+' | '-') term)*
        term       -> unary (('*' | '/') unary)*
        unary      -> ('-') unary | primary
        primary    -> NUMBER | '(' expression ')'
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: The expression string to evaluate
            
        Returns:
            The result of the evaluation as a float
            
        Raises:
            ValueError: For invalid expressions, division by zero, etc.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        # Tokenize
        tokenizer = Tokenizer(expr)
        self.tokens = tokenizer.tokenize()
        
        if not self.tokens:
            raise ValueError("Empty expression")
        
        self.pos = 0
        
        # Parse and evaluate
        result = self._parse_expression()
        
        # Check for leftover tokens
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            raise ValueError(f"Unexpected token '{token}' after expression")
        
        return result
    
    def _current_token(self) -> Optional[Token]:
        """Return the current token or None if at end."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None
    
    def _eat(self, expected_type: str = None) -> Token:
        """
        Consume and return the current token.
        
        Args:
            expected_type: Optional expected token type for validation
            
        Returns:
            The consumed token
            
        Raises:
            ValueError: If expected type doesn't match
        """
        token = self._current_token()
        if token is None:
            raise ValueError("Unexpected end of expression")
        
        if expected_type and token.type != expected_type:
            raise ValueError(f"Expected '{expected_type}', got '{token}'")
        
        self.pos += 1
        return token
    
    def _parse_expression(self) -> float:
        """
        Parse an expression (handles + and - operators).
        
        expression -> term (('+' | '-') term)*
        """
        value = self._parse_term()
        
        while True:
            token = self._current_token()
            if token is None:
                break
            
            if token.char == '+':
                self._eat()
                value = value + self._parse_term()
            elif token.char == '-':
                self._eat()
                value = value - self._parse_term()
            else:
                break
        
        return value
    
    def _parse_term(self) -> float:
        """
        Parse a term (handles * and / operators).
        
        term -> unary (('*' | '/') unary)*
        """
        value = self._parse_unary()
        
        while True:
            token = self._current_token()
            if token is None:
                break
            
            if token.char == '*':
                self._eat()
                value = value * self._parse_unary()
            elif token.char == '/':
                self._eat()
                divisor = self._parse_unary()
                if divisor == 0:
                    raise ValueError("Division by zero")
                value = value / divisor
            else:
                break
        
        return value
    
    def _parse_unary(self) -> float:
        """
        Parse a unary expression (handles unary minus).
        
        unary -> ('-') unary | primary
        """
        token = self._current_token()
        if token is not None and token.char == '-':
            self._eat()
            return -self._parse_unary()
        
        return self._parse_primary()
    
    def _parse_primary(self) -> float:
        """
        Parse a primary expression (numbers and parenthesized expressions).
        
        primary -> NUMBER | '(' expression ')'
        """
        token = self._current_token()
        
        if token is None:
            raise ValueError("Unexpected end of expression")
        
        if token.type == 'NUMBER':
            self._eat()
            return token.value
        
        if token.char == '(':
            self._eat()
            value = self._parse_expression()
            
            closing = self._current_token()
            if closing is None or closing.char != ')':
                raise ValueError("Missing closing parenthesis")
            
            self._eat()
            return value
        
        raise ValueError(f"Unexpected token '{token}'")


# Test suite
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

# test_expression_evaluator.py
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
        assert self.evaluator.evaluate("6 * 7") == 42.0
        assert self.evaluator.evaluate("20 / 4") == 5.0
        assert self.evaluator.evaluate("3.14 + 2.86") == 6.0
    
    def test_operator_precedence(self):
        """Test that operator precedence is correctly handled."""
        # Multiplication before addition
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
        assert self.evaluator.evaluate("3 * 4 + 2") == 14.0
        
        # Division before subtraction
        assert self.evaluator.evaluate("10 - 8 / 2") == 6.0
        
        # Mixed precedence
        assert self.evaluator.evaluate("2 + 3 * 4 - 8 / 2") == 10.0
        
        # Left-to-right associativity for same precedence
        assert self.evaluator.evaluate("10 - 5 - 2") == 3.0
        assert self.evaluator.evaluate("20 / 4 / 2") == 2.5
    
    def test_parentheses(self):
        """Test parentheses for grouping."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert self.evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20.0
        assert self.evaluator.evaluate("(1 + 2) * (3 + 4)") == 21.0
        assert self.evaluator.evaluate("(((1 + 2)))") == 3.0
    
    def test_unary_minus(self):
        """Test unary minus operator."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-3 + 5") == 2.0
        assert self.evaluator.evaluate("5 + -3") == 2.0
        assert self.evaluator.evaluate("-(-5)") == 5.0
        assert self.evaluator.evaluate("-(2 + 3)") == -5.0
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
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Missing closing parenthesis"):
            self.evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Unexpected token"):
            self.evaluator.evaluate("2 + 3)")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("1 / 0")
        
        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 + a")
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 @ 3")
        
        # Unexpected token after expression
        with pytest.raises(ValueError, match="Unexpected token"):
            self.evaluator.evaluate("2 + 3 +")