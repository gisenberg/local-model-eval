from typing import List, Union, Tuple
from dataclasses import dataclass


@dataclass
class Token:
    """Represents a lexical token in the expression."""
    type: str
    value: Union[int, float, str]
    position: int


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
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result as a float.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The result of the evaluation as a float
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")
        
        self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Expression cannot be empty")
        
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(
                f"Unexpected token '{self.tokens[self.pos].value}' at position "
                f"{self.tokens[self.pos].position}"
            )
        
        return result
    
    def _tokenize(self, expr: str) -> None:
        """
        Convert the expression string into a list of tokens.
        
        Args:
            expr: The expression string to tokenize
            
        Raises:
            ValueError: If the expression contains invalid characters or number formats
        """
        self.tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
            
            if char.isdigit() or char == '.':
                # Parse number (integer or float)
                start = i
                has_dot = False
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(
                                f"Invalid number format at position {i}: "
                                f"multiple decimal points"
                            )
                        has_dot = True
                    i += 1
                
                num_str = expr[start:i]
                if num_str == '.':
                    raise ValueError(
                        f"Invalid number format at position {start}: "
                        f"'.' is not a valid number"
                    )
                
                try:
                    value = float(num_str)
                except ValueError:
                    raise ValueError(
                        f"Invalid number format at position {start}: '{num_str}'"
                    )
                
                self.tokens.append(Token('NUMBER', value, start))
            
            elif char in '+-*/()':
                self.tokens.append(Token('OPERATOR', char, i))
                i += 1
            
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")
    
    def _parse_expression(self) -> float:
        """
        Parse addition and subtraction (lowest precedence).
        
        Grammar: expr -> term (('+' | '-') term)*
        
        Returns:
            The result of the parsed expression
        """
        left = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos].value in ('+', '-'):
            op = self.tokens[self.pos].value
            self.pos += 1
            right = self._parse_term()
            
            if op == '+':
                left = left + right
            else:
                left = left - right
        
        return left
    
    def _parse_term(self) -> float:
        """
        Parse multiplication and division (higher precedence).
        
        Grammar: term -> factor (('*' | '/') factor)*
        
        Returns:
            The result of the parsed term
            
        Raises:
            ValueError: If division by zero is attempted
        """
        left = self._parse_factor()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos].value in ('*', '/'):
            op = self.tokens[self.pos].value
            self.pos += 1
            right = self._parse_factor()
            
            if op == '*':
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
        
        return left
    
    def _parse_factor(self) -> float:
        """
        Parse unary operators, parentheses, and numbers (highest precedence).
        
        Grammar: factor -> ('-' | '+') factor | '(' expr ')' | number
        
        Returns:
            The result of the parsed factor
            
        Raises:
            ValueError: If the expression is malformed
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        
        token = self.tokens[self.pos]
        
        # Handle unary minus and plus
        if token.value in ('-', '+'):
            self.pos += 1
            value = self._parse_factor()
            if token.value == '-':
                return -value
            else:
                return value
        
        # Handle parentheses
        if token.value == '(':
            self.pos += 1
            value = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos].value != ')':
                raise ValueError(
                    f"Expected closing parenthesis at position {self.pos}, "
                    f"got '{self.tokens[self.pos].value if self.pos < len(self.tokens) else 'end of expression'}'"
                )
            
            self.pos += 1
            return value
        
        # Handle numbers
        if token.type == 'NUMBER':
            self.pos += 1
            return token.value
        
        # Invalid token
        raise ValueError(
            f"Unexpected token '{token.value}' at position {token.position}"
        )


# Test suite
import pytest


class TestExpressionEvaluator:
    """Test suite for ExpressionEvaluator."""
    
    def setup_method(self):
        self.evaluator = ExpressionEvaluator()
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations with correct precedence."""
        assert self.evaluator.evaluate("2 + 3") == 5.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("3 * 4") == 12.0
        assert self.evaluator.evaluate("15 / 3") == 5.0
        # Precedence: multiplication before addition
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
    
    def test_precedence(self):
        """Test operator precedence rules."""
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
        assert self.evaluator.evaluate("2 * 3 + 4") == 10.0
        assert self.evaluator.evaluate("10 / 2 + 3") == 8.0
        assert self.evaluator.evaluate("10 + 3 / 2") == 11.5
        assert self.evaluator.evaluate("2 * 3 * 4") == 24.0
        assert self.evaluator.evaluate("24 / 4 / 2") == 3.0
    
    def test_parentheses(self):
        """Test parentheses for grouping and overriding precedence."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert self.evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20.0
        assert self.evaluator.evaluate("(2 + 3) * (4 + 5)") == 45.0
        assert self.evaluator.evaluate("10 / (2 + 3)") == 2.0
    
    def test_unary_minus(self):
        """Test unary minus operator in various contexts."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-3 + 5") == 2.0
        assert self.evaluator.evaluate("5 + -3") == 2.0
        assert self.evaluator.evaluate("-(-3)") == 3.0
        assert self.evaluator.evaluate("-(2 + 3)") == -5.0
        assert self.evaluator.evaluate("-(2 * 3)") == -6.0
        assert self.evaluator.evaluate("-2 * 3") == -6.0
        assert self.evaluator.evaluate("2 * -3") == -6.0
    
    def test_error_cases(self):
        """Test error handling for invalid inputs."""
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
            self.evaluator.evaluate("10 / 0")
        
        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("2 + a")
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("2.3.4")