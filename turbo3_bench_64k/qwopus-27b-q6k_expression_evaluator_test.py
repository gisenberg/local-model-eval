# expression_evaluator.py
import re
from typing import List, Tuple, Union


class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    Supports +, -, *, / with correct precedence, parentheses, and unary minus.
    """
    
    def __init__(self):
        self.tokens: List[Tuple[str, Union[str, float]]] = []
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
            raise ValueError(f"Unexpected token '{self.tokens[self.pos][0]}' at position {self.pos}")
        
        return result
    
    def _tokenize(self, expr: str) -> None:
        """
        Tokenize the input expression into a list of tokens.
        
        Args:
            expr: The input expression string
            
        Raises:
            ValueError: If an invalid token is encountered
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
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format at position {j}: multiple decimal points")
                        has_dot = True
                    j += 1
                
                if i == j:
                    raise ValueError(f"Invalid character '{char}' at position {i}")
                
                num_str = expr[i:j]
                if num_str == '.':
                    raise ValueError(f"Invalid number format at position {i}: '.' is not a valid number")
                
                try:
                    num = float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number format at position {i}: '{num_str}'")
                
                self.tokens.append(('NUMBER', num))
                i = j
                continue
            
            if char in '+-*/()':
                self.tokens.append((char, char))
                i += 1
                continue
            
            raise ValueError(f"Invalid character '{char}' at position {i}")
        
        if not self.tokens:
            raise ValueError("Expression cannot be empty")
    
    def _parse_expression(self) -> float:
        """
        Parse an expression (handles + and - operators).
        
        Returns:
            The evaluated result of the expression
        """
        result = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos][0] in ('+', '-'):
            op = self.tokens[self.pos][0]
            self.pos += 1
            right = self._parse_term()
            
            if op == '+':
                result += right
            else:
                result -= right
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse a term (handles * and / operators).
        
        Returns:
            The evaluated result of the term
        """
        result = self._parse_unary()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos][0] in ('*', '/'):
            op = self.tokens[self.pos][0]
            self.pos += 1
            right = self._parse_unary()
            
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        
        return result
    
    def _parse_unary(self) -> float:
        """
        Parse a unary expression (handles unary minus).
        
        Returns:
            The evaluated result of the unary expression
        """
        if self.pos < len(self.tokens) and self.tokens[self.pos][0] == '-':
            self.pos += 1
            return -self._parse_unary()
        
        return self._parse_primary()
    
    def _parse_primary(self) -> float:
        """
        Parse a primary expression (numbers and parenthesized expressions).
        
        Returns:
            The evaluated result of the primary expression
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        
        token_type, token_value = self.tokens[self.pos]
        
        if token_type == 'NUMBER':
            self.pos += 1
            return token_value
        
        if token_type == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos][0] != ')':
                raise ValueError("Mismatched parentheses: expected ')'")
            
            self.pos += 1
            return result
        
        if token_type == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")
        
        raise ValueError(f"Unexpected token '{token_value}'")


# test_expression_evaluator.py
import pytest



class TestExpressionEvaluator:
    """Test suite for ExpressionEvaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ExpressionEvaluator()
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations with integers and floats."""
        assert self.evaluator.evaluate("1 + 2") == 3.0
        assert self.evaluator.evaluate("10 - 3") == 7.0
        assert self.evaluator.evaluate("4 * 5") == 20.0
        assert self.evaluator.evaluate("15 / 3") == 5.0
        assert self.evaluator.evaluate("3.14 + 2.86") == 6.0
        assert self.evaluator.evaluate("10.5 / 2.1") == 5.0
    
    def test_operator_precedence(self):
        """Test that multiplication/division has higher precedence than addition/subtraction."""
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + 12
        assert self.evaluator.evaluate("2 * 3 + 4") == 10.0  # 6 + 4
        assert self.evaluator.evaluate("10 / 2 + 3") == 8.0  # 5 + 3
        assert self.evaluator.evaluate("10 + 2 / 2") == 11.0  # 10 + 1
        assert self.evaluator.evaluate("2 * 3 * 4") == 24.0
        assert self.evaluator.evaluate("24 / 4 / 2") == 3.0
        assert self.evaluator.evaluate("2 + 3 * 4 - 5 / 5") == 13.0  # 2 + 12 - 1
    
    def test_parentheses(self):
        """Test parentheses for grouping and overriding precedence."""
        assert self.evaluator.evaluate("(1 + 2) * 3") == 9.0
        assert self.evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert self.evaluator.evaluate("(2 + 3) * (4 + 5)") == 45.0
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20.0
        assert self.evaluator.evaluate("1 + (2 * (3 + 4))") == 15.0
        assert self.evaluator.evaluate("(1 + 2) * (3 + 4) / (5 - 2)") == 7.0
    
    def test_unary_minus(self):
        """Test unary minus operator with various combinations."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-3 + 5") == 2.0
        assert self.evaluator.evaluate("5 - -3") == 8.0
        assert self.evaluator.evaluate("-(2 + 1)") == -3.0
        assert self.evaluator.evaluate("-(-3)") == 3.0
        assert self.evaluator.evaluate("-2 * 3") == -6.0
        assert self.evaluator.evaluate("2 * -3") == -6.0
        assert self.evaluator.evaluate("--3") == 3.0
        assert self.evaluator.evaluate("-2.5") == -2.5
    
    def test_error_cases(self):
        """Test error handling for invalid inputs."""
        # Empty expression
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate("")
        
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate("   ")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("1 / 0")
        
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("10 / (2 - 2)")
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched"):
            self.evaluator.evaluate("(1 + 2")
        
        with pytest.raises(ValueError, match="Mismatched"):
            self.evaluator.evaluate("1 + 2)")
        
        with pytest.raises(ValueError, match="Mismatched"):
            self.evaluator.evaluate("((1 + 2)")
        
        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("1 + a")
        
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("1 + + 2")
        
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("1..2")
        
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate(".")
        
        # Unexpected end of expression
        with pytest.raises(ValueError, match="Unexpected"):
            self.evaluator.evaluate("1 +")