from typing import Iterator, List, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers
    
    Raises ValueError for invalid expressions, division by zero, etc.
    """
    
    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Parse and evaluate a mathematical expression string.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The evaluated result as a float
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")
        
        self._tokenize(expr)
        
        if not self.tokens:
            raise ValueError("Expression cannot be empty")
            
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at position {self.pos}")
            
        return result
    
    def _tokenize(self, expr: str) -> None:
        """
        Convert expression string into tokens.
        
        Args:
            expr: The expression string to tokenize
        """
        self.tokens = []
        i = 0
        expr = expr.replace(' ', '')  # Remove spaces
        
        while i < len(expr):
            char = expr[i]
            
            if char.isdigit() or char == '.':
                # Parse number (integer or float)
                num_str = ""
                has_dot = False
                
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format at position {i}")
                        has_dot = True
                    num_str += expr[i]
                    i += 1
                
                self.tokens.append(num_str)
                continue
            
            elif char in '+-*/()':
                self.tokens.append(char)
                i += 1
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")
    
    def _parse_expression(self) -> float:
        """
        Parse addition and subtraction (lowest precedence).
        """
        result = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            
            if op == '+':
                result = result + right
            else:
                result = result - right
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse multiplication and division (higher precedence).
        """
        result = self._parse_factor()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()
            
            if op == '*':
                result = result * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result = result / right
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parse numbers, parentheses, and unary minus.
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        
        token = self.tokens[self.pos]
        
        # Handle unary minus
        if token == '-':
            self.pos += 1
            return -self._parse_factor()
        
        # Handle parentheses
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Missing closing parenthesis")
            
            self.pos += 1
            return result
        
        # Handle numbers
        if token.replace('.', '').isdigit():
            self.pos += 1
            try:
                return float(token)
            except ValueError:
                raise ValueError(f"Invalid number: {token}")
        
        raise ValueError(f"Unexpected token: {token}")


# Test suite
import pytest

class TestExpressionEvaluator:
    def setup_method(self):
        self.evaluator = ExpressionEvaluator()
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations"""
        assert self.evaluator.evaluate("2+3") == 5.0
        assert self.evaluator.evaluate("10-4") == 6.0
        assert self.evaluator.evaluate("3*4") == 12.0
        assert self.evaluator.evaluate("15/3") == 5.0
    
    def test_operator_precedence(self):
        """Test that * and / have higher precedence than + and -"""
        assert self.evaluator.evaluate("2+3*4") == 14.0  # 2 + (3*4) = 14
        assert self.evaluator.evaluate("2*3+4*5") == 26.0  # (2*3) + (4*5) = 26
        assert self.evaluator.evaluate("10-2*3") == 4.0  # 10 - (2*3) = 4
        assert self.evaluator.evaluate("10/2+3") == 8.0  # (10/2) + 3 = 8
    
    def test_parentheses(self):
        """Test parentheses for grouping"""
        assert self.evaluator.evaluate("(2+3)*4") == 20.0
        assert self.evaluator.evaluate("2*(3+4)") == 14.0
        assert self.evaluator.evaluate("((2+3)*4)/2") == 10.0
        assert self.evaluator.evaluate("(10-2)/(4-2)") == 4.0
    
    def test_unary_minus(self):
        """Test unary minus operations"""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-(-3)") == 3.0
        assert self.evaluator.evaluate("5+-3") == 2.0
        assert self.evaluator.evaluate("5-(-3)") == 8.0
        assert self.evaluator.evaluate("-(2+3)") == -5.0
        assert self.evaluator.evaluate("-(-2*-3)") == -6.0
    
    def test_error_cases(self):
        """Test various error conditions"""
        # Empty expression
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate("")
        
        # Invalid character
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2+3a")
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Missing closing parenthesis"):
            self.evaluator.evaluate("(2+3")
        
        with pytest.raises(ValueError, match="Unexpected token"):
            self.evaluator.evaluate("2+3)")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("5/0")
        
        # Invalid number format
        with pytest.raises(ValueError, match="Invalid number"):
            self.evaluator.evaluate("2..3")