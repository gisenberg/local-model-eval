from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    Supports +, -, *, / with correct precedence, parentheses, unary minus, and floats.
    """
    
    def __init__(self):
        self.pos = 0
        self.tokens: List[str] = []
    
    def evaluate(self, expr: str) -> float:
        """
        Parse and evaluate a mathematical expression string.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The evaluated result as a float
            
        Raises:
            ValueError: For invalid expressions, mismatched parentheses, division by zero, etc.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")
        
        self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Expression cannot be empty")
        
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at position {self.pos}")
            
        return result
    
    def _tokenize(self, expr: str) -> None:
        """Convert expression string into tokens."""
        self.tokens = []
        i = 0
        expr = expr.replace(' ', '')  # Remove spaces
        
        while i < len(expr):
            char = expr[i]
            
            if char.isdigit() or char == '.':
                # Parse number (integer or float)
                num_str = ''
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
                continue
                
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")
        
        # Check for balanced parentheses
        paren_count = 0
        for token in self.tokens:
            if token == '(':
                paren_count += 1
            elif token == ')':
                paren_count -= 1
                if paren_count < 0:
                    raise ValueError("Mismatched parentheses: closing parenthesis without opening")
        
        if paren_count != 0:
            raise ValueError("Mismatched parentheses: unclosed opening parenthesis")
    
    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            
            if op == '+':
                left = left + right
            else:  # op == '-'
                left = left - right
                
        return left
    
    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        left = self._parse_factor()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()
            
            if op == '*':
                left = left * right
            else:  # op == '/'
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
                
        return left
    
    def _parse_factor(self) -> float:
        """Parse numbers, unary minus, and parenthesized expressions."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
            
        token = self.tokens[self.pos]
        
        # Handle unary minus
        if token == '-':
            self.pos += 1
            return -self._parse_factor()
            
        # Handle unary plus (though not explicitly required, it's good practice)
        if token == '+':
            self.pos += 1
            return self._parse_factor()
            
        # Handle parenthesized expression
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Missing closing parenthesis")
                
            self.pos += 1
            return result
            
        # Handle number
        try:
            num = float(token)
            self.pos += 1
            return num
        except ValueError:
            raise ValueError(f"Invalid token '{token}'")

# test_expression_evaluator.py
import pytest


def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0

def test_operator_precedence():
    """Test that multiplication/division have higher precedence than addition/subtraction."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + (3 * 4) = 14
    assert evaluator.evaluate("10 - 2 * 3") == 4.0   # 10 - (2 * 3) = 4
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0  # (2 * 3) + (4 * 5) = 26
    assert evaluator.evaluate("10 / 2 + 3") == 8.0   # (10 / 2) + 3 = 8

def test_parentheses():
    """Test parentheses for grouping."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 - (2 + 3)") == 5.0
    assert evaluator.evaluate("(2 + 3) * (4 - 1)") == 15.0
    assert evaluator.evaluate("((2 + 3) * 4) / 2") == 10.0

def test_unary_minus():
    """Test unary minus operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(-5)") == 5.0
    assert evaluator.evaluate("5 + -3") == 2.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("-2 * 3") == -6.0

def test_error_cases():
    """Test various error conditions."""
    evaluator = ExpressionEvaluator()
    
    # Empty expression
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate("")
    
    # Invalid character
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + a")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    
    # Invalid number format
    with pytest.raises(ValueError, match="Invalid number format"):
        evaluator.evaluate("3.14.15")