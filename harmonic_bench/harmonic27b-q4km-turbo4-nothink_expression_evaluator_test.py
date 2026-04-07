from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    Supports +, -, *, / with correct precedence, parentheses, unary minus, and floats.
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
            The numerical result of the expression
            
        Raises:
            ValueError: For invalid expressions, mismatched parentheses, 
                       division by zero, or empty expressions
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
        Parse multiplication and division (higher precedence than +/-).
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
        Parse numbers, parentheses, and unary operators.
        """
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
            
        # Handle parentheses
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Missing closing parenthesis")
                
            self.pos += 1
            return result
            
        # Handle numbers
        if self._is_number(token):
            self.pos += 1
            return float(token)
            
        raise ValueError(f"Unexpected token '{token}'")
    
    def _is_number(self, token: str) -> bool:
        """
        Check if a token represents a valid number.
        """
        try:
            float(token)
            return True
        except ValueError:
            return False


# Test suite
import pytest

def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # Precedence test

def test_operator_precedence():
    """Test that operator precedence is correctly handled."""
    evaluator = ExpressionEvaluator()
    
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("2 * 3 + 4") == 10.0
    
    # Division before subtraction
    assert evaluator.evaluate("10 - 8 / 2") == 6.0
    
    # Complex precedence
    assert evaluator.evaluate("2 + 3 * 4 - 8 / 2") == 10.0

def test_parentheses():
    """Test parentheses for grouping."""
    evaluator = ExpressionEvaluator()
    
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("(2 + 3) * (4 - 1)") == 15.0
    assert evaluator.evaluate("((2 + 3) * 4) / 2") == 10.0

def test_unary_minus():
    """Test unary minus operator."""
    evaluator = ExpressionEvaluator()
    
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("5 + -3") == 2.0
    assert evaluator.evaluate("-(-3)") == 3.0
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
    with pytest.raises(ValueError, match="parenthesis"):
        evaluator.evaluate("(2 + 3")
    
    with pytest.raises(ValueError, match="parenthesis"):
        evaluator.evaluate("2 + 3)")
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    
    # Invalid number format
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("3..14")