from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """
    
    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: The mathematical expression to evaluate
            
        Returns:
            The numerical result of the expression
            
        Raises:
            ValueError: For invalid expressions, division by zero, or mismatched parentheses
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Empty expression")
            
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
            
        return result
    
    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert expression string into tokens.
        
        Args:
            expr: The expression string to tokenize
            
        Returns:
            List of tokens (numbers, operators, parentheses)
        """
        tokens = []
        i = 0
        
        while i < len(expr):
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
                
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
                
                if not num_str or num_str == '.':
                    raise ValueError(f"Invalid number format at position {i-1}")
                    
                tokens.append(num_str)
                continue
                
            if char in '+-*/()':
                tokens.append(char)
                i += 1
                continue
                
            raise ValueError(f"Invalid character: '{char}' at position {i}")
            
        return tokens
    
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
                result += right
            else:
                result -= right
                
        return result
    
    def _parse_term(self) -> float:
        """
        Parse multiplication and division (higher precedence than +,-).
        """
        result = self._parse_factor()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()
            
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
                
        return result
    
    def _parse_factor(self) -> float:
        """
        Parse unary operators, numbers, and parenthesized expressions.
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
            
        # Handle parenthesized expression
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
                
            self.pos += 1
            return result
            
        # Handle number
        if self._is_number(token):
            self.pos += 1
            return float(token)
            
        raise ValueError(f"Unexpected token: {token}")
    
    def _is_number(self, token: str) -> bool:
        """Check if a token represents a valid number."""
        try:
            float(token)
            return True
        except ValueError:
            return False


# Test cases
import pytest

def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # precedence test

def test_operator_precedence():
    """Test correct operator precedence."""
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
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0
    assert evaluator.evaluate("(2 + 3) * (4 + 5)") == 45.0

def test_unary_minus():
    """Test unary minus operations."""
    evaluator = ExpressionEvaluator()
    
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 + -3") == 2.0
    assert evaluator.evaluate("-2 * 3") == -6.0
    assert evaluator.evaluate("-(2 * 3)") == -6.0
    assert evaluator.evaluate("--3") == 3.0  # double negative

def test_error_cases():
    """Test error handling for invalid expressions."""
    evaluator = ExpressionEvaluator()
    
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + a")
    
    with pytest.raises(ValueError, match="Invalid number format"):
        evaluator.evaluate("2 + .")