from typing import List

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    Supports +, -, *, / with correct precedence, parentheses, unary minus, and floats.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.tokens: List[str] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: The expression string to evaluate
            
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
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at position {self.pos}")
        
        return result
    
    def _tokenize(self, expr: str) -> List[str]:
        """
        Tokenize the expression string into numbers, operators, and parentheses.
        
        Args:
            expr: The expression string
            
        Returns:
            List of tokens
            
        Raises:
            ValueError: For invalid characters or number formats
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
            
            raise ValueError(f"Invalid character '{char}' at position {i}")
        
        return tokens
    
    def _parse_expression(self) -> float:
        """
        Parse an expression (handles + and - operators with lowest precedence).
        
        Returns:
            The evaluated expression value
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
        Parse a term (handles * and / operators with medium precedence).
        
        Returns:
            The evaluated term value
            
        Raises:
            ValueError: For division by zero
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
        Parse a factor (handles unary minus with high precedence).
        
        Returns:
            The evaluated factor value
        """
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1
            return -self._parse_factor()
        
        return self._parse_primary()
    
    def _parse_primary(self) -> float:
        """
        Parse a primary expression (numbers or parenthesized expressions).
        
        Returns:
            The evaluated primary value
            
        Raises:
            ValueError: For mismatched parentheses or invalid tokens
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        
        token = self.tokens[self.pos]
        
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            
            self.pos += 1
            return result
        
        if token == ')':
            raise ValueError("Unexpected closing parenthesis")
        
        # Must be a number
        try:
            result = float(token)
            self.pos += 1
            return result
        except ValueError:
            raise ValueError(f"Invalid token '{token}'")


# Test suite
import pytest

def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("2 * 3") == 6.0
    assert evaluator.evaluate("10 / 2") == 5.0

def test_operator_precedence():
    """Test that operator precedence is correct."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + 12 = 14
    assert evaluator.evaluate("10 / 2 + 3") == 8.0   # 5 + 3 = 8
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0  # 6 + 20 = 26

def test_parentheses():
    """Test parentheses for grouping."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 4) / 2") == 10.0

def test_unary_minus():
    """Test unary minus support."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-2 * 3") == -6.0
    assert evaluator.evaluate("3 - -2") == 5.0

def test_error_cases():
    """Test various error conditions."""
    evaluator = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + a")
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("2 / 0")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3 +")