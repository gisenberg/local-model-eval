from typing import List

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    Supports +, -, *, / with correct precedence, parentheses, unary minus, and floating point numbers.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.pos = 0
        self.tokens = []
    
    def _tokenize(self, expr: str) -> List[str]:
        """
        Tokenize the expression string into numbers, operators, and parentheses.
        
        Args:
            expr: The expression string to tokenize
            
        Returns:
            List of tokens
            
        Raises:
            ValueError: If the expression contains invalid characters
        """
        tokens = []
        i = 0
        
        while i < len(expr):
            char = expr[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
            
            # Check for numbers (including decimals)
            if char.isdigit() or char == '.':
                num_str = ""
                has_decimal = False
                
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_decimal:
                            raise ValueError(f"Invalid number format at position {i}: multiple decimal points")
                        has_decimal = True
                    num_str += expr[i]
                    i += 1
                
                if not num_str or num_str == '.':
                    raise ValueError(f"Invalid number at position {i-1}")
                
                tokens.append(num_str)
                continue
            
            # Check for operators and parentheses
            if char in '+-*/()':
                tokens.append(char)
                i += 1
                continue
            
            # Invalid character
            raise ValueError(f"Invalid character '{char}' at position {i}")
        
        return tokens
    
    def _parse_expression(self) -> float:
        """
        Parse an expression (handles + and - operators with lowest precedence).
        
        Returns:
            The evaluated result of the expression
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
        Parse a term (handles * and / operators with higher precedence).
        
        Returns:
            The evaluated result of the term
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
        Parse a factor (handles unary minus, parentheses, and numbers).
        
        Returns:
            The evaluated result of the factor
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
        try:
            self.pos += 1
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid number '{token}'")
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Args:
            expr: The mathematical expression to evaluate
            
        Returns:
            The result of the evaluation
            
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

import pytest

def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("20 / 4") == 5.0

def test_precedence():
    """Test operator precedence (multiplication before addition)."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses():
    """Test parentheses for grouping."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 + (3 * 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0

def test_unary_minus():
    """Test unary minus operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-2 * 3") == -6.0
    assert evaluator.evaluate("-(-3)") == 3.0

def test_error_cases():
    """Test error handling for invalid expressions."""
    evaluator = ExpressionEvaluator()
    
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + *")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("2 / 0")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")