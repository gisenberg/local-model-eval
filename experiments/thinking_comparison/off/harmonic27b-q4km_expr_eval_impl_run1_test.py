from typing import List, Tuple, Iterator

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, / with correct precedence
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers
    
    Raises ValueError for invalid expressions, mismatched parentheses, 
    division by zero, or empty input.
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
            ValueError: If the expression is invalid, empty, or contains errors
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")
        
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Expression cannot be empty")
        
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at position {self.pos}")
        
        return result
    
    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert expression string into tokens.
        
        Args:
            expr: The expression string to tokenize
            
        Returns:
            List of tokens (numbers, operators, parentheses)
            
        Raises:
            ValueError: If invalid characters are encountered
        """
        tokens = []
        i = 0
        expr = expr.replace(' ', '')  # Remove all spaces
        
        while i < len(expr):
            char = expr[i]
            
            if char.isdigit() or char == '.':
                # Parse number (integer or float)
                num_str = ''
                has_decimal = False
                
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_decimal:
                            raise ValueError(f"Invalid number format at position {i}")
                        has_decimal = True
                    num_str += expr[i]
                    i += 1
                
                if not num_str or num_str == '.':
                    raise ValueError(f"Invalid number format at position {i-1}")
                
                tokens.append(num_str)
                continue
                
            elif char in '+-*/()':
                tokens.append(char)
                i += 1
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")
        
        return tokens
    
    def _parse_expression(self) -> float:
        """
        Parse addition and subtraction (lowest precedence).
        
        Returns:
            The evaluated result of the expression
        """
        result = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            operator = self.tokens[self.pos]
            self.pos += 1
            
            right = self._parse_term()
            
            if operator == '+':
                result = result + right
            else:  # operator == '-'
                result = result - right
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse multiplication and division (higher precedence than +/-).
        
        Returns:
            The evaluated result of the term
        """
        result = self._parse_factor()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            operator = self.tokens[self.pos]
            self.pos += 1
            
            right = self._parse_factor()
            
            if operator == '*':
                result = result * right
            else:  # operator == '/'
                if right == 0:
                    raise ValueError("Division by zero")
                result = result / right
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parse unary operators, numbers, and parenthesized expressions.
        
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
        
        # Handle unary plus (though not explicitly required, it's good practice)
        if token == '+':
            self.pos += 1
            return self._parse_factor()
        
        # Handle parenthesized expressions
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Missing closing parenthesis")
            
            self.pos += 1
            return result
        
        # Handle numbers
        try:
            value = float(token)
            self.pos += 1
            return value
        except ValueError:
            raise ValueError(f"Invalid token '{token}'")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("15 / 4") == 3.75

def test_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0

def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_errors(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3")