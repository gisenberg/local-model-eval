from typing import List

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    Supports +, -, *, / operators with correct precedence, parentheses,
    unary minus, and floating point numbers.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self._tokens: List[str] = []
        self._pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The numerical result of the expression
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")
        
        self._tokens = self._tokenize(expr)
        self._pos = 0
        
        if not self._tokens:
            raise ValueError("Expression cannot be empty")
        
        result = self._parse_expression()
        
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token '{self._tokens[self._pos]}' at position {self._pos}")
        
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
        Parse an expression (handles + and - operators).
        
        Returns:
            The evaluated result of the expression
        """
        result = self._parse_term()
        
        while self._pos < len(self._tokens) and self._tokens[self._pos] in '+-':
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_term()
            
            if op == '+':
                result = result + right
            else:  # op == '-'
                result = result - right
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse a term (handles * and / operators).
        
        Returns:
            The evaluated result of the term
        """
        result = self._parse_factor()
        
        while self._pos < len(self._tokens) and self._tokens[self._pos] in '*/':
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_factor()
            
            if op == '*':
                result = result * right
            else:  # op == '/'
                if right == 0:
                    raise ValueError("Division by zero")
                result = result / right
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parse a factor (handles numbers, parentheses, and unary minus).
        
        Returns:
            The evaluated result of the factor
        """
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression")
        
        token = self._tokens[self._pos]
        
        # Handle unary minus
        if token == '-':
            self._pos += 1
            return -self._parse_factor()
        
        # Handle parentheses
        if token == '(':
            self._pos += 1
            result = self._parse_expression()
            
            if self._pos >= len(self._tokens) or self._tokens[self._pos] != ')':
                raise ValueError("Missing closing parenthesis")
            
            self._pos += 1
            return result
        
        # Handle numbers
        if self._is_number(token):
            self._pos += 1
            return float(token)
        
        raise ValueError(f"Unexpected token '{token}'")
    
    def _is_number(self, token: str) -> bool:
        """
        Check if a token represents a number.
        
        Args:
            token: The token to check
            
        Returns:
            True if the token is a valid number
        """
        try:
            float(token)
            return True
        except ValueError:
            return False

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