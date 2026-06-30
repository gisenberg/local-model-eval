import re
from typing import List, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: The mathematical expression string.
            
        Returns:
            The result of the evaluation.
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Empty expression")
            
        parser = Parser(tokens)
        result = parser.parse_expression()
        
        # Check if there are leftover tokens (syntax error)
        if parser.pos < len(parser.tokens):
            raise ValueError(f"Unexpected token: {parser.tokens[parser.pos]}")
            
        return result

    def _tokenize(self, expr: str) -> List[Union[float, str]]:
        """
        Tokenizes the input string into numbers, operators, and parentheses.
        """
        tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
            
            # Numbers (integers and floats)
            if char.isdigit() or char == '.':
                start = i
                # Consume digits and dots
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                
                num_str = expr[start:i]
                
                # Validation: check for multiple dots
                if num_str.count('.') > 1:
                    raise ValueError(f"Invalid number format: {num_str}")
                
                try:
                    tokens.append(float(num_str))
                except ValueError:
                    raise ValueError(f"Invalid number format: {num_str}")
                continue
            
            # Operators and Parentheses
            if char in '+-*/()':
                tokens.append(char)
                i += 1
                continue
            
            # Invalid character
            raise ValueError(f"Invalid token: '{char}'")
            
        return tokens


class Parser:
    """Internal parser class for recursive descent."""
    
    def __init__(self, tokens: List[Union[float, str]]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Union[float, str, None]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> Union[float, str]:
        token = self.current_token()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self.pos += 1
        return token

    def parse_expression(self) -> float:
        """Handles addition and subtraction (lowest precedence)."""
        left = self.parse_term()
        
        while self.current_token() in ('+', '-'):
            op = self.consume()
            right = self.parse_term()
            if op == '+':
                left += right
            else:
                left -= right
        return left

    def parse_term(self) -> float:
        """Handles multiplication and division (higher precedence)."""
        left = self.parse_factor()
        
        while self.current_token() in ('*', '/'):
            op = self.consume()
            right = self.parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def parse_factor(self) -> float:
        """Handles unary minus and atoms."""
        if self.current_token() == '-':
            self.consume()
            # Recursive call handles chained unary minus like --3
            return -self.parse_factor()
        return self.parse_atom()

    def parse_atom(self) -> float:
        """Handles numbers and parenthesized expressions."""
        token = self.current_token()
        
        if token == '(':
            self.consume()
            expr = self.parse_expression()
            if self.current_token() != ')':
                raise ValueError("Mismatched parentheses")
            self.consume() # consume ')'
            return expr
        elif token is None:
            raise ValueError("Unexpected end of expression")
        else:
            # Must be a number
            self.consume()
            return token

import pytest

def test_basic_precedence():
    """Test operator precedence (*, / before +, -)."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses():
    """Test grouping with parentheses."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((2 + 3))") == 5.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    """Test unary minus support."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0  # Chained unary minus
    assert evaluator.evaluate("3 * -2") == -6.0

def test_floating_point():
    """Test floating point number support."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("1.5 / 3") == pytest.approx(0.5)

def test_error_handling():
    """Test various error conditions."""
    evaluator = ExpressionEvaluator()
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
        
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
        
    # Invalid token
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1 + a")
        
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")