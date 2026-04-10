import re
from typing import List

class ExpressionEvaluator:
    """
    A recursive descent parser for mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """
    
    def __init__(self) -> None:
        """Initialize the evaluator."""
        self.tokens: List[str] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The result of the evaluation as a float
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        self._tokenize(expr)
        
        if not self.tokens:
            raise ValueError("Empty expression")
        
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at position {self.pos}")
        
        return result
    
    def _tokenize(self, expr: str) -> None:
        """
        Tokenize the expression string into a list of tokens.
        
        Args:
            expr: The expression string to tokenize
            
        Raises:
            ValueError: If the expression contains invalid characters
        """
        # Pattern matches: numbers (int or float), operators, parentheses
        pattern = r'(\d+\.?\d*|\.\d+|[+*/()-])'
        tokens = re.findall(pattern, expr)
        
        # Check for invalid characters (characters not matched by pattern)
        cleaned = ''.join(tokens)
        if cleaned != re.sub(r'\s+', '', expr):
            raise ValueError("Invalid token in expression")
        
        self.tokens = tokens
        self.pos = 0
    
    def _current_token(self) -> str:
        """Return the current token or empty string if at end."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ""
    
    def _consume(self) -> str:
        """
        Consume and return the current token.
        
        Raises:
            ValueError: If at end of tokens
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        token = self.tokens[self.pos]
        self.pos += 1
        return token
    
    def _parse_expression(self) -> float:
        """
        Parse an expression: Term (('+' | '-') Term)*
        
        Returns:
            The evaluated value of the expression
        """
        value = self._parse_term()
        
        while self._current_token() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                value += right
            else:
                value -= right
        
        return value
    
    def _parse_term(self) -> float:
        """
        Parse a term: Factor (('*' | '/') Factor)*
        
        Returns:
            The evaluated value of the term
        """
        value = self._parse_factor()
        
        while self._current_token() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                value *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                value /= right
        
        return value
    
    def _parse_factor(self) -> float:
        """
        Parse a factor: Number | '(' Expression ')' | '-' Factor (unary minus)
        
        Returns:
            The evaluated value of the factor
            
        Raises:
            ValueError: If the token is invalid or parentheses are mismatched
        """
        token = self._current_token()
        
        if token == '(':
            self._consume()  # consume '('
            value = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses: expected ')'")
            self._consume()  # consume ')'
            return value
        
        elif token == '-':
            # Unary minus
            self._consume()
            return -self._parse_factor()
        
        elif token and (token[0].isdigit() or (token[0] == '.' and len(token) > 1 and token[1].isdigit())):
            # Number
            self._consume()
            try:
                return float(token)
            except ValueError:
                raise ValueError(f"Invalid number: '{token}'")
        
        else:
            raise ValueError(f"Unexpected token: '{token}'")

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