import re
from typing import List, Union, Optional

Token = Union[str, float]

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Operators: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14, .5, 5.)
    
    Raises ValueError for invalid syntax, mismatched parentheses, 
    division by zero, and empty expressions.
    """
    
    def __init__(self) -> None:
        self.tokens: List[Token] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.
        
        Args:
            expr: A string containing a mathematical expression.
            
        Returns:
            The evaluated result as a float.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Empty expression")
            
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
            
        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Convert expression string into a list of tokens."""
        if not expr.strip():
            raise ValueError("Empty expression")
            
        # Matches numbers (int/float) and operators/parentheses, ignoring whitespace
        pattern = re.compile(r'\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+|[+\-*/()])\s*')
        tokens: List[Token] = []
        pos = 0
        
        while pos < len(expr):
            match = pattern.match(expr, pos)
            if not match:
                raise ValueError(f"Invalid token at position {pos}: '{expr[pos]}'")
                
            token_str = match.group(1)
            if token_str in '+-*/()':
                tokens.append(token_str)
            else:
                try:
                    tokens.append(float(token_str))
                except ValueError:
                    raise ValueError(f"Invalid number: '{token_str}'")
                    
            pos = match.end()
            
        return tokens

    def _current_token(self) -> Optional[Token]:
        """Return the current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self, expected: Optional[str] = None) -> Token:
        """Consume and return the current token. Optionally validate it."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
            
        token = self.tokens[self.pos]
        if expected is not None and token != expected:
            raise ValueError(f"Expected '{expected}', got '{token}'")
            
        self.pos += 1
        return token  # type: ignore

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary operators (+, -)."""
        token = self._current_token()
        if token == '-':
            self._consume()
            return -self._parse_factor()
        if token == '+':
            self._consume()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        token = self._current_token()
        if isinstance(token, float):
            self._consume()
            return token
        if token == '(':
            self._consume()
            result = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()
            return result
        raise ValueError(f"Unexpected token: {token}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_precedence(evaluator):
    """Test correct operator precedence for +, -, *, /"""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping(evaluator):
    """Test parentheses override default precedence"""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert evaluator.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

def test_unary_minus(evaluator):
    """Test unary minus handling in various contexts"""
    assert evaluator.evaluate("-3 * 2") == -6.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("10 + -2.5") == 7.5

def test_floating_point_numbers(evaluator):
    """Test support for decimal numbers"""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("10.0 / 3.0") == pytest.approx(3.3333333333333335)

def test_error_handling(evaluator):
    """Test ValueError raising for invalid inputs"""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(3 + 2")
        
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 @ 2")
        
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
        
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("3 + + 2")