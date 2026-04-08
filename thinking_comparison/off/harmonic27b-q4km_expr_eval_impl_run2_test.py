from typing import List, Tuple, Iterator

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
        """Convert expression string into tokens."""
        self.tokens = []
        i = 0
        
        while i < len(expr):
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
                
            if char.isdigit() or char == '.':
                num_str = ""
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
                
            if char in '+-*/()':
                self.tokens.append(char)
                i += 1
                continue
                
            raise ValueError(f"Invalid character '{char}' at position {i}")

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            
            if op == '+':
                left = left + right
            else:
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
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
                
        return left

    def _parse_factor(self) -> float:
        """Parse unary operators and primary expressions."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
            
        token = self.tokens[self.pos]
        
        if token == '-':
            self.pos += 1
            return -self._parse_factor()
            
        if token == '+':
            self.pos += 1
            return self._parse_factor()
            
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
            
        token = self.tokens[self.pos]
        
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Missing closing parenthesis")
                
            self.pos += 1
            return result
            
        if token in '+-*/':
            raise ValueError(f"Unexpected operator '{token}' at position {self.pos}")
            
        # Parse number
        try:
            result = float(token)
        except ValueError:
            raise ValueError(f"Invalid number '{token}' at position {self.pos}")
            
        self.pos += 1
        return result

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