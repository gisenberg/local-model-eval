import re
from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports:
    - Binary operators: +, -, *, /
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers (e.g., '3.14')
    
    Uses recursive descent parsing with proper operator precedence.
    
    Raises:
        ValueError: For mismatched parentheses, division by zero, invalid tokens, or empty expressions
    """
    
    def __init__(self):
        self.tokens: List[Union[float, str]] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: The expression to evaluate (e.g., "3 + 4 * (2 - 1)")
            
        Returns:
            The result of the evaluation as a float
            
        Raises:
            ValueError: If the expression is invalid
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        try:
            result = self._expr()
            if self.pos < len(self.tokens):
                raise ValueError(f"Invalid token at position {self.pos}")
            return result
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")
    
    def _tokenize(self, expr: str) -> List[Union[float, str]]:
        """Convert expression string into list of tokens (numbers and operators)."""
        # Remove spaces and handle unary minus by inserting 0 before minus if needed
        expr = expr.replace(' ', '')
        tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            if expr[i] in '+-*/()':
                # Check for unary minus: if minus is at start or after '('
                if expr[i] == '-' and (i == 0 or expr[i-1] in '(*'):
                    # Handle unary minus by inserting 0
                    tokens.append(0.0)
                    tokens.append('-')
                else:
                    tokens.append(expr[i])
                i += 1
            else:
                # Parse number (including float)
                j = i
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                if j == i:
                    raise ValueError(f"Invalid token: '{expr[i]}'")
                tokens.append(float(expr[i:j]))
                i = j
                
        return tokens
    
    def _expr(self) -> float:
        """Parse and evaluate an expression (addition and subtraction)."""
        result = self._term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            if op == '+':
                result += self._term()
            else:
                result -= self._term()
        return result
    
    def _term(self) -> float:
        """Parse and evaluate a term (multiplication and division)."""
        result = self._factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            if op == '*':
                result *= self._factor()
            else:
                divisor = self._factor()
                if divisor == 0:
                    raise ZeroDivisionError("Division by zero")
                result /= divisor
        return result
    
    def _factor(self) -> float:
        """Parse and evaluate a factor (parentheses or number)."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
            
        token = self.tokens[self.pos]
        
        if token == '(':
            self.pos += 1
            result = self._expr()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result
        elif token == '-':
            self.pos += 1
            return -self._factor()
        else:
            # Must be a number
            if not isinstance(token, float):
                raise ValueError(f"Invalid token: '{token}'")
            self.pos += 1
            return token

# Pytest tests
import pytest

def test_basic_operations():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    assert evaluator.evaluate("3.5 * 2") == 7.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(1 + 2) * 3") == 9.0
    assert evaluator.evaluate("3 * (2 + 1)") == 9.0
    assert evaluator.evaluate("((1 + 2) * 3)") == 9.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 * -2") == -6.0

def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

def test_mismatched_parentheses():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("1 + 2)")

def test_invalid_token():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1 + x")

def test_empty_expression():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")