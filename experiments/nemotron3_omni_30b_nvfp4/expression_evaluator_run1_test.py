from typing import List, Union
import math


class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports:
    - Binary operators: +, -, *, /
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers (e.g., '3.14')
    
    Uses a recursive descent parser with proper operator precedence.
    
    Raises:
        ValueError: For mismatched parentheses, division by zero, invalid tokens, or empty expressions
    
    Example:
        evaluator = ExpressionEvaluator()
        result = evaluator.evaluate("3 + 4 * (2 - 1)")  # Returns 7.0
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
            ValueError: If the expression is invalid (mismatched parentheses, division by zero, etc.)
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        self.tokens = self._tokenize(expr)
        self.pos = 0
        result = self._expr()
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
        return result

    def _tokenize(self, expr: str) -> List[Union[float, str]]:
        """Convert expression string into a list of tokens (numbers and operators)."""
        tokens = []
        i = 0
        n = len(expr)
        while i < n:
            char = expr[i]
            if char.isspace():
                i += 1
                continue
            if char.isdigit() or char == '.':
                j = i
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                tokens.append(float(expr[i:j]))
                i = j
            elif char in '+-*/()':
                # Handle unary minus: if '-' is at start or after '('
                if char == '-' and (i == 0 or expr[i-1] in '(*+'):
                    # Look ahead to see if it's part of a number
                    j = i + 1
                    while j < n and (expr[j].isdigit() or expr[j] == '.'):
                        j += 1
                    if j > i + 1:  # There's a number following
                        tokens.append(float(expr[i:j]))
                        i = j
                        continue
                tokens.append(char)
                i += 1
            else:
                raise ValueError(f"Invalid character: {char}")
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
                    raise ValueError("Division by zero")
                result /= divisor
        return result

    def _factor(self) -> float:
        """Parse and evaluate a factor (parentheses or unary/binary number)."""
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
            if not isinstance(token, float):
                raise ValueError(f"Unexpected token: {token}")
            self.pos += 1
            return token


# Pytest tests
import pytest

def test_basic_addition():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 4") == 7.0

def test_operator_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 4 * 2") == 11.0
    assert evaluator.evaluate("3 * 4 + 2") == 14.0

def test_parentheses_and_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 + (-4 * 2)") == -5.0
    assert evaluator.evaluate("-(3.5)") == -3.5

def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

def test_mismatched_parentheses():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(3 + 4")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("3 + 4)")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("((3 + 4)")

def test_invalid_token():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("3 + x")

def test_empty_expression():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")