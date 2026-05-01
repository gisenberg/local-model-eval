import re
from typing import List, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports:
    - Binary operators: +, -, *, /
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers
    
    Uses a recursive descent parser with proper operator precedence.
    
    Raises:
        ValueError: For mismatched parentheses, division by zero, invalid tokens, or empty expressions.
    """
    
    def __init__(self):
        # Token patterns for lexer
        self.number_pattern = r'\d+(?:\.\d*)?'
        self.operator_pattern = r'[+\-*/()]'
        self.whitespace_pattern = r'\s+'
        
        # Combined token pattern (numbers, operators, parentheses, whitespace)
        self.token_pattern = re.compile(
            f'({self.number_pattern})|({self.operator_pattern})|({self.whitespace_pattern})'
        )
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: The expression to evaluate (e.g., "3 + 4 * (2 - 1)")
            
        Returns:
            The result of the evaluated expression as a float
            
        Raises:
            ValueError: If the expression is invalid (mismatched parentheses, division by zero, etc.)
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Invalid expression: no valid tokens found")
            
        parser = Parser(tokens)
        result = parser.parse()
        if parser.has_error:
            raise ValueError(f"Invalid expression: {parser.error_message}")
        return result
    
    def _tokenize(self, expr: str) -> List[str]:
        """Tokenize the input expression into numbers, operators, and parentheses."""
        tokens = []
        pos = 0
        while pos < len(expr):
            match = self.token_pattern.match(expr, pos)
            if not match:
                raise ValueError(f"Invalid character at position {pos}: '{expr[pos]}'")
                
            token = match.group(1) or match.group(2) or match.group(3)
            if token.strip():  # Skip whitespace
                tokens.append(token)
            pos = match.end()
            
        return tokens
    
    def _peek(self, tokens: List[str], index: int) -> Union[str, None]:
        """Return the token at index without consuming it, or None if out of bounds."""
        if index < len(tokens):
            return tokens[index]
        return None
    
    def _advance(self, tokens: List[str], index: int) -> Union[str, None]:
        """Return the token at index and advance the index, or None if out of bounds."""
        if index < len(tokens):
            token = tokens[index]
            return token, index + 1
        return None, index


class Parser:
    """
    Recursive descent parser for mathematical expressions.
    
    Grammar:
        expression   : term (('+' | '-') term)*
        term         : factor (('*' | '/') factor)*
        factor       : ('+' | '-')? primary
        primary      : number | '(' expression ')'
    """
    
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0
        self.has_error = False
        self.error_message = ""
    
    def parse(self) -> float:
        """Parse and evaluate the expression."""
        result = self.expression()
        if self.pos >= len(self.tokens):
            self.has_error = False
            return result
        self.has_error = True
        self.error_message = f"Unexpected token: '{self.tokens[self.pos]}'"
        raise ValueError(self.error_message)
    
    def expression(self) -> float:
        """Parse an expression: term (('+' | '-') term)*"""
        result = self.term()
        while self._peek(self.tokens, self.pos) in ('+', '-'):
            op = self._advance(self.tokens, self.pos)[0]
            if op == '+':
                result += self.term()
            elif op == '-':
                result -= self.term()
        return result
    
    def term(self) -> float:
        """Parse a term: factor (('*' | '/') factor)*"""
        result = self.factor()
        while self._peek(self.tokens, self.pos) in ('*', '/'):
            op = self._advance(self.tokens, self.pos)[0]
            if op == '*':
                result *= self.factor()
            elif op == '/':
                divisor = self.factor()
                if divisor == 0:
                    self.has_error = True
                    self.error_message = "Division by zero"
                    return 0
                result /= divisor
        return result
    
    def factor(self) -> float:
        """Parse a factor: ('+' | '-')? primary"""
        # Handle unary plus/minus
        if self._peek(self.tokens, self.pos) == '-':
            self._advance(self.tokens, self.pos)
            result = -self.factor()
        elif self._peek(self.tokens, self.pos) == '+':
            self._advance(self.tokens, self.pos)
            result = self.factor()
        else:
            result = self.primary()
        return result
    
    def primary(self) -> float:
        """Parse a primary expression: number | '(' expression ')'"""
        if self._peek(self.tokens, self.pos) == '(':
            self._advance(self.tokens, self.pos)  # consume '('
            result = self.expression()
            if self._peek(self.tokens, self.pos) != ')':
                self.has_error = True
                self.error_message = "Mismatched parentheses: expected ')'"
                return 0
            self._advance(self.tokens, self.pos)  # consume ')'
            return result
        else:
            # Parse number
            token = self._advance(self.tokens, self.pos)[0]
            if not re.match(self.number_pattern, token):
                self.has_error = True
                self.error_message = f"Invalid token: '{token}'"
                return 0
            return float(token)

import pytest

def test_basic_operations():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 4 * 2") == 11.0
    assert evaluator.evaluate("3 + 4 - 2") == 5.0
    assert evaluator.evaluate("3 * 4 - 2") == 10.0
    assert evaluator.evaluate("3 / 2") == 1.5

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(3 + 4) * 2") == 14.0
    assert evaluator.evaluate("3 + (4 * 2)") == 11.0
    assert evaluator.evaluate("3 + (4 + 2)") == 9.0
    assert evaluator.evaluate("((3 + 4) * 2)") == 14.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(3 + 4)") == -7.0
    assert evaluator.evaluate("-(3 * 4)") == -12.0
    assert evaluator.evaluate("3 + -4") == -1.0

def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 + 2.5") == 5.64
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("3.14 / 2") == 1.57

def test_errors():
    evaluator = ExpressionEvaluator()
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(3 + 4")
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("3 / 0")
    
    # Invalid token
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 + x")
    
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")