from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    
    Supports basic arithmetic operations (+, -, *, /) with correct precedence,
    parentheses for grouping, unary minus, and floating point numbers.
    """
    
    def __init__(self):
        """Initialize the evaluator with empty token list and position."""
        self._tokens: List[str] = []
        self._pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The numerical result of evaluating the expression
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        self._tokens = self._tokenize(expr)
        self._pos = 0
        
        if not self._tokens:
            raise ValueError("Empty expression")
        
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
            ValueError: If invalid characters or number formats are found
        """
        tokens: List[str] = []
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
                
                # Check for invalid number formats
                if num_str.startswith('.') or num_str.endswith('.'):
                    raise ValueError(f"Invalid number format: '{num_str}'")
                
                tokens.append(num_str)
                continue
            
            elif char in '+-*/()':
                tokens.append(char)
                i += 1
                continue
            
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")
        
        return tokens
    
    def _current_token(self) -> Optional[str]:
        """Get the current token or None if we've consumed all tokens."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None
    
    def _advance(self) -> str:
        """Advance to the next token and return the current one."""
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression")
        token = self._tokens[self._pos]
        self._pos += 1
        return token
    
    def _parse_expression(self) -> float:
        """
        Parse an expression: term (('+' | '-') term)*
        
        This handles addition and subtraction with lowest precedence.
        """
        result = self._parse_term()
        
        while self._current_token() in ('+', '-'):
            op = self._advance()
            right = self._parse_term()
            
            if op == '+':
                result = result + right
            else:  # op == '-'
                result = result - right
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse a term: factor (('*' | '/') factor)*
        
        This handles multiplication and division with higher precedence.
        """
        result = self._parse_factor()
        
        while self._current_token() in ('*', '/'):
            op = self._advance()
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
        Parse a factor: unary
        
        This is a wrapper that delegates to unary parsing.
        """
        return self._parse_unary()
    
    def _parse_unary(self) -> float:
        """
        Parse a unary expression: ('-' | '+') factor | primary
        
        This handles unary minus and plus operators.
        """
        if self._current_token() == '-':
            self._advance()
            return -self._parse_unary()
        elif self._current_token() == '+':
            self._advance()
            return self._parse_unary()
        else:
            return self._parse_primary()
    
    def _parse_primary(self) -> float:
        """
        Parse a primary: number | '(' expression ')'
        
        This handles numbers and parenthesized expressions.
        """
        token = self._current_token()
        
        if token is None:
            raise ValueError("Unexpected end of expression")
        
        if token == '(':
            self._advance()  # consume '('
            result = self._parse_expression()
            
            if self._current_token() != ')':
                raise ValueError("Missing closing parenthesis")
            
            self._advance()  # consume ')'
            return result
        
        elif token == ')':
            raise ValueError("Unexpected closing parenthesis")
        
        else:
            # Must be a number
            self._advance()
            try:
                return float(token)
            except ValueError:
                raise ValueError(f"Invalid number: '{token}'")

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