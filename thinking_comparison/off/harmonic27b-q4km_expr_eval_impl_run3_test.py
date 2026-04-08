from typing import Iterator, List, Tuple, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    Supports +, -, *, / with correct precedence, parentheses, unary minus,
    and floating point numbers.
    """
    
    def __init__(self):
        self._tokens: List[str] = []
        self._pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The numerical result of the expression
            
        Raises:
            ValueError: For invalid expressions, mismatched parentheses, 
                       division by zero, or other parsing errors
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        self._tokens = self._tokenize(expr)
        self._pos = 0
        
        if not self._tokens:
            raise ValueError("Empty expression")
            
        result = self._parse_expression()
        
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token: {self._tokens[self._pos]}")
            
        return result
    
    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert expression string into a list of tokens.
        
        Args:
            expr: The input expression string
            
        Returns:
            List of token strings (numbers, operators, parentheses)
        """
        tokens = []
        i = 0
        expr = expr.replace(' ', '')
        
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
                
                if not num_str or num_str == '.':
                    raise ValueError(f"Invalid number format at position {i-1}")
                    
                tokens.append(num_str)
                continue
                
            elif char in '+-*/()':
                tokens.append(char)
                i += 1
                continue
                
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")
        
        return tokens
    
    def _current_token(self) -> Union[str, None]:
        """Return the current token or None if at end."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None
    
    def _consume(self) -> str:
        """Consume and return the current token."""
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression")
        token = self._tokens[self._pos]
        self._pos += 1
        return token
    
    def _parse_expression(self) -> float:
        """
        Parse addition and subtraction (lowest precedence).
        """
        left = self._parse_term()
        
        while self._current_token() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            
            if op == '+':
                left = left + right
            else:  # op == '-'
                left = left - right
                
        return left
    
    def _parse_term(self) -> float:
        """
        Parse multiplication and division (higher precedence than +/-).
        """
        left = self._parse_factor()
        
        while self._current_token() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            
            if op == '*':
                left = left * right
            else:  # op == '/'
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
                
        return left
    
    def _parse_factor(self) -> float:
        """
        Parse unary operators, numbers, and parenthesized expressions.
        """
        # Handle unary minus
        if self._current_token() == '-':
            self._consume()
            return -self._parse_factor()
            
        # Handle unary plus (though not explicitly required, it's good practice)
        if self._current_token() == '+':
            self._consume()
            return self._parse_factor()
            
        # Handle parenthesized expressions
        if self._current_token() == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            
            if self._current_token() != ')':
                raise ValueError("Missing closing parenthesis")
            self._consume()  # consume ')'
            
            return result
            
        # Handle numbers
        if self._current_token() and self._is_number(self._current_token()):
            num_str = self._consume()
            try:
                return float(num_str)
            except ValueError:
                raise ValueError(f"Invalid number: {num_str}")
        
        raise ValueError(f"Unexpected token: {self._current_token()}")
    
    def _is_number(self, token: str) -> bool:
        """Check if a token represents a valid number."""
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