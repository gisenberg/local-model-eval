from typing import List, Tuple, Union


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports: +, -, *, / with correct precedence, parentheses, unary minus,
    and floating point numbers.
    """
    
    def __init__(self):
        self.tokens: List[Tuple[str, Union[str, float]]] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: A string containing a mathematical expression with numbers,
                  operators (+, -, *, /), parentheses, and unary minus.
                  
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                       has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")
        
        self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Expression cannot be empty")
        
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
        
        return result
    
    def _tokenize(self, expr: str) -> None:
        """
        Convert the expression string into a list of tokens.
        
        Args:
            expr: The expression string to tokenize.
            
        Raises:
            ValueError: If the expression contains invalid characters or malformed numbers.
        """
        self.tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
            
            if char.isdigit() or char == '.':
                # Parse number (integer or float)
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format at position {j}: multiple decimal points")
                        has_dot = True
                    j += 1
                
                if i == j:
                    raise ValueError(f"Invalid character: '{char}'")
                
                num_str = expr[i:j]
                if num_str == '.':
                    raise ValueError(f"Invalid number format at position {i}: '.' is not a valid number")
                
                try:
                    num = float(num_str)
                    self.tokens.append(('NUMBER', num))
                except ValueError:
                    raise ValueError(f"Invalid number: '{num_str}'")
                
                i = j
                continue
            
            if char in '+-*/()':
                self.tokens.append((char, char))
                i += 1
                continue
            
            raise ValueError(f"Invalid character: '{char}'")
    
    def _parse_expression(self) -> float:
        """
        Parse addition and subtraction operations (lowest precedence).
        
        Grammar: Expression = Term (('+' | '-') Term)*
        
        Returns:
            The result of the parsed expression.
            
        Raises:
            ValueError: If the expression is malformed.
        """
        result = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos][0] in ('+', '-'):
            op = self.tokens[self.pos][0]
            self.pos += 1
            
            if self.pos >= len(self.tokens):
                raise ValueError("Unexpected end of expression after operator")
            
            right = self._parse_term()
            
            if op == '+':
                result += right
            else:
                result -= right
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse multiplication and division operations (higher precedence).
        
        Grammar: Term = Factor (('*' | '/') Factor)*
        
        Returns:
            The result of the parsed term.
            
        Raises:
            ValueError: If the expression is malformed or involves division by zero.
        """
        result = self._parse_factor()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos][0] in ('*', '/'):
            op = self.tokens[self.pos][0]
            self.pos += 1
            
            if self.pos >= len(self.tokens):
                raise ValueError("Unexpected end of expression after operator")
            
            right = self._parse_factor()
            
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parse numbers, parentheses, and unary minus (highest precedence).
        
        Grammar: Factor = Number | '(' Expression ')' | '-' Factor
        
        Returns:
            The result of the parsed factor.
            
        Raises:
            ValueError: If the expression is malformed or has mismatched parentheses.
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        
        token_type, token_value = self.tokens[self.pos]
        
        # Handle unary minus
        if token_type == '-':
            self.pos += 1
            if self.pos >= len(self.tokens):
                raise ValueError("Unexpected end of expression after unary minus")
            return -self._parse_factor()
        
        # Handle number
        if token_type == 'NUMBER':
            self.pos += 1
            return token_value
        
        # Handle parentheses
        if token_type == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos][0] != ')':
                raise ValueError("Mismatched parentheses: expected closing parenthesis")
            
            self.pos += 1
            return result
        
        raise ValueError(f"Unexpected token: {token_type}")

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