import re
from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    
    Supports:
    - Binary operators: +, -, *, /
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers (e.g., '3.14')
    
    Raises:
    - ValueError: for mismatched parentheses, division by zero, invalid tokens, or empty expressions
    
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
            The result as a float
            
        Raises:
            ValueError: If the expression is invalid
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Invalid expression")
            
        result = self._expr()
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
            
        return result
    
    def _tokenize(self, expr: str) -> List[Union[float, str]]:
        """Tokenize the expression into numbers and operators."""
        # Remove spaces and handle unary minus by inserting 'u-' before minus
        expr = expr.replace(' ', '')
        expr = re.sub(r'-(?=[0-9])', 'u-', expr)  # Unary minus
        expr = re.sub(r'-(?=[(])', 'u-', expr)    # Unary minus before '('
        
        # Tokenize numbers and operators
        token_spec = [
            ('NUMBER', r'\d+\.\d*|\.\d+|\d+'),  # Integer or decimal number
            ('OP', r'[+\-*/]'),                  # Arithmetic operators
            ('LPAREN', r'\('),                   # Left parenthesis
            ('RPAREN', r'\)'),                   # Right parenthesis
        ]
        
        tokens = []
        for tok_type, pattern in token_spec:
            regex = re.compile(pattern)
            while True:
                match = regex.match(expr)
                if not match:
                    break
                value = match.group()
                if tok_type == 'NUMBER':
                    tokens.append(float(value))
                else:
                    tokens.append(value)
                expr = expr[match.end():]
                
        if expr:
            raise ValueError(f"Invalid token: {expr}")
            
        return tokens
    
    def _expr(self) -> float:
        """Parse an expression (addition and subtraction)."""
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
        """Parse a term (multiplication and division)."""
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
        """Parse a factor (numbers, parentheses, or unary operations)."""
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
        elif token == 'u-':  # Unary minus
            self.pos += 1
            return -self._factor()
        else:
            if not isinstance(token, float):
                raise ValueError(f"Invalid token: {token}")
            self.pos += 1
            return token

# Pytest tests
def test_basic_addition():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 4") == 7.0

def test_operator_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 4 * 2") == 11.0
    assert evaluator.evaluate("3 * 4 + 2") == 14.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(3 + 4) * 2") == 14.0
    assert evaluator.evaluate("3 + (4 * 2)") == 11.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 * -2") == -6.0

def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    try:
        evaluator.evaluate("5 / 0")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Division by zero" in str(e)

def test_mismatched_parentheses():
    evaluator = ExpressionEvaluator()
    try:
        evaluator.evaluate("(3 + 4")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Mismatched parentheses" in str(e)

def test_invalid_token():
    evaluator = ExpressionEvaluator()
    try:
        evaluator.evaluate("3 + 4 $ 2")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Invalid token" in str(e)

def test_empty_expression():
    evaluator = ExpressionEvaluator()
    try:
        evaluator.evaluate("")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Empty expression" in str(e)