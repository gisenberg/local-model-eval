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
        self.tokens: List[Union[str, float]] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: The expression to evaluate (e.g., "3 + 4 * (2 - 1)")
            
        Returns:
            The result of the expression as a float
            
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
        except (ValueError, IndexError):
            raise ValueError("Invalid expression")
    
    def _tokenize(self, expr: str) -> List[Union[str, float]]:
        """Tokenize the expression into numbers, operators, and parentheses."""
        tokens = []
        # Regex pattern to match numbers (including floats), operators, and parentheses
        pattern = r'\s*([0-9]*\.?[0-9]+|[+\-*/()])\s*'
        for match in re.finditer(pattern, expr):
            token = match.group(1)
            if token in '+-*/()':
                tokens.append(token)
            else:
                # Convert numeric tokens to float
                tokens.append(float(token))
        return tokens
    
    def _expr(self) -> float:
        """Parse and evaluate an expression (lowest precedence)."""
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
        """Parse and evaluate a term (medium precedence)."""
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
        """Parse and evaluate a factor (highest precedence)."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
            
        token = self.tokens[self.pos]
        
        # Handle unary minus
        if token == '-' and (self.pos == 0 or self.tokens[self.pos - 1] in ('+', '-', '*', '/', '(')):
            self.pos += 1
            if self.pos >= len(self.tokens):
                raise ValueError("Invalid unary minus")
            return -self._factor()
        
        # Handle parentheses
        if token == '(':
            self.pos += 1
            result = self._expr()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result
        
        # Handle numbers
        if isinstance(token, float):
            self.pos += 1
            return token
            
        raise ValueError(f"Invalid token: {token}")

# Pytest tests
def test_basic_addition():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 4") == 7.0

def test_operator_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + (3 * 4)
    assert evaluator.evaluate("2 * 3 + 4") == 10.0  # (2 * 3) + 4

def test_parentheses_and_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-3 * (2 + 1)") == -9.0
    assert evaluator.evaluate("3 + -4 * 2") == -5.0

def test_floating_point_and_division():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("10 / 2") == 5.0
    assert evaluator.evaluate("5 / 2") == 2.5

def test_errors():
    evaluator = ExpressionEvaluator()
    try:
        evaluator.evaluate("(3 + 2")
        assert False, "Should raise ValueError for mismatched parentheses"
    except ValueError as e:
        assert "Mismatched parentheses" in str(e)
    
    try:
        evaluator.evaluate("5 / 0")
        assert False, "Should raise ValueError for division by zero"
    except ValueError as e:
        assert "Division by zero" in str(e)
    
    try:
        evaluator.evaluate("3 + * 4")
        assert False, "Should raise ValueError for invalid token"
    except ValueError as e:
        assert "Invalid token" in str(e)
    
    try:
        evaluator.evaluate("")
        assert False, "Should raise ValueError for empty expression"
    except ValueError as e:
        assert "Empty expression" in str(e)