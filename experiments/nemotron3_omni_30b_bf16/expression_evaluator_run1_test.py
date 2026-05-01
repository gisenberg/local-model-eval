import re
from typing import List, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports:
    - Binary operators: +, -, *, /
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers (e.g., '3.14')
    
    Uses recursive descent parsing with proper operator precedence.
    
    Raises:
        ValueError: For mismatched parentheses, division by zero, invalid tokens, or empty expressions.
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

    def _tokenize(self, expr: str) -> List[Union[float, str]]:
        """Tokenize the expression into numbers and operators."""
        # Pattern matches numbers (including floats), operators, and parentheses
        token_pattern = r'\d+\.?\d*|[+\-*/()]'
        tokens = re.findall(token_pattern, expr)
        
        # Convert number strings to floats
        tokens = [float(token) if re.match(r'\d+\.?\d*', token) else token for token in tokens]
        
        # Check for invalid tokens
        if len(tokens) != len(re.findall(token_pattern, expr)):
            raise ValueError("Invalid token in expression")
            
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
        """Parse and evaluate a factor (handling unary minus and parentheses)."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
            
        token = self.tokens[self.pos]
        
        if token == '-':
            self.pos += 1
            if self.pos >= len(self.tokens):
                raise ValueError("Unary minus at end of expression")
            return -self._factor()
            
        elif token == '(':
            self.pos += 1
            result = self._expr()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result
            
        elif isinstance(token, float):
            self.pos += 1
            return token
            
        else:
            raise ValueError(f"Unexpected token: {token}")

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
    assert evaluator.evaluate("3 + -4") == -1.0

def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    try:
        evaluator.evaluate("3 / 0")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Division by zero"

def test_mismatched_parentheses():
    evaluator = ExpressionEvaluator()
    try:
        evaluator.evaluate("(3 + 4")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Mismatched parentheses" in str(e)

def test_invalid_token():
    evaluator = ExpressionEvaluator()
    try:
        evaluator.evaluate("3 + x")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid token" in str(e)

def test_empty_expression():
    evaluator = ExpressionEvaluator()
    try:
        evaluator.evaluate("")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Empty expression" in str(e)