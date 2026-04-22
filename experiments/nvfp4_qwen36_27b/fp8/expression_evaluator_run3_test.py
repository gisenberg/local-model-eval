import pytest
from typing import Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Operator precedence (*, / before +, -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14)
    
    Raises ValueError for invalid syntax, division by zero, or empty expressions.
    """
    
    def __init__(self):
        self.expr: str = ""
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: A string containing a mathematical expression.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens, 
                        has mismatched parentheses, or involves division by zero.
        """
        self.expr = expr
        self.pos = 0
        
        self._skip_whitespace()
        if self.pos >= len(self.expr):
            raise ValueError("Empty expression")
            
        result = self._parse_expression()
        
        self._skip_whitespace()
        if self.pos < len(self.expr):
            raise ValueError(f"Invalid token at end of expression: '{self.expr[self.pos]}'")
            
        return result

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Grammar: Expression -> Term ( ('+' | '-') Term )*
        """
        result = self._parse_term()
        
        while True:
            self._skip_whitespace()
            if self._peek() in ('+', '-'):
                op = self._consume()
                right = self._parse_term()
                if op == '+':
                    result += right
                else:
                    result -= right
            else:
                break
        return result

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Grammar: Term -> Factor ( ('*' | '/') Factor )*
        """
        result = self._parse_factor()
        
        while True:
            self._skip_whitespace()
            if self._peek() in ('*', '/'):
                op = self._consume()
                right = self._parse_factor()
                if op == '*':
                    result *= right
                else:
                    if right == 0:
                        raise ValueError("Division by zero")
                    result /= right
            else:
                break
        return result

    def _parse_factor(self) -> float:
        """
        Parses numbers, parentheses, and unary operators.
        Grammar: Factor -> Number | '(' Expression ')' | ('-' | '+') Factor
        """
        self._skip_whitespace()
        char = self._peek()
        
        if char == '(':
            self._consume()
            result = self._parse_expression()
            self._skip_whitespace()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: expected ')'")
            self._consume()
            return result
        elif char == '-':
            self._consume()
            return -self._parse_factor()
        elif char == '+':
            self._consume()
            return self._parse_factor()
        elif char is not None and (char.isdigit() or char == '.'):
            return self._parse_number()
        else:
            if char is None:
                raise ValueError("Unexpected end of expression")
            raise ValueError(f"Invalid token: '{char}'")

    def _parse_number(self) -> float:
        """
        Parses a numeric literal (integer or float).
        """
        start = self.pos
        while self.pos < len(self.expr) and (self.expr[self.pos].isdigit() or self.expr[self.pos] == '.'):
            self.pos += 1
        
        token = self.expr[start:self.pos]
        if not token:
            raise ValueError("Invalid number format")
            
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid number: '{token}'")

    def _peek(self) -> Optional[str]:
        """Returns the character at the current position without consuming it."""
        if self.pos < len(self.expr):
            return self.expr[self.pos]
        return None

    def _consume(self) -> str:
        """Returns the character at the current position and advances the pointer."""
        if self.pos >= len(self.expr):
            raise ValueError("Unexpected end of expression")
        char = self.expr[self.pos]
        self.pos += 1
        return char

    def _skip_whitespace(self) -> None:
        """Advances the pointer past any whitespace characters."""
        while self.pos < len(self.expr) and self.expr[self.pos].isspace():
            self.pos += 1


# --- Pytest Tests ---

def test_operator_precedence():
    """Test that multiplication/division happens before addition/subtraction."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping():
    """Test that parentheses override default precedence."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((2 + 3) * 4) / 2") == 10.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus_and_floats():
    """Test support for unary minus and floating point numbers."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3.14") == -3.14
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0
    assert evaluator.evaluate("--5") == 5.0

def test_division_by_zero():
    """Test that division by zero raises ValueError."""
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("(2 - 2) / 5")

def test_invalid_syntax_errors():
    """Test error handling for invalid tokens and mismatched parentheses."""
    evaluator = ExpressionEvaluator()
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    
    # Invalid token
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")
        
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")