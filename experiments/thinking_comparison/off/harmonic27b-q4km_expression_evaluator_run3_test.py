from typing import List, Tuple, Iterator


class ExpressionEvaluator:
    """
    A recursive descent parser for mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus
    - Floating point numbers
    
    Raises ValueError for invalid expressions.
    """
    
    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The numerical result of the expression
            
        Raises:
            ValueError: If the expression is invalid
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")
        
        self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Expression cannot be empty")
        
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at position {self.pos}")
        
        return result
    
    def _tokenize(self, expr: str) -> None:
        """Convert expression string into tokens."""
        self.tokens = []
        i = 0
        expr = expr.strip()
        
        while i < len(expr):
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
            
            if char.isdigit() or char == '.':
                # Parse number (integer or float)
                num_str = ""
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
                
                self.tokens.append(num_str)
                continue
            
            if char in '+-*/()':
                self.tokens.append(char)
                i += 1
                continue
            
            raise ValueError(f"Invalid character '{char}' at position {i}")
    
    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            
            if op == '+':
                result = result + right
            else:
                result = result - right
        
        return result
    
    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()
            
            if op == '*':
                result = result * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result = result / right
        
        return result
    
    def _parse_factor(self) -> float:
        """Parse numbers, parentheses, and unary minus."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        
        token = self.tokens[self.pos]
        
        # Handle unary minus
        if token == '-':
            self.pos += 1
            return -self._parse_factor()
        
        # Handle parentheses
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Missing closing parenthesis")
            
            self.pos += 1
            return result
        
        # Handle numbers
        if token.replace('.', '').isdigit():
            self.pos += 1
            return float(token)
        
        raise ValueError(f"Unexpected token '{token}'")


# Test suite
def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0
    assert evaluator.evaluate("2.5 + 3.5") == 6.0


def test_operator_precedence():
    """Test that multiplication/division have higher precedence than addition/subtraction."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + (3*4) = 14
    assert evaluator.evaluate("10 - 2 * 3") == 4.0   # 10 - (2*3) = 4
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0  # (2*3) + (4*5) = 26
    assert evaluator.evaluate("10 / 2 + 3") == 8.0     # (10/2) + 3 = 8


def test_parentheses():
    """Test parentheses for grouping."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0
    assert evaluator.evaluate("(10 - 2) / (3 - 1)") == 4.0


def test_unary_minus():
    """Test unary minus operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(-3)") == 3.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("5 + -3") == 2.0
    assert evaluator.evaluate("5 - -3") == 8.0
    assert evaluator.evaluate("-2 * 3") == -6.0


def test_error_cases():
    """Test various error conditions."""
    evaluator = ExpressionEvaluator()
    
    # Empty expression
    try:
        evaluator.evaluate("")
        assert False, "Should raise ValueError for empty expression"
    except ValueError as e:
        assert "empty" in str(e).lower()
    
    # Mismatched parentheses
    try:
        evaluator.evaluate("(2 + 3")
        assert False, "Should raise ValueError for missing closing paren"
    except ValueError as e:
        assert "parenthesis" in str(e).lower()
    
    try:
        evaluator.evaluate("2 + 3)")
        assert False, "Should raise ValueError for extra closing paren"
    except ValueError as e:
        assert "unexpected" in str(e).lower()
    
    # Division by zero
    try:
        evaluator.evaluate("5 / 0")
        assert False, "Should raise ValueError for division by zero"
    except ValueError as e:
        assert "division by zero" in str(e).lower()
    
    # Invalid tokens
    try:
        evaluator.evaluate("2 + a")
        assert False, "Should raise ValueError for invalid token"
    except ValueError as e:
        assert "invalid" in str(e).lower() or "unexpected" in str(e).lower()


if __name__ == "__main__":
    # Run tests
    test_basic_arithmetic()
    test_operator_precedence()
    test_parentheses()
    test_unary_minus()
    test_error_cases()
    print("All tests passed!")