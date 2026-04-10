from typing import Optional, List, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser for mathematical expressions.
    
    Supports: +, -, *, /, parentheses, unary minus, and floating point numbers.
    """
    
    def __init__(self):
        self.tokens: List[Tuple[str, Optional[float]]] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Args:
            expr: The expression string to evaluate
            
        Returns:
            The result as a float
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        result = self._parse_expression()
        
        if self._peek() != 'EOF':
            raise ValueError(f"Unexpected token after expression: {self._peek()}")
        
        return result
    
    def _tokenize(self, expr: str) -> List[Tuple[str, Optional[float]]]:
        """
        Convert expression string into tokens.
        
        Args:
            expr: The expression string
            
        Returns:
            List of (token_type, value) tuples
            
        Raises:
            ValueError: If invalid characters are found
        """
        tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
            
            elif char.isdigit() or char == '.':
                # Parse number
                j = i
                has_dot = False
                
                while j < n:
                    c = expr[j]
                    if c.isdigit():
                        j += 1
                    elif c == '.' and not has_dot:
                        has_dot = True
                        j += 1
                    else:
                        break
                
                if i == j:
                    raise ValueError(f"Invalid character: {char}")
                
                num_str = expr[i:j]
                try:
                    num_val = float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")
                
                tokens.append(('NUMBER', num_val))
                i = j
            
            elif char == '+':
                tokens.append(('PLUS', None))
                i += 1
            
            elif char == '-':
                tokens.append(('MINUS', None))
                i += 1
            
            elif char == '*':
                tokens.append(('MUL', None))
                i += 1
            
            elif char == '/':
                tokens.append(('DIV', None))
                i += 1
            
            elif char == '(':
                tokens.append(('LPAREN', None))
                i += 1
            
            elif char == ')':
                tokens.append(('RPAREN', None))
                i += 1
            
            else:
                raise ValueError(f"Invalid character: {char}")
        
        tokens.append(('EOF', None))
        return tokens
    
    def _peek(self) -> str:
        """
        Look at current token type without consuming.
        
        Returns:
            The token type string
        """
        if self.pos < len(self.tokens):
            return self.tokens[self.pos][0]
        return 'EOF'
    
    def _consume(self) -> Tuple[str, Optional[float]]:
        """
        Consume current token and advance position.
        
        Returns:
            The token tuple (type, value)
            
        Raises:
            ValueError: If end of tokens reached unexpectedly
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        token = self.tokens[self.pos]
        self.pos += 1
        return token
    
    def _parse_expression(self) -> float:
        """
        Parse addition and subtraction (lowest precedence).
        
        Returns:
            The evaluated value
        """
        value = self._parse_term()
        
        while self._peek() in ('PLUS', 'MINUS'):
            op = self._consume()[0]
            right = self._parse_term()
            if op == 'PLUS':
                value += right
            else:
                value -= right
        
        return value
    
    def _parse_term(self) -> float:
        """
        Parse multiplication and division (higher precedence).
        
        Returns:
            The evaluated value
            
        Raises:
            ValueError: If division by zero occurs
        """
        value = self._parse_factor()
        
        while self._peek() in ('MUL', 'DIV'):
            op = self._consume()[0]
            right = self._parse_factor()
            if op == 'MUL':
                value *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                value /= right
        
        return value
    
    def _parse_factor(self) -> float:
        """
        Parse unary operators and primary expressions.
        
        Returns:
            The evaluated value
        """
        if self._peek() == 'MINUS':
            self._consume()
            return -self._parse_factor()
        elif self._peek() == 'PLUS':
            self._consume()
            return self._parse_factor()
        else:
            return self._parse_primary()
    
    def _parse_primary(self) -> float:
        """
        Parse numbers and parenthesized expressions.
        
        Returns:
            The evaluated value
            
        Raises:
            ValueError: If parentheses are mismatched or unexpected token found
        """
        if self._peek() == 'NUMBER':
            return self._consume()[1]
        elif self._peek() == 'LPAREN':
            self._consume()
            value = self._parse_expression()
            if self._peek() != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self._consume()
            return value
        else:
            raise ValueError(f"Unexpected token: {self._peek()}")


# Tests
import pytest

def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2+3") == 5.0
    assert ev.evaluate("10-4") == 6.0
    assert ev.evaluate("2*3") == 6.0
    assert ev.evaluate("10/2") == 5.0

def test_precedence():
    """Test operator precedence (* and / before + and -)."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2+3*4") == 14.0  # 2 + (3*4)
    assert ev.evaluate("2*3+4*5") == 26.0  # (2*3) + (4*5)
    assert ev.evaluate("10/2+3") == 8.0  # (10/2) + 3
    assert ev.evaluate("2+3*4-5") == 9.0  # 2 + 12 - 5

def test_parentheses():
    """Test parentheses for grouping."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2+3)*4") == 20.0
    assert ev.evaluate("2*(3+4)") == 14.0
    assert ev.evaluate("((2+3))") == 5.0
    assert ev.evaluate("(2+3)*(4-1)") == 15.0

def test_unary_minus():
    """Test unary minus operator."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("--3") == 3.0
    assert ev.evaluate("-(2+1)") == -3.0
    assert ev.evaluate("-2*-3") == 6.0
    assert ev.evaluate("2+-3") == -1.0
    assert ev.evaluate("2--3") == 5.0

def test_error_cases():
    """Test various error conditions."""
    ev = ExpressionEvaluator()
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("10/0")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched"):
        ev.evaluate("(2+3")
    
    # Invalid token
    with pytest.raises(ValueError, match="Invalid"):
        ev.evaluate("2@3")
    
    # Empty expression
    with pytest.raises(ValueError, match="Empty"):
        ev.evaluate("")
    
    # Invalid number format
    with pytest.raises(ValueError, match="Invalid"):
        ev.evaluate("3..5")