from typing import List

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus
    - Floating point numbers
    """
    
    def __init__(self):
        """Initialize the evaluator with empty token list and position."""
        self.tokens: List[str] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result as a float.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The result of the evaluation as a float
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors
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
        """
        Convert the expression string into a list of tokens.
        
        Args:
            expr: The expression string to tokenize
            
        Raises:
            ValueError: If an invalid character is encountered
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
                # Parse number
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format at position {j}: multiple decimal points")
                        has_dot = True
                    j += 1
                
                num_str = expr[i:j]
                
                # Validate number format
                if num_str == '.':
                    raise ValueError(f"Invalid number format at position {i}: '.'")
                if num_str.startswith('.') and len(num_str) == 1:
                    raise ValueError(f"Invalid number format at position {i}: '.'")
                if num_str.endswith('.') and len(num_str) > 1 and not num_str[-2].isdigit():
                    raise ValueError(f"Invalid number format at position {i}: '{num_str}'")
                    
                self.tokens.append(num_str)
                i = j
            elif char in '+-*/()':
                self.tokens.append(char)
                i += 1
            else:
                raise ValueError(f"Invalid character at position {i}: '{char}'")
    
    def _parse_expression(self) -> float:
        """
        Parse an expression (handles + and - operators).
        
        Returns:
            The evaluated result of the expression
        """
        left = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            
            if op == '+':
                left = left + right
            else:
                left = left - right
                
        return left
    
    def _parse_term(self) -> float:
        """
        Parse a term (handles * and / operators).
        
        Returns:
            The evaluated result of the term
        """
        left = self._parse_unary()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_unary()
            
            if op == '*':
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
                
        return left
    
    def _parse_unary(self) -> float:
        """
        Parse a unary expression (handles unary minus).
        
        Returns:
            The evaluated result of the unary expression
        """
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1
            return -self._parse_unary()
        return self._parse_factor()
    
    def _parse_factor(self) -> float:
        """
        Parse a factor (numbers, parentheses).
        
        Returns:
            The evaluated result of the factor
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
            
        token = self.tokens[self.pos]
        
        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses: expected ')'")
            self.pos += 1
            return result
        elif token == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")
        else:
            # Should be a number
            try:
                value = float(token)
                self.pos += 1
                return value
            except ValueError:
                raise ValueError(f"Invalid token: '{token}'")


# Test suite
import pytest

class TestExpressionEvaluator:
    def setup_method(self):
        self.evaluator = ExpressionEvaluator()
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert self.evaluator.evaluate("2 + 3") == 5.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("3 * 4") == 12.0
        assert self.evaluator.evaluate("15 / 3") == 5.0
        assert self.evaluator.evaluate("2.5 + 3.5") == 6.0
    
    def test_precedence(self):
        """Test operator precedence (* and / before + and -)."""
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + 12
        assert self.evaluator.evaluate("3 * 4 + 2") == 14.0  # 12 + 2
        assert self.evaluator.evaluate("10 - 2 * 3") == 4.0  # 10 - 6
        assert self.evaluator.evaluate("2 * 3 + 4 * 5") == 26.0  # 6 + 20
        assert self.evaluator.evaluate("10 / 2 + 3") == 8.0  # 5 + 3
    
    def test_parentheses(self):
        """Test parentheses for grouping."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert self.evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20.0
        assert self.evaluator.evaluate("10 / (2 + 3)") == 2.0
        assert self.evaluator.evaluate("(1 + 2) * (3 + 4)") == 21.0
    
    def test_unary_minus(self):
        """Test unary minus operator."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-3 + 5") == 2.0
        assert self.evaluator.evaluate("5 + -3") == 2.0
        assert self.evaluator.evaluate("-3 * 4") == -12.0
        assert self.evaluator.evaluate("3 * -4") == -12.0
        assert self.evaluator.evaluate("-3 * -4") == 12.0
        assert self.evaluator.evaluate("-(2 + 3)") == -5.0
        assert self.evaluator.evaluate("-(-3)") == 3.0
        assert self.evaluator.evaluate("--3") == 3.0
    
    def test_error_cases(self):
        """Test error handling."""
        # Empty expression
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate("")
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate("   ")
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched"):
            self.evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Mismatched"):
            self.evaluator.evaluate("2 + 3)")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("1 / 0")
        
        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("2 + a")
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("2 + 3.4.5")
        
        # Unexpected end of expression
        with pytest.raises(ValueError):
            self.evaluator.evaluate("2 +")