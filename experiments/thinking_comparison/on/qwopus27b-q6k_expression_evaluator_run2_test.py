from typing import List


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports: +, -, *, / with correct precedence, parentheses, unary minus,
    and floating point numbers.
    """
    
    def __init__(self) -> None:
        """Initialize the evaluator with empty token list and position."""
        self.tokens: List[str] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result as a float.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The result of evaluating the expression
            
        Raises:
            ValueError: If the expression is empty, has invalid tokens, 
                      mismatched parentheses, or division by zero
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        self._tokenize(expr)
        self.pos = 0
        
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if token == ')':
                raise ValueError("Mismatched parentheses: extra closing parenthesis")
            else:
                raise ValueError(f"Unexpected token '{token}' at position {self.pos}")
        
        return result
    
    def _tokenize(self, expr: str) -> None:
        """
        Convert expression string into a list of tokens.
        
        Args:
            expr: The expression string to tokenize
            
        Raises:
            ValueError: If an invalid character or malformed number is found
        """
        self.tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
            
            # Parse numbers (integers and floats)
            if char.isdigit() or char == '.':
                num_str = ""
                has_dot = False
                
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid token: multiple decimal points in number at position {i}")
                        has_dot = True
                    num_str += expr[i]
                    i += 1
                
                if num_str == '.':
                    raise ValueError(f"Invalid token: '.' is not a valid number at position {i-1}")
                
                self.tokens.append(num_str)
                continue
            
            # Parse operators and parentheses
            if char in '+-*/()':
                self.tokens.append(char)
                i += 1
                continue
            
            # Invalid character
            raise ValueError(f"Invalid character '{char}' at position {i}")
        
        if not self.tokens:
            raise ValueError("Empty expression")
    
    def _parse_expression(self) -> float:
        """
        Parse addition and subtraction operations (lowest precedence).
        
        Grammar: Expression -> Term (('+' | '-') Term)*
        
        Returns:
            The result of the parsed expression
        """
        value = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            
            if op == '+':
                value += right
            else:
                value -= right
        
        return value
    
    def _parse_term(self) -> float:
        """
        Parse multiplication and division operations (higher precedence).
        
        Grammar: Term -> Factor (('*' | '/') Factor)*
        
        Returns:
            The result of the parsed term
            
        Raises:
            ValueError: If division by zero is attempted
        """
        value = self._parse_factor()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()
            
            if op == '*':
                value *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                value /= right
        
        return value
    
    def _parse_factor(self) -> float:
        """
        Parse numbers, parentheses, and unary minus (highest precedence).
        
        Grammar: Factor -> Number | '(' Expression ')' | '-' Factor
        
        Returns:
            The result of the parsed factor
            
        Raises:
            ValueError: If parentheses are mismatched or unexpected end of expression
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        
        token = self.tokens[self.pos]
        
        if token == '(':
            self.pos += 1
            value = self._parse_expression()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing parenthesis")
            self.pos += 1
            return value
        
        elif token == ')':
            raise ValueError("Mismatched parentheses: unexpected closing parenthesis")
        
        elif token == '-':
            # Unary minus
            self.pos += 1
            return -self._parse_factor()
        
        elif token == '+':
            # Unary plus (optional but supported)
            self.pos += 1
            return self._parse_factor()
        
        else:
            # Should be a number
            try:
                value = float(token)
                self.pos += 1
                return value
            except ValueError:
                raise ValueError(f"Invalid token '{token}'")


# Test suite
import pytest


class TestExpressionEvaluator:
    """Test suite for ExpressionEvaluator."""
    
    def setup_method(self) -> None:
        """Set up a fresh evaluator for each test."""
        self.evaluator = ExpressionEvaluator()
    
    def test_basic_arithmetic(self) -> None:
        """Test basic arithmetic operations including floating point."""
        assert self.evaluator.evaluate("2+3") == 5.0
        assert self.evaluator.evaluate("10-4") == 6.0
        assert self.evaluator.evaluate("3*4") == 12.0
        assert self.evaluator.evaluate("12/3") == 4.0
        assert self.evaluator.evaluate("3.14+2.5") == 5.64
        assert abs(self.evaluator.evaluate("10/3") - 3.3333333333333335) < 1e-10
    
    def test_precedence(self) -> None:
        """Test operator precedence (* and / before + and -)."""
        assert self.evaluator.evaluate("2+3*4") == 14.0  # 2 + (3*4)
        assert self.evaluator.evaluate("2*3+4") == 10.0  # (2*3) + 4
        assert self.evaluator.evaluate("10-2*3") == 4.0  # 10 - (2*3)
        assert self.evaluator.evaluate("10/2+3") == 8.0  # (10/2) + 3
        assert self.evaluator.evaluate("2+3*4-5") == 9.0  # 2 + (3*4) - 5
    
    def test_parentheses(self) -> None:
        """Test parentheses for grouping and overriding precedence."""
        assert self.evaluator.evaluate("(2+3)*4") == 20.0
        assert self.evaluator.evaluate("2*(3+4)") == 14.0
        assert self.evaluator.evaluate("((2+3)*4)") == 20.0
        assert self.evaluator.evaluate("2+3*(4-1)") == 11.0
        assert self.evaluator.evaluate("((2+3)*(4-1))") == 15.0
    
    def test_unary_minus(self) -> None:
        """Test unary minus operator."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("--3") == 3.0
        assert self.evaluator.evaluate("-3*2") == -6.0
        assert self.evaluator.evaluate("3*-2") == -6.0
        assert self.evaluator.evaluate("-(2+1)") == -3.0
        assert self.evaluator.evaluate("-(-3)") == 3.0
        assert self.evaluator.evaluate("2-(-3)") == 5.0
        assert self.evaluator.evaluate("-3.14") == -3.14
    
    def test_error_cases(self) -> None:
        """Test error handling for invalid inputs."""
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("1/0")
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("(2+3")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("2+3)")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("((2+3)")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("2+3))")
        
        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("2+*3")
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("2+3a")
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("3..14")
        
        # Empty expressions
        with pytest.raises(ValueError, match="Empty"):
            self.evaluator.evaluate("")
        with pytest.raises(ValueError, match="Empty"):
            self.evaluator.evaluate("   ")