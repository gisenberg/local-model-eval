from typing import List, Union, Optional
from dataclasses import dataclass


@dataclass
class Token:
    """Represents a lexical token in the expression."""
    type: str
    value: Union[float, str]


class ExpressionEvaluator:
    """
    A recursive descent parser for mathematical expressions.
    
    Supports +, -, *, / with correct operator precedence, parentheses, 
    unary minus, and floating point numbers.
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The result of the evaluation as a float
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Empty expression")
            
        result = self._parse_expr()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token after expression: {self.tokens[self.pos]}")
            
        return result
    
    def _tokenize(self, expr: str) -> List[Token]:
        """
        Convert expression string into a list of tokens.
        
        Args:
            expr: The expression string to tokenize
            
        Returns:
            List of Token objects
            
        Raises:
            ValueError: If invalid characters or number formats are found
        """
        tokens: List[Token] = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
            
            elif char.isdigit() or char == '.':
                # Parse number (integer or float)
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
                if num_str == '.' or (num_str.startswith('.') and len(num_str) == 1):
                    raise ValueError(f"Invalid number format at position {i}")
                
                tokens.append(Token('NUMBER', float(num_str)))
                i = j
            
            elif char in '+-*/':
                tokens.append(Token('OP', char))
                i += 1
            
            elif char == '(':
                tokens.append(Token('LPAREN', '('))
                i += 1
            
            elif char == ')':
                tokens.append(Token('RPAREN', ')'))
                i += 1
            
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")
        
        return tokens
    
    def _current_token(self) -> Token:
        """Return the current token without consuming it."""
        if self.pos >= len(self.tokens):
            return Token('EOF', '')
        return self.tokens[self.pos]
    
    def _consume(self, expected_type: Optional[str] = None) -> Token:
        """
        Consume and return the current token.
        
        Args:
            expected_type: If provided, verify the token type matches
            
        Returns:
            The consumed token
            
        Raises:
            ValueError: If expected_type is provided and doesn't match, or if EOF reached
        """
        token = self._current_token()
        if token.type == 'EOF':
            raise ValueError("Unexpected end of expression")
        
        if expected_type and token.type != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token.type}")
        
        self.pos += 1
        return token
    
    def _parse_expr(self) -> float:
        """
        Parse addition and subtraction (lowest precedence).
        Grammar: expr := term (('+' | '-') term)*
        """
        value = self._parse_term()
        
        while self._current_token().type == 'OP' and self._current_token().value in '+-':
            op = self._consume('OP').value
            right = self._parse_term()
            if op == '+':
                value += right
            else:
                value -= right
        
        return value
    
    def _parse_term(self) -> float:
        """
        Parse multiplication and division (higher precedence).
        Grammar: term := unary (('*' | '/') unary)*
        """
        value = self._parse_unary()
        
        while self._current_token().type == 'OP' and self._current_token().value in '*/':
            op = self._consume('OP').value
            right = self._parse_unary()
            if op == '*':
                value *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                value /= right
        
        return value
    
    def _parse_unary(self) -> float:
        """
        Parse unary minus and plus.
        Grammar: unary := ('-' | '+') unary | factor
        """
        if self._current_token().type == 'OP' and self._current_token().value in '-+':
            op = self._consume('OP').value
            operand = self._parse_unary()
            if op == '-':
                return -operand
            else:
                return operand
        return self._parse_factor()
    
    def _parse_factor(self) -> float:
        """
        Parse numbers and parenthesized expressions.
        Grammar: factor := number | '(' expr ')'
        """
        token = self._current_token()
        
        if token.type == 'NUMBER':
            self._consume('NUMBER')
            return token.value
        
        elif token.type == 'LPAREN':
            self._consume('LPAREN')
            value = self._parse_expr()
            
            if self._current_token().type != 'RPAREN':
                raise ValueError("Mismatched parentheses: expected ')'")
            
            self._consume('RPAREN')
            return value
        
        else:
            raise ValueError(f"Unexpected token: {token}")


# Pytest tests
import pytest


class TestExpressionEvaluator:
    def setup_method(self):
        self.evaluator = ExpressionEvaluator()
    
    def test_basic_arithmetic(self):
        """Test basic +, -, *, / operations including floats."""
        assert self.evaluator.evaluate("2 + 3") == 5.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("3 * 4") == 12.0
        assert self.evaluator.evaluate("15 / 3") == 5.0
        assert self.evaluator.evaluate("2.5 + 3.5") == 6.0
        assert self.evaluator.evaluate("3.14 * 2") == 6.28
        assert self.evaluator.evaluate(".5 + .5") == 1.0
    
    def test_precedence(self):
        """Test operator precedence: * and / before + and -."""
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
        assert self.evaluator.evaluate("2 * 3 + 4") == 10.0
        assert self.evaluator.evaluate("10 / 2 + 3") == 8.0
        assert self.evaluator.evaluate("2 + 3 * 4 - 5") == 9.0
        assert self.evaluator.evaluate("10 - 2 * 3 + 4 / 2") == 4.0
    
    def test_parentheses(self):
        """Test parentheses for grouping."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert self.evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20.0
        assert self.evaluator.evaluate("(1 + 2) * (3 + 4)") == 21.0
        assert self.evaluator.evaluate("2 + (3 * (4 + 5))") == 29.0
    
    def test_unary_minus(self):
        """Test unary minus operator."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-(2 + 1)") == -3.0
        assert self.evaluator.evaluate("3 - -4") == 7.0
        assert self.evaluator.evaluate("--5") == 5.0
        assert self.evaluator.evaluate("-2 * 3") == -6.0
        assert self.evaluator.evaluate("2 * -3") == -6.0
        assert self.evaluator.evaluate("-(-5)") == 5.0
    
    def test_error_cases(self):
        """Test error handling."""
        # Empty expression
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("")
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("   ")
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("2 + 3)")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("5 / 0")
        
        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 + a")
        with pytest.raises(ValueError, match="Invalid number format"):
            self.evaluator.evaluate("2 + ..3")