from typing import List, Union, Tuple

class Token:
    """Represents a token in the expression."""
    def __init__(self, type_: str, value: Union[str, float]):
        self.type = type_
        self.value = value
    
    def __repr__(self):
        return f"Token({self.type}, {self.value})"

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    Supports +, -, *, / operators with correct precedence, parentheses,
    unary minus, and floating point numbers.
    """
    
    def __init__(self):
        """Initialize the expression evaluator."""
        pass
    
    def _tokenize(self, expr: str) -> List[Token]:
        """
        Convert expression string into list of tokens.
        
        Args:
            expr: The mathematical expression string
            
        Returns:
            List of Token objects
            
        Raises:
            ValueError: If expression contains invalid characters
        """
        tokens = []
        i = 0
        expr = expr.replace(' ', '')  # Remove spaces
        
        while i < len(expr):
            char = expr[i]
            
            if char.isdigit() or char == '.':
                # Parse number (integer or float)
                num_str = ''
                has_dot = False
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format at position {i}")
                        has_dot = True
                    num_str += expr[i]
                    i += 1
                tokens.append(Token('NUMBER', float(num_str)))
                continue
            
            elif char in '+-*/()':
                tokens.append(Token(char, char))
                i += 1
            
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")
        
        return tokens
    
    def _parse_expression(self, tokens: List[Token], pos: int) -> Tuple[float, int]:
        """
        Parse addition/subtraction level (lowest precedence).
        
        Args:
            tokens: List of tokens to parse
            pos: Current position in token list
            
        Returns:
            Tuple of (parsed value, new position)
        """
        value, pos = self._parse_term(tokens, pos)
        
        while pos < len(tokens) and tokens[pos].type in ('+', '-'):
            operator = tokens[pos].type
            pos += 1
            right_value, pos = self._parse_term(tokens, pos)
            
            if operator == '+':
                value += right_value
            else:
                value -= right_value
        
        return value, pos
    
    def _parse_term(self, tokens: List[Token], pos: int) -> Tuple[float, int]:
        """
        Parse multiplication/division level (higher precedence than +,-).
        
        Args:
            tokens: List of tokens to parse
            pos: Current position in token list
            
        Returns:
            Tuple of (parsed value, new position)
        """
        value, pos = self._parse_factor(tokens, pos)
        
        while pos < len(tokens) and tokens[pos].type in ('*', '/'):
            operator = tokens[pos].type
            pos += 1
            right_value, pos = self._parse_factor(tokens, pos)
            
            if operator == '*':
                value *= right_value
            else:
                if right_value == 0:
                    raise ValueError("Division by zero")
                value /= right_value
        
        return value, pos
    
    def _parse_factor(self, tokens: List[Token], pos: int) -> Tuple[float, int]:
        """
        Parse numbers, unary minus, and parenthesized expressions.
        
        Args:
            tokens: List of tokens to parse
            pos: Current position in token list
            
        Returns:
            Tuple of (parsed value, new position)
        """
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")
        
        token = tokens[pos]
        
        # Handle unary minus
        if token.type == '-':
            pos += 1
            value, pos = self._parse_factor(tokens, pos)
            return -value, pos
        
        # Handle numbers
        elif token.type == 'NUMBER':
            pos += 1
            return token.value, pos
        
        # Handle parenthesized expressions
        elif token.type == '(':
            pos += 1
            value, pos = self._parse_expression(tokens, pos)
            
            if pos >= len(tokens) or tokens[pos].type != ')':
                raise ValueError("Missing closing parenthesis")
            
            pos += 1
            return value, pos
        
        else:
            raise ValueError(f"Unexpected token: {token}")
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Args:
            expr: Mathematical expression string
            
        Returns:
            The evaluated result as float
            
        Raises:
            ValueError: For invalid expressions, division by zero, etc.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        tokens = self._tokenize(expr)
        
        if not tokens:
            raise ValueError("Empty expression")
        
        result, pos = self._parse_expression(tokens, 0)
        
        if pos != len(tokens):
            raise ValueError(f"Unexpected token at position {pos}: {tokens[pos]}")
        
        return result


# Test suite
import pytest

def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0

def test_operator_precedence():
    """Test that multiplication/division have higher precedence than addition/subtraction."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + (3 * 4) = 14
    assert evaluator.evaluate("10 - 2 * 3") == 4.0   # 10 - (2 * 3) = 4
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0  # (2 * 3) + (4 * 5) = 26

def test_parentheses():
    """Test parentheses for grouping and overriding precedence."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 4) / 2") == 10.0

def test_unary_minus():
    """Test unary minus operator."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("-2 * 3") == -6.0
    assert evaluator.evaluate("-(-5)") == 5.0

def test_error_cases():
    """Test various error conditions."""
    evaluator = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    
    with pytest.raises(ValueError, match="Unexpected"):
        evaluator.evaluate("2 +")
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("2 / 0")
    
    with pytest.raises(ValueError, match="Missing closing parenthesis"):
        evaluator.evaluate("(2 + 3")
    
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + a")