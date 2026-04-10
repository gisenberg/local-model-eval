from typing import List, Tuple, Union

class TokenizerError(Exception):
    """Custom exception for tokenization errors."""
    pass

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus
    - Floating point numbers
    
    Raises:
        ValueError: For mismatched parentheses, division by zero, invalid tokens, or empty expressions.
    """

    def __init__(self):
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens, 
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression provided.")
        
        try:
            self._tokens = self._tokenize(expr)
        except TokenizerError as e:
            raise ValueError(f"Invalid token: {e}") from e

        if not self._tokens:
            raise ValueError("Empty expression provided.")

        self._pos = 0
        result = self._parse_expression()
        
        if self._pos != len(self._tokens):
            raise ValueError(f"Unexpected token '{self._tokens[self._pos]}' at the end of expression.")
            
        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of tokens (numbers, operators, parentheses).
        
        Args:
            expr: The raw expression string.
            
        Returns:
            A list of token strings.
            
        Raises:
            TokenizerError: If an invalid character is encountered.
        """
        tokens = []
        i = 0
        length = len(expr)
        
        while i < length:
            char = expr[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
            
            # Handle numbers (including floats)
            if char.isdigit() or (char == '.' and i + 1 < length and expr[i+1].isdigit()):
                start = i
                while i < length and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                tokens.append(expr[start:i])
                continue
            
            # Handle operators and parentheses
            if char in '+-*/()':
                tokens.append(char)
                i += 1
                continue
            
            # Invalid character
            raise TokenizerError(f"Invalid character '{char}'")
            
        return tokens

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Grammar: expression -> term (('+' | '-') term)*
        
        Returns:
            The result of the addition/subtraction chain.
        """
        left = self._parse_term()
        
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ('+', '-'):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_term()
            
            if op == '+':
                left = left + right
            else:
                left = left - right
                
        return left

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Grammar: term -> factor (('*' | '/') factor)*
        
        Returns:
            The result of the multiplication/division chain.
        """
        left = self._parse_factor()
        
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ('*', '/'):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_factor()
            
            if op == '*':
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero.")
                left = left / right
                
        return left

    def _parse_factor(self) -> float:
        """
        Parses numbers, parentheses, and unary operators.
        Grammar: factor -> NUMBER | '(' expression ')' | '-' factor | '+' factor
        
        Returns:
            The parsed numeric value.
        """
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression.")
            
        token = self._tokens[self._pos]
        
        # Handle unary minus and plus
        if token == '-':
            self._pos += 1
            return -self._parse_factor()
        elif token == '+':
            self._pos += 1
            return self._parse_factor()
        
        # Handle parentheses
        if token == '(':
            self._pos += 1
            result = self._parse_expression()
            
            if self._pos >= len(self._tokens) or self._tokens[self._pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'.")
            
            self._pos += 1
            return result
        
        # Handle numbers
        if token.replace('.', '', 1).isdigit() and token.count('.') < 2:
            self._pos += 1
            return float(token)
        
        raise ValueError(f"Invalid token: '{token}'")

# --- Pytest Tests ---
import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    """Test basic addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("20 / 4") == 5.0
    assert evaluator.evaluate("1.5 + 2.5") == 4.0

def test_operator_precedence(evaluator):
    """Test that multiplication/division happens before addition/subtraction."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0

def test_parentheses(evaluator):
    """Test grouping with parentheses."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 - (2 * 3)") == 4.0
    assert evaluator.evaluate("((1 + 2) * 3) - 4") == 5.0
    assert evaluator.evaluate("1 + (2 + (3 + 4))") == 10.0

def test_unary_minus(evaluator):
    """Test unary minus and nested unary operators."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("- (2 + 1)") == -3.0
    assert evaluator.evaluate("5 + -3") == 2.0
    assert evaluator.evaluate("-(-5)") == 5.0
    assert evaluator.evaluate("-(2+3)") == -5.0

def test_error_cases(evaluator):
    """Test various error conditions."""
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
        
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
        
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + x")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")