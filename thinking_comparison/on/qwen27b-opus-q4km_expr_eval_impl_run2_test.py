import re
from typing import List, Tuple, Optional

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, / operators with correct precedence, parentheses,
    unary minus, and floating point numbers.
    """
    
    # Token types
    TOKEN_NUMBER = 'NUMBER'
    TOKEN_PLUS = 'PLUS'
    TOKEN_MINUS = 'MINUS'
    TOKEN_MULTIPLY = 'MULTIPLY'
    TOKEN_DIVIDE = 'DIVIDE'
    TOKEN_LPAREN = 'LPAREN'
    TOKEN_RPAREN = 'RPAREN'
    TOKEN_EOF = 'EOF'
    
    def __init__(self) -> None:
        """Initialize the ExpressionEvaluator."""
        self._tokens: List[Tuple[str, str]] = []
        self._pos: int = 0
    
    def _tokenize(self, expr: str) -> List[Tuple[str, str]]:
        """
        Convert the input expression string into a list of tokens.
        
        Args:
            expr: The input expression string
            
        Returns:
            List of (token_type, token_value) tuples
            
        Raises:
            ValueError: If an invalid character is encountered
        """
        tokens: List[Tuple[str, str]] = []
        pattern = r'(\d+\.?\d*|\.\d+|[+\-*/()])'
        matches = re.finditer(pattern, expr)
        
        prev_end = 0
        for match in matches:
            start, end = match.span()
            token_text = match.group(1)
            
            # Check for invalid characters between tokens
            if start > prev_end:
                invalid_char = expr[prev_end:start]
                raise ValueError(f"Invalid character '{invalid_char}' in expression")
            
            # Classify the token
            if token_text == '+':
                tokens.append((self.TOKEN_PLUS, token_text))
            elif token_text == '-':
                tokens.append((self.TOKEN_MINUS, token_text))
            elif token_text == '*':
                tokens.append((self.TOKEN_MULTIPLY, token_text))
            elif token_text == '/':
                tokens.append((self.TOKEN_DIVIDE, token_text))
            elif token_text == '(':
                tokens.append((self.TOKEN_LPAREN, token_text))
            elif token_text == ')':
                tokens.append((self.TOKEN_RPAREN, token_text))
            else:
                # It's a number
                tokens.append((self.TOKEN_NUMBER, token_text))
            
            prev_end = end
        
        # Check for trailing invalid characters
        if prev_end < len(expr):
            invalid_char = expr[prev_end:]
            raise ValueError(f"Invalid character '{invalid_char}' in expression")
        
        # Add EOF token
        tokens.append((self.TOKEN_EOF, ''))
        return tokens
    
    def _current_token(self) -> Tuple[str, str]:
        """
        Get the current token at the parser position.
        
        Returns:
            The current (token_type, token_value) tuple
        """
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return (self.TOKEN_EOF, '')
    
    def _consume(self, expected_type: Optional[str] = None) -> Tuple[str, str]:
        """
        Consume and return the current token, advancing the position.
        
        Args:
            expected_type: Optional expected token type for validation
            
        Returns:
            The consumed (token_type, token_value) tuple
            
        Raises:
            ValueError: If the token type doesn't match expected_type
        """
        token = self._current_token()
        
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[0]}")
        
        self._pos += 1
        return token
    
    def _parse_expression(self) -> float:
        """
        Parse an expression: Term (('+' | '-') Term)*
        Handles addition and subtraction with left associativity.
        
        Returns:
            The evaluated result of the expression
        """
        result = self._parse_term()
        
        while True:
            token = self._current_token()
            if token[0] == self.TOKEN_PLUS:
                self._consume()
                result = result + self._parse_term()
            elif token[0] == self.TOKEN_MINUS:
                self._consume()
                result = result - self._parse_term()
            else:
                break
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse a term: Factor (('*' | '/') Factor)*
        Handles multiplication and division with left associativity.
        
        Returns:
            The evaluated result of the term
        """
        result = self._parse_factor()
        
        while True:
            token = self._current_token()
            if token[0] == self.TOKEN_MULTIPLY:
                self._consume()
                result = result * self._parse_factor()
            elif token[0] == self.TOKEN_DIVIDE:
                self._consume()
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                result = result / divisor
            else:
                break
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parse a factor: Number | '(' Expression ')' | '-' Factor
        Handles numbers, parenthesized expressions, and unary minus.
        
        Returns:
            The evaluated result of the factor
        """
        token = self._current_token()
        
        # Handle unary minus
        if token[0] == self.TOKEN_MINUS:
            self._consume()
            return -self._parse_factor()
        
        # Handle unary plus (optional, for consistency)
        if token[0] == self.TOKEN_PLUS:
            self._consume()
            return self._parse_factor()
        
        # Handle parenthesized expression
        if token[0] == self.TOKEN_LPAREN:
            self._consume()
            result = self._parse_expression()
            
            # Expect closing parenthesis
            if self._current_token()[0] != self.TOKEN_RPAREN:
                raise ValueError("Expected closing parenthesis ')'")
            self._consume()
            return result
        
        # Handle number
        if token[0] == self.TOKEN_NUMBER:
            self._consume()
            return float(token[1])
        
        # Unexpected token
        raise ValueError(f"Unexpected token: {token[0]}")
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Supports:
        - Operators: +, -, *, / with correct precedence
        - Parentheses for grouping
        - Unary minus (e.g., '-3', '-(2+1)')
        - Floating point numbers (e.g., '3.14')
        
        Args:
            expr: The mathematical expression string to evaluate
            
        Returns:
            The result of evaluating the expression as a float
            
        Raises:
            ValueError: For invalid expressions, mismatched parentheses,
                       division by zero, or empty expressions
        """
        # Check for empty or whitespace-only expression
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        # Tokenize the expression
        self._tokens = self._tokenize(expr)
        self._pos = 0
        
        # Check if expression is empty after tokenization
        if len(self._tokens) == 1 and self._tokens[0][0] == self.TOKEN_EOF:
            raise ValueError("Empty expression")
        
        # Parse and evaluate
        result = self._parse_expression()
        
        # Verify we consumed all tokens
        if self._current_token()[0] != self.TOKEN_EOF:
            token = self._current_token()
            raise ValueError(f"Unexpected token '{token[1]}' at end of expression")
        
        return result

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("15 / 4") == 3.75

def test_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0

def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_errors(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3")