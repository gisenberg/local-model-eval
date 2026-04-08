import re
from typing import List, Tuple, Optional

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers
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
    
    def __init__(self):
        """Initialize the evaluator with an empty token list."""
        self._tokens: List[Tuple[str, str]] = []
        self._pos: int = 0
    
    def _tokenize(self, expr: str) -> List[Tuple[str, str]]:
        """
        Convert the input string into a list of tokens.
        
        Args:
            expr: The input expression string
            
        Returns:
            A list of (token_type, token_value) tuples
            
        Raises:
            ValueError: If an invalid character is encountered
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        # Pattern to match numbers (integers and floats) and operators/parentheses
        pattern = r'(\d+\.?\d*|\.\d+|[+\-*/()])'
        tokens: List[Tuple[str, str]] = []
        
        for match in re.finditer(pattern, expr):
            char = match.group(1)
            
            # Check for invalid characters between tokens
            start = match.start()
            if start > 0 and expr[start - 1] not in ' \t':
                raise ValueError(f"Invalid character in expression: '{expr[start - 1]}'")
            
            if char == '+':
                tokens.append((self.TOKEN_PLUS, '+'))
            elif char == '-':
                tokens.append((self.TOKEN_MINUS, '-'))
            elif char == '*':
                tokens.append((self.TOKEN_MULTIPLY, '*'))
            elif char == '/':
                tokens.append((self.TOKEN_DIVIDE, '/'))
            elif char == '(':
                tokens.append((self.TOKEN_LPAREN, '('))
            elif char == ')':
                tokens.append((self.TOKEN_RPAREN, ')'))
            else:
                # It's a number
                tokens.append((self.TOKEN_NUMBER, char))
        
        # Check for invalid characters at the end
        last_match = re.finditer(pattern, expr)
        last_match = list(last_match)
        if last_match:
            end_pos = last_match[-1].end()
            if end_pos < len(expr):
                for i in range(end_pos, len(expr)):
                    if expr[i] not in ' \t':
                        raise ValueError(f"Invalid character in expression: '{expr[i]}'")
        
        if not tokens:
            raise ValueError("Empty expression")
        
        return tokens
    
    def _current_token(self) -> Tuple[str, str]:
        """
        Return the current token at the parser position.
        
        Returns:
            A (token_type, token_value) tuple, or (TOKEN_EOF, '') at end
        """
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return (self.TOKEN_EOF, '')
    
    def _peek_token(self) -> Tuple[str, str]:
        """
        Return the next token without advancing (same as current_token).
        
        Returns:
            A (token_type, token_value) tuple
        """
        return self._current_token()
    
    def _advance(self) -> Tuple[str, str]:
        """
        Advance to the next token and return the current one.
        
        Returns:
            The token that was at the current position
        """
        token = self._current_token()
        if token[0] != self.TOKEN_EOF:
            self._pos += 1
        return token
    
    def _expect(self, token_type: str) -> Tuple[str, str]:
        """
        Expect a specific token type at the current position.
        
        Args:
            token_type: The expected token type
            
        Returns:
            The matched token
            
        Raises:
            ValueError: If the expected token is not found
        """
        token = self._current_token()
        if token[0] != token_type:
            if token[0] == self.TOKEN_EOF:
                raise ValueError(f"Unexpected end of expression, expected '{token_type}'")
            raise ValueError(f"Expected '{token_type}', got '{token[0]}'")
        return self._advance()
    
    def _parse_expression(self) -> float:
        """
        Parse an expression (handles + and - operators).
        
        Grammar: Expression → Term (('+' | '-') Term)*
        
        Returns:
            The evaluated result of the expression
        """
        result = self._parse_term()
        
        while True:
            token = self._current_token()
            if token[0] == self.TOKEN_PLUS:
                self._advance()
                right = self._parse_term()
                result = result + right
            elif token[0] == self.TOKEN_MINUS:
                self._advance()
                right = self._parse_term()
                result = result - right
            else:
                break
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse a term (handles * and / operators).
        
        Grammar: Term → Factor (('*' | '/') Factor)*
        
        Returns:
            The evaluated result of the term
        """
        result = self._parse_factor()
        
        while True:
            token = self._current_token()
            if token[0] == self.TOKEN_MULTIPLY:
                self._advance()
                right = self._parse_factor()
                result = result * right
            elif token[0] == self.TOKEN_DIVIDE:
                self._advance()
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero")
                result = result / right
            else:
                break
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parse a factor (numbers, parentheses, unary minus).
        
        Grammar: Factor → Number | '(' Expression ')' | '-' Factor
        
        Returns:
            The evaluated result of the factor
        """
        token = self._current_token()
        
        # Handle unary minus
        if token[0] == self.TOKEN_MINUS:
            self._advance()
            return -self._parse_factor()
        
        # Handle unary plus (optional, for expressions like '+3')
        if token[0] == self.TOKEN_PLUS:
            self._advance()
            return self._parse_factor()
        
        # Handle numbers
        if token[0] == self.TOKEN_NUMBER:
            self._advance()
            return float(token[1])
        
        # Handle parenthesized expressions
        if token[0] == self.TOKEN_LPAREN:
            self._advance()
            result = self._parse_expression()
            self._expect(self.TOKEN_RPAREN)
            return result
        
        # If we get here, something is wrong
        if token[0] == self.TOKEN_EOF:
            raise ValueError("Unexpected end of expression")
        
        raise ValueError(f"Unexpected token: '{token[1]}'")
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Supports:
        - Basic arithmetic: +, -, *, / with correct precedence
        - Parentheses for grouping
        - Unary minus (e.g., '-3', '-(2+1)')
        - Floating point numbers (e.g., '3.14')
        
        Args:
            expr: The mathematical expression to evaluate
            
        Returns:
            The result of the evaluation as a float
            
        Raises:
            ValueError: For invalid expressions, mismatched parentheses,
                       division by zero, or empty expressions
        """
        # Tokenize the expression
        self._tokens = self._tokenize(expr)
        self._pos = 0
        
        # Parse and evaluate
        result = self._parse_expression()
        
        # Check that we consumed all tokens
        if self._current_token()[0] != self.TOKEN_EOF:
            token = self._current_token()
            raise ValueError(f"Unexpected token: '{token[1]}'")
        
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