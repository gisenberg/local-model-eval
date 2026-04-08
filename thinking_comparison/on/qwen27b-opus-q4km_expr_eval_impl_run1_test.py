import re
from typing import List, Tuple, Any

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        # Token types
        self.NUMBER = 'NUMBER'
        self.PLUS = 'PLUS'
        self.MINUS = 'MINUS'
        self.STAR = 'STAR'
        self.SLASH = 'SLASH'
        self.LPAREN = 'LPAREN'
        self.RPAREN = 'RPAREN'
        self.EOF = 'EOF'

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty or whitespace only")
            
        # Tokenize the input
        tokens = self._tokenize(expr)
        
        if not tokens:
            raise ValueError("Expression cannot be empty")
            
        # Initialize parser state
        self.pos = 0
        self.tokens = tokens
        
        # Parse and evaluate
        result = self._parse_expression()
        
        # Ensure all tokens were consumed
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            raise ValueError(f"Unexpected token '{token[1]}' at end of expression")
            
        return result

    def _tokenize(self, text: str) -> List[Tuple[str, Any]]:
        """
        Converts the input string into a list of tokens.
        
        Args:
            text: The input expression string.
            
        Returns:
            A list of tuples containing (token_type, token_value).
        """
        tokens = []
        # Regex pattern for numbers (integers and floats)
        number_pattern = r'\d+(\.\d+)?'
        # Pattern for everything else (operators, parens, whitespace)
        token_pattern = rf'({number_pattern})|([+\-*/()])|(\s+)|(.)'
        
        for match in re.finditer(token_pattern, text):
            if match.group(1):  # Number
                tokens.append((self.NUMBER, float(match.group(1))))
            elif match.group(3):  # Operator or Parenthesis
                char = match.group(3)
                if char == '+': tokens.append((self.PLUS, '+'))
                elif char == '-': tokens.append((self.MINUS, '-'))
                elif char == '*': tokens.append((self.STAR, '*'))
                elif char == '/': tokens.append((self.SLASH, '/'))
                elif char == '(': tokens.append((self.LPAREN, '('))
                elif char == ')': tokens.append((self.RPAREN, ')'))
                else:
                    raise ValueError(f"Invalid character '{char}' in expression")
            elif match.group(4):  # Invalid character (not whitespace, not number, not operator)
                raise ValueError(f"Invalid character '{match.group(4)}' in expression")
            # Whitespace is ignored
            
        tokens.append((self.EOF, None))
        return tokens

    def _current_token(self) -> Tuple[str, Any]:
        """Returns the current token at the parser position."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (self.EOF, None)

    def _consume(self, expected_type: str = None) -> Tuple[str, Any]:
        """
        Consumes the current token and advances the position.
        
        Args:
            expected_type: Optional token type to validate against.
            
        Returns:
            The consumed token.
            
        Raises:
            ValueError: If the expected token type does not match.
        """
        token = self._current_token()
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected token type {expected_type}, got {token[0]}")
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Expression -> Term (('+' | '-') Term)*
        """
        value = self._parse_term()
        
        while True:
            token = self._current_token()
            if token[0] == self.PLUS:
                self._consume(self.PLUS)
                value = value + self._parse_term()
            elif token[0] == self.MINUS:
                self._consume(self.MINUS)
                value = value - self._parse_term()
            else:
                break
        return value

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Term -> Factor (('*' | '/') Factor)*
        """
        value = self._parse_factor()
        
        while True:
            token = self._current_token()
            if token[0] == self.STAR:
                self._consume(self.STAR)
                value = value * self._parse_factor()
            elif token[0] == self.SLASH:
                self._consume(self.SLASH)
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                value = value / divisor
            else:
                break
        return value

    def _parse_factor(self) -> float:
        """
        Parses numbers, parentheses, and unary operators (highest precedence).
        Factor -> Number | '(' Expression ')' | ('+' | '-') Factor
        """
        token = self._current_token()
        
        # Handle unary plus/minus
        if token[0] == self.PLUS:
            self._consume(self.PLUS)
            return self._parse_factor()
        elif token[0] == self.MINUS:
            self._consume(self.MINUS)
            return -self._parse_factor()
        
        # Handle numbers
        if token[0] == self.NUMBER:
            self._consume(self.NUMBER)
            return token[1]
        
        # Handle parentheses
        if token[0] == self.LPAREN:
            self._consume(self.LPAREN)
            value = self._parse_expression()
            if self._current_token()[0] != self.RPAREN:
                raise ValueError("Mismatched parentheses: missing closing parenthesis")
            self._consume(self.RPAREN)
            return value
        
        # If we reach here, it's an unexpected token
        if token[0] == self.EOF:
            raise ValueError("Unexpected end of expression")
        raise ValueError(f"Unexpected token '{token[1]}'")

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