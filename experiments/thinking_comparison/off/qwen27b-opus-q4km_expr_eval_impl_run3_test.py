import re
from typing import List, Tuple, Iterator

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: A string containing the mathematical expression.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize the input string
        self._tokenize(expr)
        
        # Reset position for parsing
        self.pos = 0
        
        # Parse and evaluate
        result = self._parse_expression()
        
        # Ensure all tokens were consumed
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at end of expression")
            
        return result

    def _tokenize(self, expr: str) -> None:
        """
        Converts the input string into a list of tokens (numbers, operators, parentheses).
        """
        # Regex pattern to match numbers (int or float) or single characters
        pattern = r"(\d+\.?\d*|\.\d+|[+\-*/()])"
        matches = re.finditer(pattern, expr)
        
        self.tokens = []
        last_end = 0
        
        for match in matches:
            start, end = match.span()
            token = match.group(0)
            
            # Check for invalid characters between tokens
            if start > last_end:
                invalid_char = expr[last_end:start]
                raise ValueError(f"Invalid character '{invalid_char}' in expression")
            
            self.tokens.append(token)
            last_end = end
            
        # Check for trailing invalid characters
        if last_end < len(expr):
            raise ValueError(f"Invalid character '{expr[last_end]}' in expression")
            
        if not self.tokens:
            raise ValueError("No valid tokens found in expression")

    def _current_token(self) -> str:
        """Returns the current token or None if end of input."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self) -> str:
        """Consumes the current token and advances the position."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        token = self.tokens[self.pos]
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
            if token == '+':
                self._consume()
                value += self._parse_term()
            elif token == '-':
                self._consume()
                value -= self._parse_term()
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
            if token == '*':
                self._consume()
                value *= self._parse_factor()
            elif token == '/':
                self._consume()
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                value /= divisor
            else:
                break
                
        return value

    def _parse_factor(self) -> float:
        """
        Parses unary operators, numbers, and parenthesized expressions.
        Factor -> ('+' | '-') Factor | Number | '(' Expression ')'
        """
        token = self._current_token()
        
        # Handle unary plus/minus
        if token == '+':
            self._consume()
            return self._parse_factor()
        elif token == '-':
            self._consume()
            return -self._parse_factor()
        
        # Handle numbers
        if token and re.match(r'^\d+\.?\d*$', token) or re.match(r'^\.\d+$', token):
            self._consume()
            return float(token)
            
        # Handle parentheses
        if token == '(':
            self._consume()
            value = self._parse_expression()
            closing = self._current_token()
            if closing != ')':
                raise ValueError("Mismatched parentheses: expected ')'")
            self._consume()
            return value
            
        # If we reach here, it's an invalid token
        if token:
            raise ValueError(f"Invalid token '{token}'")
        else:
            raise ValueError("Unexpected end of expression")

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