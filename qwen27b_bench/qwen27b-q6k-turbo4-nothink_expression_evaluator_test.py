import re
from typing import List, Tuple, Union

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
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: If the expression is invalid, empty, has mismatched parentheses,
                        or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")

        self._tokenize(expr)
        
        if not self.tokens:
            raise ValueError("Expression contains no valid tokens.")

        self.pos = 0
        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at end of expression.")

        return result

    def _tokenize(self, expr: str) -> None:
        """
        Converts the input string into a list of tokens (numbers, operators, parentheses).
        """
        # Regex to match numbers (int or float), operators, parentheses, or whitespace
        token_pattern = r'(\d+\.?\d*|\.\d+|[+\-*/()])'
        tokens = re.findall(token_pattern, expr)
        
        # Validate that the entire string was consumed by the regex (no invalid chars)
        # We reconstruct the string from tokens (ignoring whitespace) to check for leftovers
        # A simpler check: if the regex didn't match everything non-whitespace, it's invalid.
        cleaned_expr = re.sub(r'\s+', '', expr)
        reconstructed = ''.join(tokens)
        
        if cleaned_expr != reconstructed:
            # Find the first invalid character
            for i, char in enumerate(cleaned_expr):
                if char not in '+-*/().0123456789':
                    raise ValueError(f"Invalid character '{char}' at position {i}")
            raise ValueError("Invalid token found in expression.")

        self.tokens = tokens

    def _current_token(self) -> Union[str, None]:
        """Returns the current token or None if end of list."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self, expected: str = None) -> str:
        """
        Consumes the current token. If expected is provided, validates the token.
        """
        token = self._current_token()
        if token is None:
            raise ValueError("Unexpected end of expression.")
        
        if expected and token != expected:
            raise ValueError(f"Expected '{expected}', got '{token}'.")
            
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Grammar: expression = term (('+' | '-') term)*
        """
        value = self._parse_term()

        while True:
            token = self._current_token()
            if token == '+':
                self._consume('+')
                value += self._parse_term()
            elif token == '-':
                self._consume('-')
                value -= self._parse_term()
            else:
                break
        return value

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Grammar: term = factor (('*' | '/') factor)*
        """
        value = self._parse_factor()

        while True:
            token = self._current_token()
            if token == '*':
                self._consume('*')
                right = self._parse_factor()
                value *= right
            elif token == '/':
                self._consume('/')
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero.")
                value /= right
            else:
                break
        return value

    def _parse_factor(self) -> float:
        """
        Parses numbers, parentheses, and unary minus.
        Grammar: factor = ('-' | '+')? (number | '(' expression ')')
        """
        token = self._current_token()

        # Handle unary minus or plus
        if token == '-':
            self._consume('-')
            return -self._parse_factor()
        elif token == '+':
            self._consume('+')
            return self._parse_factor()

        # Handle parentheses
        if token == '(':
            self._consume('(')
            value = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses: expected ')'.")
            self._consume(')')
            return value

        # Handle numbers
        if token and re.match(r'^\d+\.?\d*$', token) or re.match(r'^\.\d+$', token):
            self._consume()
            return float(token)

        # If we reach here, it's an unexpected token
        raise ValueError(f"Unexpected token '{token}' in factor.")

import pytest


class TestExpressionEvaluator:
    def setup_method(self):
        """Initialize a fresh evaluator for each test."""
        self.evaluator = ExpressionEvaluator()

    def test_basic_arithmetic(self):
        """Test basic addition, subtraction, multiplication, and division."""
        assert self.evaluator.evaluate("2 + 2") == 4.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("3 * 5") == 15.0
        assert self.evaluator.evaluate("20 / 4") == 5.0
        assert self.evaluator.evaluate("1.5 + 2.5") == 4.0

    def test_operator_precedence(self):
        """Test that * and / are evaluated before + and -."""
        # 2 + 3 * 4 should be 2 + 12 = 14, not 20
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
        # 10 - 2 * 3 should be 10 - 6 = 4
        assert self.evaluator.evaluate("10 - 2 * 3") == 4.0
        # 2 * 3 + 4 * 5 should be 6 + 20 = 26
        assert self.evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
        # Division precedence
        assert self.evaluator.evaluate("10 / 2 + 3") == 8.0

    def test_parentheses(self):
        """Test grouping with parentheses."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert self.evaluator.evaluate("10 - (2 + 3)") == 5.0
        assert self.evaluator.evaluate("((2 + 3) * (4 - 1))") == 15.0
        assert self.evaluator.evaluate("2 * (3 + (4 * 5))") == 46.0

    def test_unary_minus(self):
        """Test unary minus at start and inside expressions."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-3 + 5") == 2.0
        assert self.evaluator.evaluate("5 - -3") == 8.0
        assert self.evaluator.evaluate("-(2 + 1)") == -3.0
        assert self.evaluator.evaluate("-(-5)") == 5.0
        assert self.evaluator.evaluate("2 * -3") == -6.0

    def test_error_cases(self):
        """Test various error conditions."""
        # Empty expression
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate("")
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched"):
            self.evaluator.evaluate("(2 + 3")
        
        with pytest.raises(ValueError, match="Mismatched"):
            self.evaluator.evaluate("2 + 3)")

        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("10 / 0")

        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("2 + a")
        
        with pytest.raises(ValueError, match="Unexpected"):
            self.evaluator.evaluate("2 + 3 +")