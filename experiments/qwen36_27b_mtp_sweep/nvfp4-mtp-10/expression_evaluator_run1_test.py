from typing import List

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the mathematical expression string.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is invalid, empty, or contains errors.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Empty expression")

        parser = _Parser(tokens)
        return parser.parse()

    def _tokenize(self, expr: str) -> List[str]:
        """Tokenizes the expression string into a list of strings."""
        tokens = []
        i = 0
        n = len(expr)
        while i < n:
            char = expr[i]
            if char.isspace():
                i += 1
                continue

            # Handle numbers (integers and floats)
            if char.isdigit() or char == '.':
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                tokens.append(expr[start:i])
                continue

            # Handle operators and parentheses
            if char in '+-*/()':
                tokens.append(char)
                i += 1
                continue

            raise ValueError(f"Invalid character: '{char}'")

        return tokens


class _Parser:
    """Internal recursive descent parser."""

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> float:
        if not self.tokens:
            raise ValueError("Empty expression")
        result = self._parse_expression()
        
        # Check for trailing tokens (syntax errors)
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if token == ')':
                raise ValueError("Mismatched parentheses")
            raise ValueError(f"Unexpected token: '{token}'")
            
        return result

    def _current_token(self) -> str:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self) -> str:
        token = self._current_token()
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        # Grammar: Expression -> Term { (+ | -) Term }
        left = self._parse_term()
        while self._current_token() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                left += right
            else:
                left -= right
        return left

    def _parse_term(self) -> float:
        # Grammar: Term -> Factor { (* | /) Factor }
        left = self._parse_factor()
        while self._current_token() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_factor(self) -> float:
        # Grammar: Factor -> Number | ( Expression ) | - Factor
        token = self._current_token()

        if token is None:
            raise ValueError("Unexpected end of expression")

        # Handle Unary Minus
        if token == '-':
            self._consume()
            return -self._parse_factor()

        # Handle Parentheses
        if token == '(':
            self._consume()
            val = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()
            return val

        # Handle unexpected closing parenthesis
        if token == ')':
            raise ValueError("Mismatched parentheses")

        # Handle invalid tokens in factor position (e.g., 2 * + 3)
        if token in ('+', '*', '/'):
            raise ValueError(f"Unexpected token '{token}'")

        # Parse Number
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid number format: '{token}'")

import pytest

def test_basic_arithmetic():
    """Test basic addition, subtraction, multiplication, and division."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3") == 5.0
    assert ev.evaluate("10 - 4") == 6.0
    assert ev.evaluate("2 * 3") == 6.0
    assert ev.evaluate("10 / 2") == 5.0

def test_precedence():
    """Test operator precedence (* and / before + and -)."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 - 8 / 2") == 6.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses():
    """Test grouping with parentheses."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("2 * (3 + 4)") == 14.0
    assert ev.evaluate("((2))") == 2.0
    assert ev.evaluate("1 + (2 * (3 + 4))") == 15.0

def test_unary_minus():
    """Test unary minus support."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-3 * 2") == -6.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("- - 3") == 3.0
    assert ev.evaluate("2 * -3") == -6.0

def test_floats_and_errors():
    """Test floating point numbers and error handling."""
    ev = ExpressionEvaluator()
    
    # Floats
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate("1.5 + 2.5") == 4.0
    
    # Errors
    with pytest.raises(ValueError):
        ev.evaluate("1 / 0")  # Division by zero
    with pytest.raises(ValueError):
        ev.evaluate("")       # Empty expression
    with pytest.raises(ValueError):
        ev.evaluate("2 + a")  # Invalid token
    with pytest.raises(ValueError):
        ev.evaluate("(2 + 3") # Mismatched parentheses
    with pytest.raises(ValueError):
        ev.evaluate("2 + 3)") # Mismatched parentheses