import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a Recursive Descent Parser.
    Supports +, -, *, /, unary minus, parentheses, and floating point numbers.
    """

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates the given mathematical expression.
        
        Args:
            expr: The string expression to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: For division by zero, mismatched parentheses, 
                         invalid tokens, or empty expressions.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression is empty")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        if not self.tokens:
            raise ValueError("No valid tokens found in expression")

        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: '{self.tokens[self.pos]}'")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the input string into a list of tokens (numbers, operators, parens)."""
        # Regex: matches floats/ints, operators, and parentheses
        token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')
        tokens = token_pattern.findall(expr)
        
        # Check if the reconstructed string matches the original (ignoring whitespace)
        # This helps detect invalid characters like 'abc' or '@'
        cleaned_expr = re.sub(r'\s+', '', expr)
        reconstructed = "".join(tokens)
        
        # Simple check: if we lost characters that weren't whitespace, there's an invalid token
        if len(reconstructed) < len(cleaned_expr):
            # Find the first character that wasn't captured
            raise ValueError("Expression contains invalid tokens")
            
        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without advancing the pointer."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        """Returns the current token and advances the pointer."""
        token = self._peek()
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """Handles addition and subtraction (lowest precedence)."""
        node = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                node += right
            else:
                node -= right
        return node

    def _parse_term(self) -> float:
        """Handles multiplication and division (medium precedence)."""
        node = self._parse_unary()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_unary()
            if op == '*':
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                node /= right
        return node

    def _parse_unary(self) -> float:
        """Handles unary minus."""
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and parentheses (highest precedence)."""
        token = selftor = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()  # consume ')'
            return result
        
        if token is None:
            raise ValueError("Unexpected end of expression")

        try:
            # Try to parse the token as a float
            val = float(self._consume())
            return val
        except ValueError:
            raise ValueError(f"Invalid token: '{token}'")

# --- Pytest Suite ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 - 2") == 13.0
    assert evaluator.evaluate("10 * 2 / 5") == 4.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Parentheses override precedence
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # Division before subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5 + 3") == -2.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("5 * -2") == -10.0
    assert evaluator.evaluate("--5") == 5.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="invalid tokens"):
        evaluator.evaluate("10 + abc")
        
    # Empty expression
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate("   ")

def test_complex_expression():
    evaluator = ExpressionEvaluator()
    # A mix of everything
    expr = "-(10 + (5 * 2)) / 2 + 3.5"
    # -(10 + 10) / 2 + 3.5 => -20 / 2 + 3.5 => -10 + 3.5 => -6.5
    assert evaluator.evaluate(expr) == -6.5