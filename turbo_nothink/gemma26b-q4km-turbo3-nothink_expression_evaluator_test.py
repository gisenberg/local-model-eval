import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    Supports +, -, *, /, parentheses, and unary minus.
    """

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.pos: int = 0

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the input string into a list of meaningful tokens."""
        # Regex to find numbers (including floats), operators, and parentheses
        # We treat '-' as a separate token to handle unary logic in the parser
        pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        tokens = re

        # Clean whitespace and find all matches
        raw_tokens = re.findall(pattern, expr)
        
        # Validation: Check if the reconstructed string matches the original 
        # (ignoring whitespace) to detect invalid characters.
        cleaned_expr = re.sub(r'\s+', '', expr)
        reconstructed = "".join(raw_tokens)
        
        # Simple check for invalid characters by comparing length/content
        # This is a basic way to ensure no 'garbage' tokens were ignored
        if len(reconstructed) < len(cleaned_expr.replace(' ', '')):
            # This is a simplified check; in a production system, 
            # you'd track the exact index of the invalid character.
            raise ValueError("Expression contains invalid tokens")
            
        return raw_tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without advancing."""
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
        node = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                node /= right
        return node

    def _parse_factor(self) -> float:
        """Handles unary minus, parentheses, and numbers (highest precedence)."""
        token = self._peek()

        if token == '-':
            self._consume()  # consume '-'
            return -self._parse_factor()
        
        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()  # consume ')'
            return result

        # Try to parse a number
        try:
            token = self._consume()
            return float(token)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid token: {token}")

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates the given mathematical expression.
        
        Args:
            expr: The string expression to evaluate.
            
        Returns:
            The float result of the evaluation.
            
        Raises:
            ValueError: For division by zero, mismatched parens, or invalid syntax.
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError("Unexpected tokens at end of expression")
            
        return result

# --- Pytest Suite ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 - 2") == 13.0
    assert evaluator.evaluate("10 * 2 / 5") == 4.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # 2 + (3 * 4) = 14
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # (10 - 2) / 4 = 2
    assert evaluator.evaluate("(10 - 2) / 4") == 2.0

def test_parentheses_and_unary():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-(3 + 2)") == -5.0
    assert evaluator.evaluate("-5 * -5") == 25.0
    assert evaluator.evaluate("2 * (3 + (4 * 1))") == 14.0

def test_floats():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("1 / 3") == pytest.approx(0.3333333333)

def test_errors():
    evaluator = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
        
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
        
    with pytest.raises(ValueError, match="invalid tokens|Unexpected tokens"):
        # Testing an invalid character '@'
        evaluator.evaluate("10 + @")

    with pytest.raises(ValueError, match="Unexpected tokens"):
        evaluator.evaluate("1 + 2 3")