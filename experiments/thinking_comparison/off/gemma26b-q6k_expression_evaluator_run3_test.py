import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser
    to handle arithmetic operations, precedence, and parentheses.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates a mathematical expression string.
        
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

        self._tokens = self._tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError("No valid tokens found in expression")

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at end: {self._tokens[self._pos]}")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of meaningful tokens.
        """
        # Regex pattern: matches floats/integers, operators, or parentheses
        pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        tokens = re.findall(pattern, expr)
        
        # Validate that the reconstructed string from tokens matches the non-whitespace input
        # This ensures no invalid characters were skipped by the regex
        cleaned_expr = re.sub(r'\s+', '', expr)
        reconstructed = "".join(tokens)
        
        # Check if there are characters in the original string that weren't captured by the regex
        # We use a more robust check by verifying if the sum of lengths of tokens matches the stripped input
        # but since regex might skip invalid chars, we check if any invalid chars exist.
        if any(c not in "0123456789.+-*/() " for c in expr):
            # Find the specific invalid character for the error message
            for char in expr:
                if char not in "0123456789.+-*/() " and not char.isspace():
                    raise ValueError(f"Invalid token found: '{char}'")

        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Consumes and returns the current token."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Handles addition and subtraction (lowest precedence).
        Grammar: expression -> term { (+|-) term }
        """
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
        """
        Handles multiplication and division.
        Grammar: term -> factor { (*|/) factor }
        """
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
        """
        Handles unary minus and parentheses.
        Grammar: factor -> '-' factor | primary
        """
        if self._peek() == '-':
            self._consume()
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Handles numbers and parenthesized expressions.
        Grammar: primary -> number | '(' expression ')'
        """
        token = self._peek()

        if token == '(':
            self._consume()
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()  # consume ')'
            return result
        
        if token is not None and (token[0].isdigit() or token == '.'):
            token = self._consume()
            try:
                return float(token)
            except ValueError:
                raise ValueError(f"Invalid number format: {token}")
        
        raise ValueError(f"Unexpected token: {token if token else 'EOF'}")


# --- Pytest Tests ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 - 2") == 13.0
    assert evaluator.evaluate("10 * 5 / 2") == 25.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Parentheses override precedence
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("1 + 2)")

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("5 + -3") == 2.0
    assert evaluator.evaluate("--5") == 5.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Invalid token found"):
        evaluator.evaluate("10 & 5")
    with pytest.raises(ValueError, match="Expression is empty"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("5 + * 3")