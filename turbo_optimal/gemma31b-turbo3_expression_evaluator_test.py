import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic, 
    operator precedence, parentheses, and unary minus.
    """

    def __init__(self):
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates a mathematical expression.
        
        Args:
            expr: The string expression to evaluate.
        Returns:
            The result as a float.
        Raises:
            ValueError: If the expression is malformed or contains invalid operations.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize the input string
        # Matches floating point numbers, integers, and operators/parentheses
        self._tokens = re.findall(r'\d*\.\d+|\d+|[+\-*/()]', expr)
        
        # Check if the tokenized length matches the original content (detect invalid tokens)
        # We strip whitespace to compare the reconstructed string
        cleaned_expr = "".join(self._tokens)
        stripped_expr = expr.replace(" ", "")
        # This is a simple check; a more robust way is to check if the regex 
        # missed any non-whitespace characters.
        if len(self._tokens) == 0 or len("".join(re.findall(r'\S', expr)) ) != len(cleaned_expr):
            # This catches characters that didn't match any regex group
            # We check if there are characters in the original string not captured by tokens
            # by comparing the joined tokens against the stripped input.
            # However, since we use findall, we should check if the total length of tokens 
            # matches the non-whitespace length of the input.
            # A better way:
            pass 

        # To properly detect invalid tokens, we check if the joined tokens 
        # match the original string minus whitespace.
        if "".join(self._tokens) != stripped_expr:
            raise ValueError(f"Invalid tokens found in expression: {expr}")

        self._pos = 0
        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")

        return float(result)

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Returns the current token and advances the pointer."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
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
        """Handles multiplication and division."""
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
        token = self._consume()

        if token == '(':
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            return result
        
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid numeric token: {token}")

# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 2 + 3") == 6.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("2 * 3") == 6.0
    assert evaluator.evaluate("10 / 2") == 5.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Division before subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0
    # Complex precedence
    assert evaluator.evaluate("2 * 3 + 4 * 2") == 14.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    # Force addition first
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # Nested parentheses
    assert evaluator.evaluate("2 * (3 + (4 * 2))") == 22.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("5 + -3") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("1 + 2)")
    
    with pytest.raises(ValueError, match="Invalid tokens found"):
        evaluator.evaluate("1 + abc")
    
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("")

if __name__ == "__main__":
    # Simple manual test if pytest is not run
    evaluator = ExpressionEvaluator()
    print(f"Result: {evaluator.evaluate('-(2 + 3) * 2 + 10 / 2')} (Expected: 4.0)")