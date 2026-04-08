import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic, 
    parentheses, unary operators, and floating point numbers using 
    a recursive descent parser.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The result of the evaluation.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens, 
                        has mismatched parentheses, or results in division by zero.
        """
        self._tokens = self._tokenize(expr)
        if not self._tokens:
            raise ValueError("Empty expression")

        self._pos = 0
        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """
        Splits the input string into a list of mathematical tokens.
        """
        # Matches floating point numbers, operators, and parentheses
        token_pattern = re.compile(r"(\d*\.\d+|\d+|[+\-*/()])")
        tokens = []
        
        # We iterate through the string to ensure no invalid characters are skipped
        last_end = 0
        for match in token_pattern.finditer(expr):
            # Check for invalid characters between matches
            between = expr[last_end:match.start()]
            if between.strip():
                raise ValueError(f"Invalid token found: {between.strip()}")
            
            tokens.append(match.group())
            last_end = match.end()
        
        # Check for trailing invalid characters
        trailing = expr[last_end:]
        if trailing.strip():
            raise ValueError(f"Invalid token found: {trailing.strip()}")
            
        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Returns the current token and advances the position."""
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
        """Handles unary minus (e.g., -5, --5)."""
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and grouped expressions in parentheses."""
        token = self._peek()

        if token == '(':
            self._consume()  # Consume '('
            result = self._parse_expression()
            
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            
            return result
        
        if token is not None and (token[0].isdigit() or token[0] == '.'):
            return float(self._consume())
        
        raise ValueError(f"Expected number or '(', found {token if token else 'end of expression'}")

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