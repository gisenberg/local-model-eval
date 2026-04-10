import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser 
    to compute results from strings containing +, -, *, /, and parentheses.
    """

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.pos: int = 0

    def tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of valid tokens.
        
        Args:
            expr: The raw string expression.
            
        Returns:
            A list of strings representing tokens.
            
        Raises:
            ValueError: If the expression is empty or contains invalid characters.
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        # Regex to find numbers (including floats), operators, and parentheses
        # We also capture any non-whitespace character to validate for invalid tokens
        token_pattern = re.compile(r"(\d+\.\d+|\d+|[+\-*/()]|\S)")
        tokens = token_pattern.findall(expr)
        
        for token in tokens:
            # If a token is not a number, an operator, or a parenthesis, it's invalid
            if not re.match(r"^\d+\.\d+$|^\d+$|[+\-*/()]", token):
                raise ValueError(f"Invalid token: {token}")
        
        return tokens

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates the mathematical expression.
        
        Args:
            expr: The string expression to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: For syntax errors, division by zero, or mismatched parentheses.
        """
        self.tokens = self.tokenize(expr)
        self.pos = 0
        
        result = self._parse_expression()
        
        # If there are tokens left after parsing the full expression, it's a syntax error (e.            e.g., "5 + 2 )")
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token after expression: {self.tokens[self.pos]}")
            
        return result

    def _peek(self) -> Optional[str]:
        """Returns the current token without advancing the pointer."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        """Returns the current token and advances the pointer."""
        token = self._peek()
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Handles lowest precedence: addition and subtraction.
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
        Handles medium precedence: multiplication and division.
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
        Handles highest precedence: unary operators and parentheses.
        Grammar: factor -> (+|-) factor | primary
        """
        if self._peek() == '-':
            self._consume()  # consume '-'
            return -self._parse_factor()
        if self._peek() == '+':
            self._consume()  # consume '+'
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Handles the base units: numbers and parenthesized expressions.
        Grammar: primary -> number | ( expression )
        """
        token = self._peek()
        
        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()  # consume ')'
            return result
        
        # Try to parse as a number
        try:
            return float(self._consume())
        except (ValueError, TypeError):
            raise ValueError(f"Expected number or parenthesis, found: {token}")

# --- Pytest Suite ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluable()
    assert evaluator.evaluate("1 + 2 + 3") == 6.0
    assert evaluator.evaluate("10 - 5") == 5.0
    assert evaluator.evaluate("4 * 2") == 8.0
    assert evaluator.evaluate("10 / 4") == 2.5

def test_precedence():
    evaluator = ExpressionEvaluable()
    # 1 + (2 * 3) = 7
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    # (10 - 2) * 3 = 24
    assert evaluator.evaluate("(10 - 2) * 3") == 24.0
    # 10 - 2 * 3 + 4 = 10 - 6 + 4 = 8
    assert evaluator.evaluate("10 - 2 * 3 + 4") == 8.0

def test_parentheses_and_unary():
    evaluator = ExpressionEvaluable()
    # Unary minus
    assert evaluator.evaluate("-5") == -5.0
    # Unary minus with expression
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    # Nested unary
    assert evaluator.evaluate("--5") == 5.0
    # Complex
    assert evaluator.evaluate("5 + (-2 * 3)") == -1.0

def test_error_cases():
    evaluator = ExpressionEvaluable()
    
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
        
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
        
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
        
    # Mismatched parentheses (extra closing)
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("1 + 2)")
        
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1 + @")

# Alias for testing compatibility if the class name is slightly different in the prompt
ExpressionEvaluable = ExpressionEvaluator