```python
import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    Supports +, -, *, /, unary minus, parentheses, and floating point numbers.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: For division by zero, mismatched parentheses, 
                        invalid tokens, or empty expressions.
        """
        self._tokens = self._tokenize(expr)
        self._pos = 0
        
        if not self._tokens:
            raise ValueError("Expression cannot be empty")
            
        result = self._parse_expression()
        
        # If there are tokens left, it means there's a syntax error (e.g., mismatched parentheses)
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token '{self._tokens[self._pos]}' at position {self._pos}")
            
        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of tokens.
        
        Args:
            expr: The raw expression string.
            
        Returns:
            A list of strings representing tokens (numbers, operators, parentheses).
            
        Raises:
            ValueError: If an invalid character is encountered.
        """
        tokens = []
        i = 0
        while i < len(expr):
            char = expr[i]
            if char.isspace():
                i += 1
                continue
            if char in "+-*/()":
                tokens.append(char)
                i += 1
            elif char.isdigit() or char == '.':
                num_str = ""
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.' and '.' in num_str:
                        raise ValueError(f"Invalid number format: multiple decimal points")
                    num_str += expr[i]
                    i += 1
                tokens.append(num_str)
            else:
                raise ValueError(f"Invalid character '{char}' in expression")
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
        """
        Parses addition and subtraction (lowest precedence).
        Grammar: Expression -> Term { ('+'|'-') Term }
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
        Parses multiplication and division.
        Grammar: Term -> Factor { ('*'|'/') Factor }
        """
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
        """
        Parses unary minus.
        Grammar: Unary -> '-' Unary | Primary
        """
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parses numbers and parenthesized expressions.
        Grammar: Primary -> Number | '(' Expression ')'
        """
        token = self._consume()
        
        if token == '(':
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            return result
        
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Expected number but found '{token}'")

# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 - 3") == 5.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Left-to-right for same precedence
    assert evaluator.evaluate("10 - 5 - 2") == 3.0
    # Division before subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 2) / 5") == 2.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5 + 3") == -2.0
    assert evaluator.evaluate("5 + -3") == 2.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("-(-5)") == 5.0
    assert evaluator.evaluate("-3 * -2") == 6.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    # Mismatched parentheses (missing closing)
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    # Mismatched parentheses (extra closing)
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("1 + 2)")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + a")
    
    # Empty expression
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")
    
    # Invalid number format
    with pytest.raises(ValueError, match="Invalid number format"):
        evaluator.evaluate("3.14.15")
```