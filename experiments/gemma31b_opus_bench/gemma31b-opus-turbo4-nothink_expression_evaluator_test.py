import re
from typing import List, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser that evaluates mathematical expressions.
    Grammar:
    expression -> term { ('+'|'-') term }
    term       -> factor { ('*'|'/') factor }
    factor     -> ['-'] primary
    primary    -> number | '(' expression ')'
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the input string into a list of tokens."""
        # Regex matches: floating point numbers, operators, and parentheses
        pattern = r'\d*\.?\d+|[+\-*/()]'
        tokens = re.findall(pattern, expr)
        
        # Validate that we didn't lose characters (detects invalid characters)
        if len(''.join(tokens)) != len(expr.replace(' ', '')):
            raise ValueError("Expression contains invalid characters")
        
        return tokens

    def _peek(self) -> str:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else ''

    def _consume(self) -> str:
        """Returns the current token and advances the position."""
        token = self._peek()
        self._pos += 1
        return token

    def _parse_number(self) -> float:
        """Parses a numeric literal."""
        token = self._consume()
        try:
            return float(token)
        except (ValueError, TypeError):
            raise ValueError(f"Expected number, got '{token}'")

    def _parse_primary(self) -> float:
        """Parses a primary element: a number or a parenthesized expression."""
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()  # consume ')'
            return result
        
        if token == '-':
            self._consume()  # consume unary minus
            return -self._parse_primary()
        
        if token and (token[0].isdigit() or (len(token) > 1 and token[1].isdigit())):
            return self._parse_number()
        
        raise ValueError(f"Unexpected token '{token}'")

    def _parse_term(self) -> float:
        """Parses multiplication and division (higher precedence)."""
        left = self._parse_primary()
        
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_primary()
            if op == '*':
                left *= right
            elif op == '/':
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_expression(self) -> float:
        """Parses addition and subtraction (lower precedence)."""
        left = self._parse_term()
        
        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                left += right
            else:
                left -= right
        return left

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string.
        
        Args:
            expr: String containing the expression (e.g., "3 + 4 * 2").
            
        Returns:
            The float result of the evaluation.
            
        Raises:
            ValueError: If the expression is malformed or mathematically invalid.
        """
        expr = expr.replace(' ', '')
        if not expr:
            raise ValueError("Empty expression")
        
        self._tokens = self._tokenize(expr)
        self._pos = 0
        
        result = self._parse_expression()
        
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token '{self._tokens[self._pos]}' at end of expression")
        
        return result

import pytest


@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    """Test simple addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 2") == 5.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_precedence(evaluator):
    """Test that * and / are evaluated before + and -."""
    # 2 + 3 * 4 = 14, not 20
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # 10 - 4 / 2 = 8, not 3
    assert evaluator.evaluate("10 - 4 / 2") == 8.0
    # Complex chain
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses(evaluator):
    """Test that parentheses override default precedence."""
    # (2 + 3) * 4 = 20
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # Nested parentheses
    assert evaluator.evaluate("2 * (3 + (4 / 2))") == 10.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0

def test_unary_minus(evaluator):
    """Test unary minus with numbers and expressions."""
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-5 + 3") == -2.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0  # Double negative
    assert evaluator.evaluate("-3 * -2") == 6.0

def test_error_cases(evaluator):
    """Test various error conditions raise ValueError."""
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("1 + 2)")
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / (5 - 5)")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="contains invalid characters"):
        evaluator.evaluate("1 + 2 @ 3")
    
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")