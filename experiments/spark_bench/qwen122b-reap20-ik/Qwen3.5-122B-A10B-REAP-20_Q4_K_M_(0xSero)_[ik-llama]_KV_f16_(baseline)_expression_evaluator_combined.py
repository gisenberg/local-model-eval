import re
from typing import List, Tuple, Union, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        self.tokens: List[Tuple[str, Union[float, str]]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: A string containing the mathematical expression.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        self._tokenize(expr)
        self.pos = 0
        
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: {self.tokens[self.pos]}")
            
        return result

    def _tokenize(self, expr: str) -> None:
        """
        Converts the input string into a list of tokens.
        
        Args:
            expr: The input expression string.
            
        Raises:
            ValueError: If an invalid character is encountered.
        """
        self.tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
            
            if char in '+-*/()':
                self.tokens.append((char, char))
                i += 1
                continue
            
            if char.isdigit() or char == '.':
                start = i
                has_dot = False
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format at index {start}")
                        has_dot = True
                    i += 1
                
                num_str = expr[start:i]
                try:
                    self.tokens.append(('NUMBER', float(num_str)))
                except ValueError:
                    raise ValueError(f"Invalid number '{num_str}' at index {start}")
                continue
            
            raise ValueError(f"Invalid character '{char}' at index {i}")
        
        self.tokens.append(('EOF', None))

    def _current_token(self) -> Tuple[str, Union[float, str]]:
        """Returns the current token or EOF if at end."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ('EOF', None)

    def _consume(self, expected_type: Optional[str] = None) -> Tuple[str, Union[float, str]]:
        """
        Consumes the current token and advances the position.
        
        Args:
            expected_type: If provided, raises ValueError if token type doesn't match.
            
        Returns:
            The consumed token.
            
        Raises:
            ValueError: If the token type doesn't match expected_type.
        """
        token = self._current_token()
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[0]}")
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses an expression handling addition and subtraction (lowest precedence).
        Grammar: Expression -> Term { ('+' | '-') Term }
        """
        left = self._parse_term()
        
        while self._current_token()[0] in ('+', '-'):
            op = self._consume()[1]
            right = self._parse_term()
            if op == '+':
                left += right
            else:
                left -= right
                
        return left

    def _parse_term(self) -> float:
        """
        Parses a term handling multiplication and division (higher precedence).
        Grammar: Term -> Factor { ('*' | '/') Factor }
        """
        left = self._parse_factor()
        
        while self._current_token()[0] in ('*', '/'):
            op = self._consume()[1]
            right = self._parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
                
        return left

    def _parse_factor(self) -> float:
        """
        Parses a factor handling unary minus and primary values.
        Grammar: Factor -> ('-')* Primary
        """
        sign = 1
        while self._current_token()[0] == '-':
            self._consume()
            sign *= -1
            
        return sign * self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parses primary values: numbers or parenthesized expressions.
        Grammar: Primary -> NUMBER | '(' Expression ')'
        """
        token = self._current_token()
        
        if token[0] == 'NUMBER':
            self._consume()
            return token[1]
        
        if token[0] == '(':
            self._consume('(')
            val = self._parse_expression()
            if self._current_token()[0] != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._consume(')')
            return val
            
        raise ValueError(f"Unexpected token: {token}")

# --- Tests ---
# To run these tests, save this file as `evaluator.py` and run:
# pytest test_evaluator.py -v

import pytest

def test_basic_arithmetic():
    """Test basic addition, subtraction, multiplication, and division."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 2") == 4.0
    assert evaluator.evaluate("10 - 5") == 5.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 2") == 5.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0

def test_precedence_and_floats():
    """Test operator precedence and floating point numbers."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2.5 + 3.5") == 6.0
    assert evaluator.evaluate("10 / 3") == pytest.approx(3.333333)
    assert evaluator.evaluate("1 + 2 * 3 - 4 / 2") == 5.0
    assert evaluator.evaluate(".5 + .5") == 1.0

def test_parentheses():
    """Test grouping with parentheses."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    """Test unary minus operator."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("- -5") == 5.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("-3 * 4") == -12.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_error_cases():
    """Test error handling for invalid inputs."""
    evaluator = ExpressionEvaluator()
    
    # Empty expression
    with pytest.raises(ValueError):
        evaluator.evaluate("")
        
    # Mismatched parentheses
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + 3)")
        
    # Division by zero
    with pytest.raises(ValueError):
        evaluator.evaluate("1 / 0")
        
    # Invalid tokens
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + 3 4")
