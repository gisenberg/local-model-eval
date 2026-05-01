from typing import List, Union

Token = Union[float, str]

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14, .5, 3.)
    
    Raises ValueError for invalid input, mismatched parentheses, 
    division by zero, or empty expressions.
    """

    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        self._tokens = self._tokenize(expr)
        self._pos = 0
        result = self._parse_expression()
        
        if self._current_token() != 'EOF':
            raise ValueError("Mismatched parentheses or unexpected token")
        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Convert expression string into a list of tokens."""
        if not expr.strip():
            raise ValueError("Empty expression")
            
        tokens: List[Token] = []
        i = 0
        n = len(expr)
        
        while i < n:
            if expr[i].isspace():
                i += 1
                continue
                
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format: multiple decimal points")
                        has_dot = True
                    j += 1
                    
                num_str = expr[i:j]
                if num_str == '.':
                    raise ValueError("Invalid token: '.'")
                tokens.append(float(num_str))
                i = j
                
            elif expr[i] in '+-*/()':
                tokens.append(expr[i])
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")
                
        if not tokens:
            raise ValueError("Empty expression")
        tokens.append('EOF')
        return tokens

    def _current_token(self) -> Token:
        """Return the token at the current position."""
        return self._tokens[self._pos]

    def _consume(self, expected: str = None) -> Token:
        """Consume and return the current token. Optionally validate it."""
        token = self._current_token()
        if expected is not None and token != expected:
            raise ValueError(f"Expected '{expected}', got '{token}'")
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary operators, numbers, and parenthesized expressions."""
        token = self._current_token()
        
        if token == '-':
            self._consume()
            return -self._parse_factor()
        if token == '+':
            self._consume()
            return self._parse_factor()
        if isinstance(token, float):
            self._consume()
            return token
        if token == '(':
            self._consume()
            result = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()
            return result
        if token == ')':
            raise ValueError("Mismatched parentheses")
            
        raise ValueError(f"Unexpected token: '{token}'")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic_and_precedence(evaluator):
    """Test correct operator precedence and basic operations."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_unary_minus_and_floats(evaluator):
    """Test unary minus support and floating point numbers."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("-.5 + 1.5") == 1.0

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (2 - 2)")

def test_mismatched_parentheses(evaluator):
    """Test that mismatched parentheses raise ValueError."""
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("((2 + 3)")

def test_invalid_tokens_and_empty(evaluator):
    """Test handling of invalid tokens and empty expressions."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + abc")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + 3 @")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + 3 *")  # Trailing operator