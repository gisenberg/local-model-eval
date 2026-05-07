from typing import List, Union

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Addition (+), subtraction (-), multiplication (*), division (/)
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers (e.g., '3.14', '.5', '3.')
    
    Raises ValueError for:
    - Empty or whitespace-only expressions
    - Mismatched parentheses
    - Division by zero
    - Invalid tokens or malformed numbers
    """

    def __init__(self) -> None:
        self.tokens: List[Union[float, str]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0
        result = self._parse_expression()

        if self.tokens[self.pos] != 'EOF':
            raise ValueError(f"Invalid expression: unexpected token '{self.tokens[self.pos]}'")

        return result

    def _tokenize(self, expr: str) -> List[Union[float, str]]:
        """Convert expression string into a list of tokens."""
        tokens: List[Union[float, str]] = []
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
                try:
                    tokens.append(float(num_str))
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")
                i = j

            elif expr[i] in '+-*/()':
                tokens.append(expr[i])
                i += 1

            else:
                raise ValueError(f"Invalid token: {expr[i]}")

        tokens.append('EOF')
        return tokens

    def _current_token(self) -> Union[float, str]:
        """Return the token at the current position."""
        return self.tokens[self.pos]

    def _consume(self, expected: Union[float, str, None] = None) -> Union[float, str]:
        """Consume and return the current token. Optionally validate it."""
        token = self._current_token()
        if expected is not None and token != expected:
            raise ValueError(f"Expected {expected}, got {token}")
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            result = result + right if op == '+' else result - right
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
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary plus and minus."""
        if self._current_token() in ('+', '-'):
            op = self._consume()
            operand = self._parse_factor()
            return operand if op == '+' else -operand
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        token = self._current_token()
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

        raise ValueError(f"Unexpected token: {token}")

import pytest

def test_basic_precedence():
    """Test correct operator precedence for +, -, *, /"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 - 2 / 2") == 9.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses():
    """Test grouping with parentheses"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert ev.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

def test_unary_minus():
    """Test unary minus in various contexts"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("--5") == 5.0
    assert ev.evaluate("3 * -4") == -12.0
    assert ev.evaluate("-3.14 + .5") == pytest.approx(-2.64)

def test_floating_point():
    """Test floating point number parsing and arithmetic"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate(".5 + .5") == 1.0
    assert ev.evaluate("10.0 / 4.0") == 2.5
    assert ev.evaluate("3.") == 3.0

def test_error_cases():
    """Test ValueError raising for invalid inputs"""
    ev = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
        
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("10 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("2 + 3)")
        
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + a")
    with pytest.raises(ValueError, match="Invalid number"):
        ev.evaluate("3.1.4")