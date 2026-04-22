from typing import Tuple, List

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Addition (+), Subtraction (-), Multiplication (*), Division (/)
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers (e.g., '3.14', '.5', '3.')
    
    Raises ValueError for:
    - Empty expressions
    - Invalid tokens
    - Mismatched parentheses
    - Division by zero
    """

    def __init__(self) -> None:
        self._tokens: List[Tuple[str, str]] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: The mathematical expression string.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._tokenize(expr)
        self._pos = 0
        result = self._parse_expression()

        if self._pos < len(self._tokens) and self._tokens[self._pos][0] != 'EOF':
            raise ValueError(f"Unexpected token: {self._tokens[self._pos][1]}")

        return result

    def _tokenize(self, expr: str) -> None:
        """Converts the input string into a list of (type, value) tokens."""
        self._tokens = []
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
                            raise ValueError("Invalid number format")
                        has_dot = True
                    j += 1
                
                num_str = expr[i:j]
                if num_str == '.':
                    raise ValueError("Invalid number format")
                self._tokens.append(('NUM', num_str))
                i = j
            elif expr[i] in '+-*/()':
                self._tokens.append((expr[i], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")

        self._tokens.append(('EOF', ''))

    def _current_token(self) -> Tuple[str, str]:
        """Returns the current token without advancing."""
        return self._tokens[self._pos]

    def _advance(self) -> None:
        """Moves to the next token."""
        self._pos += 1

    def _parse_expression(self) -> float:
        """Handles addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token()[0] in ('+', '-'):
            op = self._current_token()[0]
            self._advance()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Handles multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token()[0] in ('*', '/'):
            op = self._current_token()[0]
            self._advance()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Handles numbers, parentheses, and unary operators (highest precedence)."""
        token_type, token_val = self._current_token()
        
        if token_type == 'NUM':
            self._advance()
            return float(token_val)
        elif token_type == '(':
            self._advance()
            result = self._parse_expression()
            if self._current_token()[0] != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        elif token_type == '-':
            self._advance()
            return -self._parse_factor()
        elif token_type == '+':
            # Unary plus is supported for completeness
            self._advance()
            return self._parse_factor()
        else:
            raise ValueError(f"Unexpected token: {token_val}")

import pytest

def test_basic_precedence():
    """Tests correct operator precedence for +, -, *, /"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 / 2 - 3") == 2.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping():
    """Tests parentheses override default precedence"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert ev.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

def test_unary_minus():
    """Tests unary minus on numbers and grouped expressions"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("5 - -2") == 7.0
    assert ev.evaluate("--4") == 4.0

def test_floating_point_numbers():
    """Tests support for decimal numbers"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate(".5 + .5") == 1.0
    assert ev.evaluate("10 / 3") == pytest.approx(3.3333333333333335)
    assert ev.evaluate("3. * 2") == 6.0

def test_error_cases():
    """Tests ValueError raising for invalid inputs"""
    ev = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
        
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 & 3")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("3.14.5")
        
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("10 / 0.0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Unexpected token"):
        ev.evaluate("2 + 3)")