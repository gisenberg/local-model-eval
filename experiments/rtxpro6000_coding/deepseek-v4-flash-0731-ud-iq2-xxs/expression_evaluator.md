```python
import pytest
from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    A recursive descent parser for arithmetic expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    Raises ValueError for invalid inputs.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: The expression string to evaluate.

        Returns:
            The numeric result.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Empty expression")

        value, idx = self._parse_expression(tokens, 0)
        if idx != len(tokens):
            raise ValueError(f"Unexpected token: {tokens[idx]}")

        return value

    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert an expression string into a list of tokens.

        Tokens are strings: numbers, operators ('+', '-', '*', '/'),
        and parentheses ('(', ')').

        Raises:
            ValueError: If an invalid character is encountered.
        """
        tokens: List[str] = []
        i = 0
        n = len(expr)

        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue

            if ch.isdigit() or ch == '.':
                # Parse a number (including decimals)
                num_str = ''
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    num_str += expr[i]
                    i += 1
                # Validate that the number is a valid float
                try:
                    float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number: '{num_str}'")
                tokens.append(num_str)
                continue

            if ch in '+-*/()':
                tokens.append(ch)
                i += 1
                continue

            raise ValueError(f"Invalid character: '{ch}'")

        return tokens

    def _parse_expression(self, tokens: List[str], idx: int) -> Tuple[float, int]:
        """
        Parse an expression: Term { ('+' | '-') Term }.

        Returns:
            (value, new_index) where new_index is the position after the parsed expression.
        """
        value, idx = self._parse_term(tokens, idx)

        while idx < len(tokens) and tokens[idx] in ('+', '-'):
            op = tokens[idx]
            idx += 1
            rhs, idx = self._parse_term(tokens, idx)
            if op == '+':
                value += rhs
            else:  # '-'
                value -= rhs

        return value, idx

    def _parse_term(self, tokens: List[str], idx: int) -> Tuple[float, int]:
        """
        Parse a term: Factor { ('*' | '/') Factor }.

        Returns:
            (value, new_index).
        """
        value, idx = self._parse_factor(tokens, idx)

        while idx < len(tokens) and tokens[idx] in ('*', '/'):
            op = tokens[idx]
            idx += 1
            rhs, idx = self._parse_factor(tokens, idx)
            if op == '*':
                value *= rhs
            else:  # '/'
                if rhs == 0:
                    raise ValueError("Division by zero")
                value /= rhs

        return value, idx

    def _parse_factor(self, tokens: List[str], idx: int) -> Tuple[float, int]:
        """
        Parse a factor: '-' Factor | '(' Expression ')' | Number.

        Returns:
            (value, new_index).
        """
        if idx >= len(tokens):
            raise ValueError("Unexpected end of expression")

        token = tokens[idx]

        # Unary minus
        if token == '-':
            idx += 1
            value, idx = self._parse_factor(tokens, idx)
            return -value, idx

        # Parenthesized expression
        if token == '(':
            idx += 1
            value, idx = self._parse_expression(tokens, idx)
            if idx >= len(tokens) or tokens[idx] != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            idx += 1  # consume ')'
            return value, idx

        # Number
        try:
            value = float(token)
        except ValueError:
            raise ValueError(f"Invalid token: '{token}'")
        idx += 1
        return value, idx


# ===== Pytest tests =====

def test_basic_arithmetic():
    ev = ExpressionEvaluator()
    assert ev.evaluate("1+2") == 3
    assert ev.evaluate("2*3") == 6
    assert ev.evaluate("5/2") == 2.5
    assert ev.evaluate("1+2*3") == 7  # precedence handled

def test_precedence():
    ev = ExpressionEvaluator()
    assert ev.evaluate("2+3*4") == 14
    assert ev.evaluate("10-2*3") == 4
    assert ev.evaluate("20/4+2") == 7
    assert ev.evaluate("2*3+4/2") == 8

def test_parentheses():
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2+3)*4") == 20
    assert ev.evaluate("2*(3+4)") == 14
    assert ev.evaluate("(2+3)*(4-1)") == 15

def test_unary_minus():
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3
    assert ev.evaluate("-(2+1)") == -3
    assert ev.evaluate("3*-2") == -6
    assert ev.evaluate("5--2") == 7  # 5 - (-2)

def test_error_cases():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2+3")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("5/0")
    with pytest.raises(ValueError, match="Invalid character"):
        ev.evaluate("2+abc")
    with pytest.raises(ValueError, match="Unexpected token"):
        ev.evaluate("2+3)")
```