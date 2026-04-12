"""
Expression Evaluator Module

A simple arithmetic expression evaluator built using a recursive descent parser.
Supports addition, subtraction, multiplication, division, parentheses, unary minus,
and floating point numbers. Does not use `eval()` or `ast.literal_eval()`.
"""

import re
from typing import List


class ExpressionEvaluator:
    """
    A parser and evaluator for arithmetic expressions.

    Supports:
        - Binary operators: +, -, *, /
        - Parentheses for grouping
        - Unary minus (e.g., "-3", "-(2+1)")
        - Floating point numbers (e.g., "3.14")

    Raises:
        ValueError: If the expression is empty, contains invalid tokens,
                    has mismatched parentheses, or attempts division by zero.
    """

    def __init__(self) -> None:
        """Initialize the evaluator with an empty token list."""
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parse and evaluate an arithmetic expression.

        Args:
            expr: A string containing a valid arithmetic expression.

        Returns:
            The result of the expression as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        mismatched parentheses, or attempts division by zero.
        """
        # Strip whitespace; reject empty strings
        expr = expr.strip()
        if not expr:
            raise ValueError("Empty expression")

        # Tokenize the raw string
        self._tokens = self._tokenize(expr)
        self._pos = 0

        # Parse and evaluate the top‑level expression
        result = self._parse_expression()

        # Ensure all tokens were consumed
        if self._pos != len(self._tokens):
            raise ValueError(f"Unexpected token: {self._tokens[self._pos]}")

        return result

    # ------------------------------------------------------------------
    # Lexer (tokenizer)
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(expr: str) -> List[str]:
        """
        Convert an expression string into a list of tokens (numbers, operators, parentheses).

        Args:
            expr: The raw expression string.

        Returns:
            A list of tokens.

        Raises:
            ValueError: If an unrecognized character is encountered.
        """
        tokens: List[str] = []
        i = 0
        while i < len(expr):
            ch = expr[i]

            # Skip whitespace
            if ch.isspace():
                i += 1
                continue

            # Parentheses and operators are individual tokens
            if ch in "()+-*/":
                tokens.append(ch)
                i += 1
                continue

            # Numbers (including decimals)
            if ch.isdigit() or ch == ".":
                start = i
                # Allow a leading decimal point (e.g., ".5")
                if ch == ".":
                    # Must be followed by a digit
                    if i + 1 >= len(expr) or not expr[i + 1].isdigit():
                        raise ValueError(f"Invalid token at position {i}: '.'")
                while i < len(expr) and (expr[i].isdigit() or expr[i] == "."):
                    i += 1
                tokens.append(expr[start:i])
                continue

            # Any other character is invalid
            raise ValueError(f"Invalid token at position {i}: '{ch}'")

        return tokens

    # ------------------------------------------------------------------
    # Recursive descent parser
    # ------------------------------------------------------------------
    def _parse_expression(self) -> float:
        """
        Parse and evaluate an expression according to the grammar:

            expression  ::= term (('+' | '-') term)*
            term        ::= factor (('*' | '/') factor)*
            factor      ::= '-' factor | '(' expression ')'
                          | number

        Returns:
            The numeric value of the parsed expression.
        """
        return self._parse_add_sub()

    def _parse_add_sub(self) -> float:
        """Parse a sequence of addition/subtractions."""
        left = self._parse_mul_div()
        while self._current() in ("+", "-"):
            op = self._advance()
            right = self._parse_mul_div()
            if op == "+":
                left += right
            else:
                left -= right
        return left

    def _parse_mul_div(self) -> float:
        """Parse a sequence of multiplication/divisions."""
        left = self._parse_factor()
        while self._current() in ("*", "/"):
            op = self._advance()
            right = self._parse_factor()
            if op == "*":
                left *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_factor(self) -> float:
        """Parse a factor: unary minus, parenthesized expression, or a number."""
        token = self._current()

        if token == "-":
            self._advance()  # consume '-'
            return -self._parse_factor()
        elif token == "(":
            self._advance()  # consume '('
            value = self._parse_expression()
            if self._current() != ")":
                raise ValueError("Mismatched parentheses: missing ')'")
            self._advance()  # consume ')'
            return value
        elif token and self._is_number(token):
            self._advance()
            return float(token)
        else:
            raise ValueError(f"Unexpected token: {token}")

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _current(self) -> str:
        """Return the current token without consuming it."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return ""  # end of token stream

    def _advance(self) -> str:
        """Consume and return the current token."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    @staticmethod
    def _is_number(token: str) -> bool:
        """Return True if the token can be interpreted as a numeric literal."""
        try:
            float(token)
            return True
        except ValueError:
            return False


# ----------------------------------------------------------------------
# Pytest test suite
# ----------------------------------------------------------------------
def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 / 4") == 2.5


def test_operator_precedence():
    """Test that multiplication/division have higher precedence than addition/subtraction."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 4 / 2") == 8.0
    assert evaluator.evaluate("100 / (5 * 2)") == 10.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    assert evaluator.evaluate("2 * (3 + 4) * 5") == 70.0


def test_parentheses():
    """Test parentheses grouping."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((2 + 3))") == 5.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("(2 + 3) * (4 + 5)") == 45.0


def test_unary_minus():
    """Test unary minus handling."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("--3") == 3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-(-2)") == 2.0
    assert evaluator.evaluate("5 * -2") == -10.0


def test_error_cases():
    """Test that appropriate ValueError exceptions are raised."""
    evaluator = ExpressionEvaluator()

    # Empty expression
    try:
        evaluator.evaluate("")
    except ValueError as e:
        assert "Empty expression" in str(e)

    # Mismatched parentheses
    try:
        evaluator.evaluate("(2 + 3")
    except ValueError as e:
        assert "Mismatched parentheses" in str(e)

    # Division by zero
    try:
        evaluator.evaluate("1 / 0")
    except ValueError as e:
        assert "Division by zero" in str(e)

    # Invalid token
    try:
        evaluator.evaluate("2 & 3")
    except ValueError as e:
        assert "Invalid token" in str(e)

    # Extra token after valid expression
    try:
        evaluator.evaluate("2 + 3 *")
    except ValueError as e:
        assert "Unexpected token" in str(e)
