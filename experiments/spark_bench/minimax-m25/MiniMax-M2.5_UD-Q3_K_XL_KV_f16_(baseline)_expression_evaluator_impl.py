"""
Simple arithmetic expression evaluator.

Supports:
    - binary +, -, *, /
    - parentheses for grouping
    - unary minus (and plus)
    - floating‑point numbers

Raises ValueError for:
    - empty expression
    - mismatched parentheses
    - division by zero
    - invalid tokens
"""

from typing import List, Tuple, Optional
import re


# ----------------------------------------------------------------------
# Token definition
# ----------------------------------------------------------------------
class Token:
    """Simple token representation used by the lexer."""

    def __init__(self, type: str, value: any):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type!r}, {self.value!r})"


# Token type constants
TOKEN_NUMBER = "NUMBER"
TOKEN_PLUS = "PLUS"
TOKEN_MINUS = "MINUS"
TOKEN_MUL = "MUL"
TOKEN_DIV = "DIV"
TOKEN_LPAREN = "LPAREN"
TOKEN_RPAREN = "RPAREN"
TOKEN_EOF = "EOF"


# ----------------------------------------------------------------------
# Lexer – turns a string into a list of tokens
# ----------------------------------------------------------------------
class Lexer:
    """Tokenizes an arithmetic expression."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if self.text else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def advance(self) -> None:
        """Move the cursor one step forward."""
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def skip_whitespace(self) -> None:
        """Skip any whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self) -> Token:
        """
        Read a number (integer or floating‑point) and return a NUMBER token.
        Accepts forms like ``123``, ``0.5``, ``.5`` and ``123.``.
        """
        num_str = ""
        # Allow a leading dot
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == "."):
            num_str += self.current_char
            self.advance()

        if not num_str or num_str == ".":
            raise ValueError(f"Invalid number: {num_str!r}")

        # Convert to float – this also handles the “123.” case
        try:
            value = float(num_str)
        except ValueError:
            raise ValueError(f"Invalid number: {num_str!r}")

        return Token(TOKEN_NUMBER, value)

    # ------------------------------------------------------------------
    # Main token extraction
    # ------------------------------------------------------------------
    def get_next_token(self) -> Token:
        """
        Return the next token from the input.
        Raises ValueError for any unrecognized character.
        """
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            # Number (or dot‑starting number)
            if self.current_char.isdigit() or self.current_char == ".":
                return self.number()

            # Operators and parentheses
            if self.current_char == "+":
                self.advance()
                return Token(TOKEN_PLUS, "+")
            if self.current_char == "-":
                self.advance()
                return Token(TOKEN_MINUS, "-")
            if self.current_char == "*":
                self.advance()
                return Token(TOKEN_MUL, "*")
            if self.current_char == "/":
                self.advance()
                return Token(TOKEN_DIV, "/")
            if self.current_char == "(":
                self.advance()
                return Token(TOKEN_LPAREN, "(")
            if self.current_char == ")":
                self.advance()
                return Token(TOKEN_RPAREN, ")")

            # Anything else is illegal
            raise ValueError(f"Invalid token: {self.current_char!r}")

        # End of input
        return Token(TOKEN_EOF, None)


# ----------------------------------------------------------------------
# Recursive‑descent parser
# ----------------------------------------------------------------------
class Parser:
    """
    Recursive‑descent parser that evaluates the expression while parsing.
    Grammar (operator precedence from low to high):
        expr   → term ((+|-) term)*
        term   → factor ((*|/) factor)*
        factor → (+|-) factor | primary
        primary→ NUMBER | '(' expr ')'
    """

    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _error(self, msg: str) -> None:
        """Raise a ValueError with the given message."""
        raise ValueError(msg)

    def _eat(self, token_type: str) -> None:
        """
        Consume the current token if it matches ``token_type``.
        Otherwise raise an error.
        """
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self._error(f"Unexpected token: {self.current_token}")

    # ------------------------------------------------------------------
    # Parsing entry point
    # ------------------------------------------------------------------
    def parse(self) -> float:
        """Parse the whole expression and return its numeric value."""
        result = self._expr()
        if self.current_token.type != TOKEN_EOF:
            self._error(f"Unexpected token at end of expression: {self.current_token}")
        return result

    # ------------------------------------------------------------------
    # Grammar rules
    # ------------------------------------------------------------------
    def _expr(self) -> float:
        """Expression: term ((+|-) term)*"""
        result = self._term()
        while self.current_token.type in (TOKEN_PLUS, TOKEN_MINUS):
            op = self.current_token.type
            self._eat(op)
            right = self._term()
            if op == TOKEN_PLUS:
                result = result + right
            else:
                result = result - right
        return result

    def _term(self) -> float:
        """Term: factor ((*|/) factor)*"""
        result = self._factor()
        while self.current_token.type in (TOKEN_MUL, TOKEN_DIV):
            op = self.current_token.type
            self._eat(op)
            right = self._factor()
            if op == TOKEN_MUL:
                result = result * right
            else:
                if right == 0:
                    self._error("Division by zero")
                result = result / right
        return result

    def _factor(self) -> float:
        """Factor: (+|-) factor | primary"""
        if self.current_token.type == TOKEN_MINUS:
            self._eat(TOKEN_MINUS)
            return -self._factor()
        if self.current_token.type == TOKEN_PLUS:
            self._eat(TOKEN_PLUS)
            return self._factor()
        return self._primary()

    def _primary(self) -> float:
        """Primary: NUMBER | '(' expr ')'"""
        token = self.current_token
        if token.type == TOKEN_NUMBER:
            self._eat(TOKEN_NUMBER)
            return token.value
        if token.type == TOKEN_LPAREN:
            self._eat(TOKEN_LPAREN)
            result = self._expr()
            if self.current_token.type != TOKEN_RPAREN:
                self._error("Mismatched parentheses: missing ')'")
            self._eat(TOKEN_RPAREN)
            return result
        self._error(f"Unexpected token in primary: {self.current_token}")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
class ExpressionEvaluator:
    """
    Simple arithmetic expression evaluator.

    Example:
        >>> ev = ExpressionEvaluator()
        >>> ev.evaluate("2 + 3 * (4 - 1)")
        11.0
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate the given arithmetic expression.

        Args:
            expr: The expression string to evaluate.

        Returns:
            The numeric result as a ``float``.

        Raises:
            ValueError: If the expression is empty, contains mismatched
                        parentheses, a division by zero, or any illegal token.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        lexer = Lexer(expr)
        parser = Parser(lexer)
        # The parser itself raises ValueError for the various error cases.
        return parser.parse()


# ----------------------------------------------------------------------
# Quick manual test (can be removed or commented out in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ev = ExpressionEvaluator()
    tests = [
        "2 + 3",
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "-5",
        "--3",
        "-(2+1)",
        "3.14 * 2",
        "10 / 2 / 5",
    ]
    for t in tests:
        print(f"{t} => {ev.evaluate(t)}")
