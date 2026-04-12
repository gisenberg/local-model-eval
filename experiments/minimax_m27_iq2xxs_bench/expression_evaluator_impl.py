"""
Expression Evaluator
====================

A pure‑Python expression evaluator built with a recursive‑descent parser.
Supports:
* Basic arithmetic operators: +, -, *, /
* Parentheses for grouping
* Unary minus (e.g. "-3", "-(2+1)")
* Floating‑point numbers (e.g. "3.14")
* Proper operator precedence (MUL/DIV before ADD/SUB)
* Descriptive ValueError messages for malformed input
"""

import re
from typing import List

__all__ = ["ExpressionEvaluator"]


class ExpressionEvaluator:
    """
    Recursive‑descent parser that evaluates a simple arithmetic expression.

    Grammar (EBNF):
        expr     → term ( ('+' | '-') term )*
        term     → factor ( ('*' | '/') factor )*
        factor   → unary | '(' expr ')'
        unary    → '-' unary | primary
        primary  → NUMBER | '(' expr ')'

    Tokens are produced by a simple tokenizer; the parser then builds an AST
    and evaluates it directly.
    """

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def evaluate(self, expr: str) -> float:
        """
        Parse and evaluate *expr*.

        Parameters
        ----------
        expr: str
            The arithmetic expression to evaluate.

        Returns
        -------
        float
            The numeric result of the expression.

        Raises
        ------
        ValueError
            If the expression is malformed (mismatched parentheses,
            division by zero, invalid tokens, empty input, etc.).
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        try:
            self._tokenizer = Tokenizer(expr)
            self._tokens = self._tokenizer.tokenize()
        except ValueError as err:
            raise ValueError(f"Invalid token: {err}") from err

        # Prime the lookahead
        self._pos = 0

        result = self._parse_expr()

        # Make sure we have consumed all tokens
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token '{self._tokens[self._pos].value}'")

        return result

    # ----------------------------------------------------------------------
    # Private helpers – tokenizer
    # ----------------------------------------------------------------------

    def _peek(self) -> "Token":
        """Return the current token without advancing."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return Token("EOF", "")

    def _consume(self, expected_type: str) -> "Token":
        """Consume the next token if it matches *expected_type*."""
        token = self._peek()
        if token.type != expected_type:
            raise ValueError(f"Unexpected token '{token.value}'; expected {expected_type}")
        self._pos += 1
        return token

    # ----------------------------------------------------------------------
    # Private helpers – parser (recursive descent)
    # ----------------------------------------------------------------------

    def _parse_expr(self) -> float:
        """Parse a full expression: term (('+' | '-') term)*"""
        value = self._parse_term()
        while True:
            token = self._peek()
            if token.type == "PLUS":
                self._consume("PLUS")
                value = value + self._parse_term()
            elif token.type == "MINUS":
                self._consume("MINUS")
                value = value - self._parse_term()
            else:
                break
        return value

    def _parse_term(self) -> float:
        """Parse a term: factor (('*' | '/') factor)*"""
        value = self._parse_factor()
        while True:
            token = self._peek()
            if token.type == "MUL":
                self._consume("MUL")
                value = value * self._parse_factor()
            elif token.type == "DIV":
                self._consume("DIV")
                divisor = self._parse_factor()
                if divisor == 0.0:
                    raise ValueError("Division by zero")
                value = value / divisor
            else:
                break
        return value

    def _parse_factor(self) -> float:
        """Parse a factor: unary | '(' expr ')'"""
        token = self._peek()
        if token.type == "MINUS":
            self._consume("MINUS")
            return -self._parse_factor()
        elif token.type == "LPAREN":
            self._consume("LPAREN")
            value = self._parse_expr()
            if self._peek().type != "RPAREN":
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume("RPAREN")
            return value
        elif token.type == "NUMBER":
            self._consume("NUMBER")
            return float(token.value)
        else:
            raise ValueError(f"Unexpected token '{token.value}'")

    # ----------------------------------------------------------------------
    # Private helpers – token representation
    # ----------------------------------------------------------------------


class Token:
    """Simple token class."""

    __slots__ = ("type", "value")

    def __init__(self, token_type: str, value: str) -> None:
        self.type = token_type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type!r}, {self.value!r})"


class Tokenizer:
    """
    Simple tokenizer for arithmetic expressions.

    Recognizes:
        * NUMBER   – integer or floating point literals
        * PLUS     – '+'
        * MINUS    – '-'
        * MUL      – '*'
        * DIV      – '/'
        * LPAREN   – '('
        * RPAREN   – ')'
    """

    TOKEN_SPECS = [
        ("NUMBER",  r"(?P<num>\d+\.\d*|\d*\.?\d+)([eE][+-]?\d+)?"),
        ("PLUS",    r"\+"),
        ("MINUS",   r"-"),
        ("MUL",     r"\*"),
        ("DIV",     r"/"),
        ("LPAREN",  r"\("),
        ("RPAREN",  r"\)"),
        ("SKIP",    r"\s+"),          # whitespace
    ]

    def __init__(self, text: str) -> None:
        self.text = text
        self._regex = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in self.TOKEN_SPECS))

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        pos = 0
        while pos < len(self.text):
            match = self._regex.match(self.text, pos)
            if not match:
                raise ValueError(self.text[pos])
            group = match.lastgroup
            if group == "SKIP":
                pos = match.end()
                continue
            if group is None:
                raise ValueError(self.text[pos])
            value = match.group(group)
            tokens.append(Token(group, value))
            pos = match.end()
        return tokens


# --------------------------------------------------------------------------
# Quick demo / sanity check (optional)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ev = ExpressionEvaluator()
    print(ev.evaluate("2 + 3 * 4"))          # 14.0
    print(ev.evaluate("(2 + 3) * 4"))        # 20.0
    print(ev.evaluate("-3"))                  # -3.0
    print(ev.evaluate("-(2+1)"))              # -3.0
    print(ev.evaluate("3.14"))                # 3.14
