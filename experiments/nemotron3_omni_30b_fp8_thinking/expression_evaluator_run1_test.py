"""Expression evaluator implementing a recursive‑descent parser.

The evaluator supports the binary operators ``+``, ``-``, ``*`` and ``/`` with
the usual precedence, parentheses for grouping, unary minus and floating‑point
numbers.  It never uses ``eval`` or ``ast.literal_eval`` and raises
``ValueError`` for any syntactic or semantic error (mismatched parentheses,
division by zero, invalid tokens, empty input, …).

Typical usage
-------------
>>> >>> evaluator = ExpressionEvaluator()
>>> evaluator.evaluate("3 + 4 * (2 - 1)")
7.0
"""

from __future__ import annotations

from typing import List, Optional, NamedTuple


class Token(NamedTuple):
    """A lexical token produced by the lexer."""
    type: str          # e.g. 'NUMBER', 'PLUS', 'MINUS', 'MUL', 'DIV', 'LPAREN', 'RPAREN'
    value: str         # textual representation (numeric string for NUMBER, the operator itself otherwise)


def _tokenize(expr: str) -> List[Token]:
    """
    Convert *expr* into a list of ``Token`` objects.

    The function scans the string character‑by‑character, builds numeric
    tokens (including the decimal point) and emits operator / parenthesis
    tokens.  Whitespace is ignored.  Any character that is not part of the
    supported syntax raises ``ValueError``.
    """
    if not expr or not expr.strip():
        raise ValueError("Empty expression")

    tokens: List[Token] = []
    i = 0
    n = len(expr)

    while i < n:
        ch = expr[i]

        if ch.isspace():
            i += 1
            continue

        if ch.isdigit() or ch == '.':
            # parse a floating‑point number
            start = i
            while i < n and (expr[i].isdigit() or expr[i] == '.'):
                i += 1
            num_str = expr[start:i]
            # ``float`` will raise its own error if the string is not a valid number,
            # but we want a uniform error message.
            try:
                float(num_str)  # validation only
            except ValueError as exc:
                raise ValueError(f"Invalid number literal: {num_str}") from exc
            tokens.append(Token('NUMBER', num_str))
            continue

        if ch == '+':
            tokens.append(Token('PLUS', '+'))
            i += 1
            continue

        if ch == '-':
            tokens.append(Token('MINUS', '-'))
            i += 1
            continue

        if ch == '*':
            tokens.append(Token('MUL', '*'))
            i += 1
            continue

        if ch == '/':
            tokens.append(Token('DIV', '/'))
            i += 1
            continue

        if ch == '(':
            tokens.append(Token('LPAREN', '('))
            i += 1
            continue

        if ch == ')':
            tokens.append(Token('RPAREN', ')'))
            i += 1
            continue

        raise ValueError(f"Invalid character '{ch}' at position {i}")

    return tokens


class Parser:
    """
    Recursive‑descent parser for the grammar:

    .. grammar::
        expr   : term (('+' | '-') term)*
        term   : factor (('*' | '/') factor)*
        factor : ('-' factor) | primary
        primary: NUMBER | '(' expr ')'

    The parser works on a list of :class:`Token` objects produced by
    :func:`_tokenize`.
    """

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens: List[Token] = tokens
        self.pos: int = 0

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def peek(self) -> Optional[Token]:
        """Return the current token without consuming it, or ``None`` if at EOF."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected_type: str) -> Token:
        """
        Consume the current token if its type matches *expected_type*,
        otherwise raise ``ValueError`` indicating the expected token.
        """
        token = self.peek()
        if token is None:
            raise ValueError(f"Unexpected end of input (expected {expected_type})")
        if token.type != expected_type:
            raise ValueError(f"Expected token type '{expected_type}', got '{token.type}'")
        self.pos += 1
        return token

    # --------------------------------------------------------------------- #
    # Grammar implementation
    # --------------------------------------------------------------------- #
    def parse_expr(self) -> float:
        """Parse an expression (addition / subtraction)."""
        value = self.parse_term()
        while (tok := self.peek()) and tok.type in ('PLUS', 'MINUS'):
            self.consume(tok.type)          # consume the operator
            if tok.type == 'PLUS':
                value += self.parse_term()
            else:  # MINUS
                value -= self.parse_term()
        return value

    def parse_term(self) -> float:
        """Parse a term (multiplication / division)."""
        value = self.parse_factor()
        while (tok := self.peek()) and tok.type in ('MUL', 'DIV'):
            self.consume(tok.type)          # consume the operator
            if tok.type == 'MUL':
                value *= self.parse_factor()
            else:  # DIV
                divisor = self.parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                value /= divisor
        return value

    def parse_factor(self) -> float:
        """
        Parse a factor, which may be a unary minus followed by a primary.
        ``-`` here is always unary because the grammar treats it as part of the
        factor production.
        """
        if self.peek() and self.peek().type == 'MINUS':
            self.consume('MINUS')
            return -self.parse_factor()
        return self.parse_primary()

    def parse_primary(self) -> float:
        """Parse a primary expression: a number or a parenthesised sub‑expression."""
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of input while parsing primary")
        if tok.type == 'NUMBER':
            self.consume('NUMBER')
            return float(tok.value)
        if tok.type == 'LPAREN':
            self.consume('LPAREN')
            value = self.parse_expr()
            self.consume('RPAREN')
            return value
        raise ValueError(f"Unexpected token {tok.type} in primary")


class ExpressionEvaluator:
    """
    Public façade that evaluates an arithmetic expression string.

    Parameters
    ----------
    expr : str
        The arithmetic expression to evaluate.  Whitespace is ignored.

    Returns
    -------
    float
        The numeric result of the evaluation.

    Raises
    ------
    ValueError
        If the expression is empty, contains invalid tokens,
        has mismatched parentheses, division by zero, etc.
    """

    def evaluate(self, expr: str) -> float:
        """Parse and evaluate *expr*; see class docstring for details."""
        tokens = _tokenize(expr)
        parser = Parser(tokens)
        result = parser.parse_expr()
        if parser.peek() is not None:
            raise ValueError(f"Invalid token(s) after expression: {parser.peek()}")
        return result

import pytest


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Provide a fresh evaluator instance for each test."""
    return ExpressionEvaluator()


def test_basic_addition(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("1.5 + 2.5") == 4.0


def test_operator_precedence(evaluator: ExpressionEvaluator) -> None:
    # multiplication has higher precedence than addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # division has higher precedence than subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0
    # mixed precedence with parentheses
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0


def test_unary_minus(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-(2 * 3)") == -6.0
    # unary minus combined with parentheses and other operators
    assert evaluator.evaluate("-(2 + 3 * 4)") == -14.0


def test_division_by_zero(evaluator: ExpressionEvaluator) -> None:
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / (2 - 2)")


def test_mismatched_parentheses(evaluator: ExpressionEvaluator) -> None:
    with pytest.raises(ValueError, match="Unexpected end of input"):
        evaluator.evaluate("(1 + 2")          # missing ')'

    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("1 + 2)")           # extra ')'

    with pytest.raises(ValueError, match="Unexpected end of input"):
        evaluator.evaluate("((1 + 2)")          # too many '('