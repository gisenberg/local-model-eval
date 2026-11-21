Here's a complete implementation split into two files: `evaluator.py` (the evaluator) and `test_evaluator.py` (the tests).

## `evaluator.py`

```python
"""A recursive descent mathematical expression evaluator.

Supports +, -, *, / with correct operator precedence, parentheses for
grouping, unary minus, and floating-point numbers. Raises ValueError
with a descriptive message for empty expressions, invalid tokens,
mismatched parentheses and division by zero. Does NOT use eval() or
ast.literal_eval().
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class TokenType(Enum):
    """The lexical token kinds understood by the parser."""

    NUMBER = "NUMBER"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    EOF = "EOF"


@dataclass(frozen=True)
class Token:
    """A lexical token: its kind, raw text and position in the source."""

    type: TokenType
    value: str
    position: int


class ExpressionEvaluator:
    """Evaluate expressions such as ``2 + 3 * 4`` or ``-(1.5 * 2)``.

    The implementation is a hand-written recursive descent parser::

        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := ('+' | '-')* primary
        primary    := NUMBER | '(' expression ')'
    """

    def __init__(self) -> None:
        """Initialise the parser state (token list and cursor)."""
        self._tokens: List[Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate `expr` and return the result as a float.

        Args:
            expr: The expression to evaluate, e.g. ``"2 + 3 * 4"``.

        Returns:
            The numeric value of the expression.

        Raises:
            TypeError: If `expr` is not a string.
            ValueError: If the expression is empty, contains invalid
                tokens, has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"Expression must be a string, got {type(expr).__name__}")
        if not expr.strip():
            raise ValueError("Expression is empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0
        result = self._parse_expression()

        # Anything left over after a successful parse means a stray token,
        # e.g. an extra closing parenthesis.
        if self._current.type is not TokenType.EOF:
            tok = self._current
            if tok.type is TokenType.RPAREN:
                raise ValueError(
                    f"Mismatched parentheses: unexpected ')' at position {tok.position}"
                )
            raise ValueError(
                f"Unexpected token {tok.value!r} at position {tok.position}"
            )
        return result

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _tokenize(self, expr: str) -> List[Token]:
        """Split `expr` into a list of tokens (including a trailing EOF token).

        Raises:
            ValueError: If a character or malformed number is not recognised.
        """
        tokens: List[Token] = []
        i = 0
        n = len(expr)

        while i < n:
            ch = expr[i]

            if ch.isspace():
                i += 1
                continue

            if ch.isdigit():
                start = i
                while i < n and expr[i].isdigit():
                    i += 1
                if i < n and expr[i] == ".":
                    i += 1
                    if i >= n or not expr[i].isdigit():
                        raise ValueError(
                            f"Invalid number at position {start}: expected digits "
                            f"after '.' in {expr[start:i]!r}"
                        )
                    while i < n and expr[i].isdigit():
                        i += 1
                tokens.append(Token(TokenType.NUMBER, expr[start:i], start))
                continue

            if ch in "+-*/":
                kind = {
                    "+": TokenType.PLUS,
                    "-": TokenType.MINUS,
                    "*": TokenType.STAR,
                    "/": TokenType.SLASH,
                }[ch]
                tokens.append(Token(kind, ch, i))
                i += 1
                continue

            if ch == "(":
                tokens.append(Token(TokenType.LPAREN, ch, i))
                i += 1
                continue

            if ch == ")":
                tokens.append(Token(TokenType.RPAREN, ch, i))
                i += 1
                continue

            raise ValueError(f"Invalid token {ch!r} at position {i}")

        tokens.append(Token(TokenType.EOF, "", n))
        return tokens

    # ------------------------------------------------------------------
    # Parser state helpers
    # ------------------------------------------------------------------

    @property
    def _current(self) -> Token:
        """Return the token the parser is currently positioned at."""
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        """Consume the current token and return it (never past EOF)."""
        tok = self._tokens[self._pos]
        if tok.type is not TokenType.EOF:
            self._pos += 1
        return tok

    # ------------------------------------------------------------------
    # Recursive descent grammar
    # ------------------------------------------------------------------

    def _parse_expression(self) -> float:
        """Parse an additive expression: `term (('+' | '-') term)*`."""
        value = self._parse_term()
        while self._current.type in (TokenType.PLUS, TokenType.MINUS):
            op = self._advance().type
            rhs = self._parse_term()
            value = value + rhs if op is TokenType.PLUS else value - rhs
        return value

    def _parse_term(self) -> float:
        """Parse a multiplicative term: `factor (('*' | '/') factor)*`."""
        value = self._parse_factor()
        while self._current.type in (TokenType.STAR, TokenType.SLASH):
            op = self._advance().type
            rhs = self._parse_factor()
            if op is TokenType.STAR:
                value = value * rhs
            else:
                if rhs == 0.0:
                    raise ValueError("Division by zero")
                value = value / rhs
        return value

    def _parse_factor(self) -> float:
        """Parse a factor with an optional unary sign, e.g. `-3`, `+(2+1)`."""
        if self._current.type is TokenType.MINUS:
            self._advance()
            return -self._parse_factor()
        if self._current.type is TokenType.PLUS:
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a primary: a NUMBER literal or a parenthesised expression."""
        tok = self._current

        if tok.type is TokenType.NUMBER:
            self._advance()
            return float(tok.value)

        if tok.type is TokenType.LPAREN:
            self._advance()
            value = self._parse_expression()
            if self._current.type is not TokenType.RPAREN:
                cur = self._current
                if cur.type is TokenType.EOF:
                    raise ValueError(
                        "Mismatched parentheses: missing closing ')' "
                        f"for '(' at position {tok.position}"
                    )
                raise ValueError(
                    f"Expected ')' but found {cur.value!r} at position {cur.position}"
                )
            self._advance()
            return value

        if tok.type is TokenType.RPAREN:
            raise ValueError(
                f"Mismatched parentheses: unexpected ')' at position {tok.position}"
            )

        if tok.type is TokenType.EOF:
            raise ValueError("Unexpected end of expression: expected a number or '('")

        raise ValueError(
            f"Expected a number or '(' but found {tok.value!r} "
            f"at position {tok.position}"
        )


if __name__ == "__main__":
    evaluator = ExpressionEvaluator()
    for example in ["2 + 3 * 4", "(2 + 3) * 4", "-(2 + 1)", "3.14 * 2"]:
        print(f"{example} = {evaluator.evaluate(example)}")
```

## `test_evaluator.py`

```python
"""Pytest tests for the ExpressionEvaluator."""

import pytest

from evaluator import ExpressionEvaluator


def test_basic_arithmetic() -> None:
    """Basic +, -, *, / operations, including floating point."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3") == pytest.approx(5.0)
    assert ev.evaluate("10 - 4") == pytest.approx(6.0)
    assert ev.evaluate("6 * 7") == pytest.approx(42.0)
    assert ev.evaluate("8 / 2") == pytest.approx(4.0)
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_operator_precedence() -> None:
    """Multiplication and division bind tighter than addition and subtraction."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == pytest.approx(14.0)
    assert ev.evaluate("10 - 2 * 3") == pytest.approx(4.0)
    assert ev.evaluate("10 / 2 - 1") == pytest.approx(4.0)
    assert ev.evaluate("2 - 3 - 1") == pytest.approx(-2.0)  # left-associative
    assert ev.evaluate("8 / 4 / 2") == pytest.approx(1.0)   # left-associative


def test_parentheses() -> None:
    """Parentheses override the default precedence."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == pytest.approx(20.0)
    assert ev.evaluate("2 * (3 + 4)") == pytest.approx(14.0)
    assert ev.evaluate("((2 + 3) * 4) - 6") == pytest.approx(14.0)
    assert ev.evaluate("3 + (4 * (2 + 1))") == pytest.approx(15.0)


def test_unary_minus() -> None:
    """Unary minus applies to a factor, e.g. -3, -(2+1)."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == pytest.approx(-3.0)
    assert ev.evaluate("-(2 + 1)") == pytest.approx(-3.0)
    assert ev.evaluate("5 + -3") == pytest.approx(2.0)
    assert ev.evaluate("--3") == pytest.approx(3.0)
    assert ev.evaluate("-3 * 2") == pytest.approx(-6.0)
    assert ev.evaluate("2 * -3") == pytest.approx(-6.0)


def test_error_cases() -> None:
    """Empty expressions, mismatched parentheses, division by zero, invalid tokens."""
    ev = ExpressionEvaluator()

    # Empty expressions
    with pytest.raises(ValueError, match="empty"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="empty"):
        ev.evaluate("   ")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")      # missing closing paren
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("2 + 3)")      # extra closing paren
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3))")    # extra closing paren
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("()")          # empty parens

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / (2 - 2)")

    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + @")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 & 3")
    with pytest.raises(ValueError):
        ev.evaluate("hello")
```

## How it works

- **Tokenizer** (`_tokenize`) splits the input into typed tokens, tracking each token's position so error messages can point at the offending character.
- **Recursive descent grammar**: `_parse_expression` (additive) → `_parse_term` (multiplicative) → `_parse_factor` (unary sign) → `_parse_primary` (number or parenthesised expression). Precedence and left-associativity fall out naturally from the nesting of the levels.
- **Errors**: mismatched parentheses are caught both where a `)` is expected (`"(2 + 3"`) and where a stray `)` appears after the top-level parse (`"2 + 3)"`); invalid tokens and empty expressions are raised at tokenize/entry time; division by zero is checked in `_parse_term` before dividing.

## Running the tests

```bash
pytest test_evaluator.py -v
```

Two small notes:

- Floating-point numbers are limited to `digits[.digits]` (e.g. `3.14`); scientific notation like `1e5` is intentionally not supported and will raise an `Invalid token` error.
- `evaluate()` raises `TypeError` for non-string input (not `ValueError`), since that's a programming error rather than a bad expression — everything else is `ValueError` as required.