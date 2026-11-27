# Mathematical Expression Evaluator

Below is a complete implementation using a tokenizer + recursive descent parser (no `eval()`).

## `expression_evaluator.py`

```python
"""A recursive-descent mathematical expression evaluator.

Supports +, -, *, / with standard precedence, parentheses, unary minus/plus,
and floating-point literals. Raises ValueError with descriptive messages for
empty expressions, mismatched parentheses, division by zero, and invalid tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    """A single lexical token produced by the tokenizer."""

    type: str      # 'NUMBER' | 'OP' | 'LPAREN' | 'RPAREN' | 'END'
    value: str     # literal text of the token ('' for END)
    position: int  # offset of the token in the source expression


class ExpressionEvaluator:
    """Evaluate arithmetic expressions without using eval().

    Supported syntax:
        - Binary operators: +, -, *, /  (* and / have higher precedence)
        - Parentheses for grouping
        - Unary minus and plus (e.g. '-3', '+3', '-(2+1)')
        - Floating-point literals (e.g. '3', '3.14', '3.', '.5')

    Raises:
        ValueError: for empty expressions, invalid tokens, mismatched
            parentheses, or division by zero.
    """

    def __init__(self) -> None:
        """Initialize the evaluator; parser state is reset on each evaluate() call."""
        self._tokens: list[Token] = []
        self._pos: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression.

        Args:
            expr: The expression to evaluate, e.g. '2 + 3.5 * (1 - 2)'.

        Returns:
            The numeric result of the expression.

        Raises:
            TypeError: If expr is not a string.
            ValueError: If expr is empty, contains invalid tokens, has
                mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"Expression must be a string, got {type(expr).__name__}")

        if not expr.strip():
            raise ValueError("Empty expression: expected a mathematical expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        if self._current.type != "END":
            if self._current.type == "RPAREN":
                raise ValueError(
                    f"Mismatched parentheses: unexpected ')' at position {self._current.position}"
                )
            raise ValueError(
                f"Invalid token at position {self._current.position}: "
                f"unexpected {self._describe(self._current)} after expression"
            )

        return result

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _tokenize(self, expr: str) -> list[Token]:
        """Convert a raw expression into a list of tokens (plus a trailing END token)."""
        tokens: list[Token] = []
        i = 0
        while i < len(expr):
            ch = expr[i]

            if ch.isspace():
                i += 1
                continue

            if ch.isdigit() or ch == ".":
                match = re.match(r"\d+(\.\d*)?|\.\d+", expr[i:])
                if match is None:
                    raise ValueError(f"Invalid token at position {i}: '{ch}'")
                tokens.append(Token("NUMBER", match.group(), i))
                i += match.end()
                continue

            if ch in "+-*/":
                tokens.append(Token("OP", ch, i))
                i += 1
                continue

            if ch == "(":
                tokens.append(Token("LPAREN", ch, i))
                i += 1
                continue

            if ch == ")":
                tokens.append(Token("RPAREN", ch, i))
                i += 1
                continue

            raise ValueError(f"Invalid token at position {i}: '{ch}'")

        tokens.append(Token("END", "", len(expr)))
        return tokens

    # ------------------------------------------------------------------
    # Parser state helpers
    # ------------------------------------------------------------------

    @property
    def _current(self) -> Token:
        """Return the token the parser is currently looking at."""
        return self._tokens[self._pos]

    def _advance(self) -> None:
        """Move the parser forward, never past the END token."""
        if self._pos < len(self._tokens) - 1:
            self._pos += 1

    def _describe(self, token: Token) -> str:
        """Return a human-readable description of a token for error messages."""
        if token.type == "END":
            return "end of expression"
        if token.type == "NUMBER":
            return f"number '{token.value}'"
        return f"'{token.value}'"

    # ------------------------------------------------------------------
    # Recursive descent parser
    # Grammar:
    #   expression := term (('+' | '-') term)*
    #   term       := factor (('*' | '/') factor)*
    #   factor     := ('-' | '+')* primary
    #   primary    := NUMBER | '(' expression ')'
    # ------------------------------------------------------------------

    def _parse_expression(self) -> float:
        """Parse an expression (addition/subtraction level)."""
        left = self._parse_term()
        while self._current.type == "OP" and self._current.value in ("+", "-"):
            op = self._current.value
            self._advance()
            right = self._parse_term()
            left = left + right if op == "+" else left - right
        return left

    def _parse_term(self) -> float:
        """Parse a term (multiplication/division level)."""
        left = self._parse_factor()
        while self._current.type == "OP" and self._current.value in ("*", "/"):
            op = self._current.value
            self._advance()
            right = self._parse_factor()
            if op == "*":
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
        return left

    def _parse_factor(self) -> float:
        """Parse a factor: zero or more unary '-' / '+' followed by a primary."""
        if self._current.type == "OP" and self._current.value == "-":
            self._advance()
            return -self._parse_factor()
        if self._current.type == "OP" and self._current.value == "+":
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a primary: a number literal or a parenthesized expression."""
        token = self._current

        if token.type == "NUMBER":
            self._advance()
            return float(token.value)

        if token.type == "LPAREN":
            self._advance()
            value = self._parse_expression()
            if self._current.type != "RPAREN":
                raise ValueError(
                    "Mismatched parentheses: expected ')' but found "
                    + self._describe(self._current)
                )
            self._advance()
            return value

        raise ValueError(
            f"Invalid token at position {token.position}: expected a number or '(' "
            f"but found {self._describe(token)}"
        )


if __name__ == "__main__":
    evaluator = ExpressionEvaluator()
    demo = ["2 + 3 * 4", "(2 + 3) * 4", "-(2 + 1)", "3.14 * 2", "10 / 4", "1 + 2 * -3"]
    for expression in demo:
        print(f"{expression!r:>16} = {evaluator.evaluate(expression)}")
```

## `test_expression_evaluator.py`

```python
"""Pytest tests for the ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


def test_basic_arithmetic() -> None:
    """Basic +, -, *, / operations and floating-point literals."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3") == 5.0
    assert ev.evaluate("10 - 4") == 6.0
    assert ev.evaluate("3 * 4") == 12.0
    assert ev.evaluate("8 / 2") == 4.0
    assert ev.evaluate("10 / 4") == 2.5
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_operator_precedence() -> None:
    """* and / bind more tightly than + and -; operators are left-associative."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("2 * 3 + 4") == 10.0
    assert ev.evaluate("10 - 4 - 3") == 3.0
    assert ev.evaluate("10 / 2 / 5") == 1.0
    assert ev.evaluate("2 + 3 * 4 - 5") == 9.0


def test_parentheses() -> None:
    """Parentheses override the default operator precedence."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("((2 + 3))") == 5.0
    assert ev.evaluate("(2 + 3) * (4 - 1)") == 15.0
    assert ev.evaluate("2 * (3 + 4)") == 14.0


def test_unary_minus() -> None:
    """Unary minus applies to the immediately following value (including groups)."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("--3") == 3.0
    assert ev.evaluate("2 * -3") == -6.0
    assert ev.evaluate("-3 + 5") == 2.0
    assert ev.evaluate("-3.14") == pytest.approx(-3.14)


def test_error_cases() -> None:
    """Empty expressions, mismatched parentheses, division by zero, invalid tokens."""
    ev = ExpressionEvaluator()

    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")

    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("1 + 2)")

    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")

    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 @ 3")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 +")
```

## How it works

- **Tokenizer** (`_tokenize`): splits the input into `NUMBER`, `OP`, `LPAREN`, `RPAREN`, and a trailing `END` token. Anything unrecognized (e.g. `@`) raises `ValueError("Invalid token at position N: ...")` immediately.
- **Grammar** (recursive descent): `expression → term → factor → primary`. Each level handles one concern, which gives correct precedence for free (`*`/`/` bind tighter than `+`/`-`), left-associativity via the loops, and unary minus via recursion in `factor`.
- **Errors**: unmatched `)` is caught in `_parse_primary` (missing `)`) and in `evaluate` (trailing `)`); division by zero is checked explicitly; a trailing unconsumed token or a missing operand raises an invalid-token error.

## Running the tests

```bash
pip install pytest
python -m pytest test_expression_evaluator.py -v
```

And to try the evaluator directly:

```bash
python expression_evaluator.py
```

Note: `evaluate` raises `TypeError` (rather than `ValueError`) for non-string input, since that's a type mismatch rather than a malformed expression — everything else raises `ValueError` as required.