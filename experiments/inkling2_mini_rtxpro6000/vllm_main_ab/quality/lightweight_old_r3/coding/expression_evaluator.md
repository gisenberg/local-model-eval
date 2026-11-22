Here's a complete implementation. It uses a two-phase pipeline — a tokenizer that turns the source string into tokens, and a recursive descent parser that evaluates them — so no `eval()` or `ast` tricks are involved.

## `expression_evaluator.py`

```python
"""Recursive descent mathematical expression evaluator.

Supports +, -, *, / (with standard precedence), parentheses, unary minus
(and plus), and floating-point numbers.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass(frozen=True)
class Token:
    """A lexical token produced by the tokenizer.

    Attributes:
        type: One of 'NUMBER', 'OP', 'LPAREN', 'RPAREN'.
        value: The raw text of the token.
        position: Index of the token's first character in the source string.
    """

    type: Literal['NUMBER', 'OP', 'LPAREN', 'RPAREN']
    value: str
    position: int


class Tokenizer:
    """Converts a raw expression string into a list of tokens."""

    def __init__(self, text: str) -> None:
        """Initialize the tokenizer with the source text."""
        self._text = text
        self._pos = 0

    def tokenize(self) -> List[Token]:
        """Tokenize the full source text.

        Returns:
            A list of tokens in source order.

        Raises:
            ValueError: If an unrecognized character or malformed number is found.
        """
        tokens: List[Token] = []
        while self._pos < len(self._text):
            ch = self._text[self._pos]
            if ch.isspace():
                self._pos += 1
            elif ch in '0123456789.':
                tokens.append(self._read_number())
            elif ch in '+-*/':
                tokens.append(Token('OP', ch, self._pos))
                self._pos += 1
            elif ch == '(':
                tokens.append(Token('LPAREN', ch, self._pos))
                self._pos += 1
            elif ch == ')':
                tokens.append(Token('RPAREN', ch, self._pos))
                self._pos += 1
            else:
                raise ValueError(f"Invalid token {ch!r} at position {self._pos}")
        return tokens

    def _read_number(self) -> Token:
        """Read a numeric literal starting at the current position.

        Returns:
            A NUMBER token.

        Raises:
            ValueError: If the literal contains no digits or multiple decimal points.
        """
        start = self._pos
        seen_dot = False
        while self._pos < len(self._text):
            ch = self._text[self._pos]
            if ch in '0123456789':
                self._pos += 1
            elif ch == '.':
                if seen_dot:
                    raise ValueError(
                        f"Invalid number at position {start}: unexpected '.' "
                        "(multiple decimal points)"
                    )
                seen_dot = True
                self._pos += 1
            else:
                break
        text = self._text[start:self._pos]
        if not any(c in '0123456789' for c in text):
            raise ValueError(f"Invalid number {text!r} at position {start}")
        return Token('NUMBER', text, start)


class Parser:
    """Recursive descent parser for the expression grammar."""

    def __init__(self, tokens: List[Token]) -> None:
        """Initialize the parser with a token list."""
        self._tokens = tokens
        self._pos = 0

    def parse(self) -> float:
        """Parse a complete expression.

        Returns:
            The numeric value of the expression.

        Raises:
            ValueError: If the token list is malformed (mismatched parentheses,
                unexpected tokens, or missing operands).
        """
        value = self._parse_expression()
        if self._pos < len(self._tokens):
            tok = self._tokens[self._pos]
            if tok.type == 'RPAREN':
                raise ValueError(
                    f"Mismatched parentheses: unexpected ')' at position {tok.position}"
                )
            raise ValueError(
                f"Unexpected token {tok.value!r} at position {tok.position}: "
                "expected an operator or end of expression"
            )
        return value

    def _peek(self) -> Optional[Token]:
        """Return the current token without consuming it, or None if exhausted."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> Token:
        """Consume and return the current token."""
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _parse_expression(self) -> float:
        """expression := term (('+' | '-') term)*  (lowest precedence)."""
        value = self._parse_term()
        while True:
            tok = self._peek()
            if tok is None or tok.type != 'OP' or tok.value not in ('+', '-'):
                break
            self._advance()
            rhs = self._parse_term()
            value = value + rhs if tok.value == '+' else value - rhs
        return value

    def _parse_term(self) -> float:
        """term := factor (('*' | '/') factor)*  (higher precedence)."""
        value = self._parse_factor()
        while True:
            tok = self._peek()
            if tok is None or tok.type != 'OP' or tok.value not in ('*', '/'):
                break
            self._advance()
            rhs = self._parse_factor()
            if tok.value == '*':
                value = value * rhs
            else:
                if rhs == 0:
                    raise ValueError(f"Division by zero at position {tok.position}")
                value = value / rhs
        return value

    def _parse_factor(self) -> float:
        """factor := '-' factor | '+' factor | primary  (unary)."""
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression: expected a number or '('")
        if tok.type == 'OP' and tok.value == '-':
            self._advance()
            return -self._parse_factor()
        if tok.type == 'OP' and tok.value == '+':
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """primary := NUMBER | '(' expression ')'  (highest precedence)."""
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression: expected a number or '('")
        if tok.type == 'NUMBER':
            self._advance()
            return float(tok.value)
        if tok.type == 'LPAREN':
            self._advance()
            value = self._parse_expression()
            closing = self._peek()
            if closing is None:
                raise ValueError(
                    f"Mismatched parentheses: missing closing ')' "
                    f"for '(' at position {tok.position}"
                )
            if closing.type != 'RPAREN':
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found "
                    f"{closing.value!r} at position {closing.position}"
                )
            self._advance()
            return value
        raise ValueError(
            f"Unexpected token {tok.value!r} at position {tok.position}: "
            "expected a number or '('"
        )


class ExpressionEvaluator:
    """Evaluates mathematical expressions via a recursive descent parser.

    Example:
        >>> ev = ExpressionEvaluator()
        >>> ev.evaluate("2 + 3 * 4")
        14.0
    """

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression.

        Args:
            expr: The expression to evaluate, e.g. "2 + 3 * 4" or "(1.5 + 2.5) * 2".

        Returns:
            The numeric result of the expression as a float.

        Raises:
            TypeError: If expr is not a string.
            ValueError: If the expression is empty, contains invalid tokens,
                has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"Expression must be a string, got {type(expr).__name__}")
        if not expr.strip():
            raise ValueError("Empty expression")
        tokens = Tokenizer(expr).tokenize()
        if not tokens:
            raise ValueError("Empty expression")
        return Parser(tokens).parse()
```

## `test_expression_evaluator.py`

```python
"""Tests for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Provide a fresh evaluator for each test."""
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3") == pytest.approx(5.0)
    assert evaluator.evaluate("10 - 4") == pytest.approx(6.0)
    assert evaluator.evaluate("6 * 7") == pytest.approx(42.0)
    assert evaluator.evaluate("8 / 2") == pytest.approx(4.0)
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_operator_precedence(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3 * 4") == pytest.approx(14.0)
    assert evaluator.evaluate("10 - 4 / 2") == pytest.approx(8.0)
    assert evaluator.evaluate("2 * 3 + 4 * 5") == pytest.approx(26.0)


def test_parentheses(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("(2 + 3) * 4") == pytest.approx(20.0)
    assert evaluator.evaluate("2 * (3 + 4)") == pytest.approx(14.0)
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == pytest.approx(15.0)


def test_unary_minus(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("-3") == pytest.approx(-3.0)
    assert evaluator.evaluate("-(2 + 1)") == pytest.approx(-3.0)
    assert evaluator.evaluate("3 - -2") == pytest.approx(5.0)
    assert evaluator.evaluate("-2 * 3") == pytest.approx(-6.0)


def test_error_cases(evaluator: ExpressionEvaluator) -> None:
    # Empty expressions
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (1 - 1)")
    # Invalid tokens / numbers
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate(".")
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("3.14.15")
```

## Running the tests

```bash
pytest test_expression_evaluator.py -v
```

## Design notes

- **Precedence** comes from the grammar structure, not from any trickery: `expression` → `term` → `factor` → `primary`, with `*`/`/` handled one level down from `+`/`-`. This is the standard recursive descent approach.
- **Unary minus** is a dedicated grammar rule (`factor := '-' factor | '+' factor | primary`), which naturally supports chained unary operators like `--3` and `-(2+1)`.
- **Errors** are raised at the exact point of failure with position information: the tokenizer flags unknown characters and malformed numbers (e.g., `.` or `3.14.15`); the parser flags missing/unexpected closing parens, unexpected trailing tokens, and division by zero (checked explicitly so the message is a clear `ValueError`, not a raw `ZeroDivisionError`).
- **Edge cases handled:** `.5` and `5.` are valid numbers; whitespace is skipped everywhere; `TypeError` is raised for non-string input (per standard Python practice, separate from the required `ValueError` cases).

One caveat to be aware of: since this evaluates with native floats, results like `0.1 + 0.2` will reflect normal IEEE 754 rounding — if you need exact decimal arithmetic (e.g., for money), you'd want to swap the float operations for `decimal.Decimal` in the parser.