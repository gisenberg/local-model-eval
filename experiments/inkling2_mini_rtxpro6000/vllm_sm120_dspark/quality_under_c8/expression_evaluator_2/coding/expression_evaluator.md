Here's a complete implementation. The design is a two-phase recursive descent parser: a lexer that tokenizes the expression, then a set of mutually recursive parse functions implementing the grammar:

```
expression := term (('+' | '-') term)*
term        := factor (('*' | '/') factor)*
factor      := ('+' | '-')* primary
primary     := NUMBER | '(' expression ')'
```

## `expression_evaluator.py`

```python
"""Recursive descent mathematical expression evaluator."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class _TokenType(Enum):
    """Token types produced by the lexer."""
    NUMBER = "NUMBER"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"


@dataclass(frozen=True)
class _Token:
    """A lexical token with its source position."""
    type: _TokenType
    value: str
    position: int


class ExpressionEvaluator:
    """Evaluates arithmetic expressions with +, -, *, /, parentheses, unary minus, and floats.

    Raises ValueError with a descriptive message for empty expressions, invalid
    tokens, mismatched parentheses, unexpected input, and division by zero.
    """

    def __init__(self) -> None:
        """Initialize empty parser state (reset on each evaluate call)."""
        self._tokens: List[_Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate *expr* and return the result as a float.

        Args:
            expr: The arithmetic expression to evaluate.

        Returns:
            The numeric result of the expression.

        Raises:
            TypeError: If *expr* is not a string.
            ValueError: If the expression is empty, malformed, or divides by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"Expression must be a string, got {type(expr).__name__}")
        if not expr.strip():
            raise ValueError("Expression is empty or whitespace")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        remaining = self._peek()
        if remaining is not None:
            if remaining.type is _TokenType.RPAREN:
                raise ValueError(
                    f"Mismatched parentheses: unexpected ')' at position {remaining.position}"
                )
            raise ValueError(
                f"Unexpected token '{remaining.value}' at position {remaining.position}"
            )
        return result

    # ---- lexer ----

    def _tokenize(self, expr: str) -> List[_Token]:
        """Split *expr* into tokens, raising ValueError on invalid characters."""
        tokens: List[_Token] = []
        single_char = {
            '+': _TokenType.PLUS,
            '-': _TokenType.MINUS,
            '*': _TokenType.STAR,
            '/': _TokenType.SLASH,
            '(': _TokenType.LPAREN,
            ')': _TokenType.RPAREN,
        }
        i, n = 0, len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or ch == '.':
                start = i
                dot_count = 0
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        dot_count += 1
                        if dot_count > 1:
                            raise ValueError(
                                f"Invalid number at position {start}: multiple decimal points"
                            )
                    i += 1
                text = expr[start:i]
                try:
                    float(text)
                except ValueError:
                    raise ValueError(f"Invalid number '{text}' at position {start}") from None
                tokens.append(_Token(_TokenType.NUMBER, text, start))
                continue
            if ch in single_char:
                tokens.append(_Token(single_char[ch], ch, i))
                i += 1
                continue
            raise ValueError(f"Invalid token '{ch}' at position {i}")
        return tokens

    # ---- parser helpers ----

    def _peek(self) -> Optional[_Token]:
        """Return the current token without consuming it (None at end of input)."""
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        """Consume and return the current token."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse an additive expression: term (('+' | '-') term)*."""
        left = self._parse_term()
        while True:
            token = self._peek()
            if token is None or token.type not in (_TokenType.PLUS, _TokenType.MINUS):
                return left
            self._advance()
            right = self._parse_term()
            if token.type is _TokenType.PLUS:
                left += right
            else:
                left -= right

    def _parse_term(self) -> float:
        """Parse a multiplicative expression: factor (('*' | '/') factor)*."""
        left = self._parse_factor()
        while True:
            token = self._peek()
            if token is None or token.type not in (_TokenType.STAR, _TokenType.SLASH):
                return left
            self._advance()
            right = self._parse_factor()
            if token.type is _TokenType.STAR:
                left *= right
            else:
                if right == 0:
                    raise ValueError(f"Division by zero at position {token.position}")
                left /= right

    def _parse_factor(self) -> float:
        """Parse a factor: optional unary sign followed by a primary."""
        token = self._peek()
        if token is not None and token.type is _TokenType.MINUS:
            self._advance()
            return -self._parse_factor()
        if token is not None and token.type is _TokenType.PLUS:
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a primary: a number literal or a parenthesized expression."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        if token.type is _TokenType.NUMBER:
            self._advance()
            return float(token.value)
        if token.type is _TokenType.LPAREN:
            self._advance()
            value = self._parse_expression()
            closing = self._peek()
            if closing is None:
                raise ValueError(
                    "Mismatched parentheses: expected ')' but reached end of expression"
                )
            if closing.type is not _TokenType.RPAREN:
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found '{closing.value}' at position {closing.position}"
                )
            self._advance()
            return value
        raise ValueError(
            f"Expected a number or '(' but found '{token.value}' at position {token.position}"
        )
```

## `test_expression_evaluator.py`

```python
"""Pytest tests for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


def test_basic_arithmetic():
    """Basic +, -, *, / operations and float literals."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3") == pytest.approx(5.0)
    assert ev.evaluate("10 - 4") == pytest.approx(6.0)
    assert ev.evaluate("6 * 7") == pytest.approx(42.0)
    assert ev.evaluate("8 / 2") == pytest.approx(4.0)
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_operator_precedence():
    """Multiplication/division bind tighter than addition/subtraction."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == pytest.approx(14.0)
    assert ev.evaluate("10 - 2 * 3") == pytest.approx(4.0)
    assert ev.evaluate("2 * 3 + 4 * 5") == pytest.approx(26.0)
    assert ev.evaluate("100 / 10 - 2") == pytest.approx(8.0)


def test_parentheses():
    """Parenthesized subexpressions override default precedence."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == pytest.approx(20.0)
    assert ev.evaluate("2 * (3 + 4)") == pytest.approx(14.0)
    assert ev.evaluate("((1 + 2) * (3 + 4))") == pytest.approx(21.0)
    assert ev.evaluate("2 + 3 * (4 + 5)") == pytest.approx(29.0)


def test_unary_minus():
    """Unary minus applies to numbers and parenthesized expressions."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == pytest.approx(-3.0)
    assert ev.evaluate("-(2 + 1)") == pytest.approx(-3.0)
    assert ev.evaluate("2 + -3") == pytest.approx(-1.0)
    assert ev.evaluate("--5") == pytest.approx(5.0)
    assert ev.evaluate("-3.5 * 2") == pytest.approx(-7.0)


def test_error_cases():
    """Empty, mismatched, invalid, and divide-by-zero inputs raise ValueError."""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="empty"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="empty"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("0 / 0")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + @")
    with pytest.raises(ValueError, match="Invalid number"):
        ev.evaluate("1.2.3")
    with pytest.raises(ValueError, match="Unexpected token"):
        ev.evaluate("2 3")
```

## Running the tests

```bash
python -m pytest test_expression_evaluator.py -v
```

A few design notes:

- **Precedence** is handled structurally — `_parse_expression` handles `+`/`-` and calls `_parse_term`, which handles `*`/`/` and calls `_parse_factor`, so no manual precedence tables are needed.
- **Unary minus** lives in `_parse_factor`, which recurses on itself, so it composes with parens and nested signs (`--5`, `-(2+1)`).
- **Errors** all carry position information where useful, and empty/whitespace input is rejected before lexing so the error message stays accurate.
- The evaluator is **not** thread-safe by design (it holds token state on `self`); create one instance per evaluation if you need concurrency.