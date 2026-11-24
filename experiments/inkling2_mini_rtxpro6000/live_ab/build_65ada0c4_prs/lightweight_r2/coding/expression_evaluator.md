Here's a complete implementation. I split it into two files: the evaluator and its pytest suite.

## `expression_evaluator.py`

```python
"""Recursive descent mathematical expression evaluator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    """A lexical token produced by the tokenizer.

    Attributes:
        kind: Token type — one of 'number', 'op', 'lparen', 'rparen', 'eof'.
        value: The raw text of the token (empty string for 'eof').
        position: Index in the source expression where the token starts.
    """
    kind: str
    value: str
    position: int


class ExpressionEvaluator:
    """Evaluates arithmetic expressions using a recursive descent parser.

    Supports +, -, *, /, parentheses, unary minus (and unary plus), and
    floating point numbers. Does not use eval() or ast.literal_eval().

    Grammar (lowest precedence first):

        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := ('+' | '-')* primary
        primary    := number | '(' expression ')'
    """

    def __init__(self) -> None:
        self._tokens: list[Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression.

        Args:
            expr: The expression to evaluate, e.g. '3.14 * (2 + 1)'.

        Returns:
            The numeric result of the expression.

        Raises:
            TypeError: If expr is not a string.
            ValueError: If the expression is empty, contains invalid tokens,
                has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"Expression must be a string, got {type(expr).__name__}")
        if not expr.strip():
            raise ValueError("Empty expression: cannot evaluate an empty or whitespace-only string")

        self._tokens = self._tokenize(expr)
        self._pos = 0
        result = self._parse_expression()

        # Anything left over is an unmatched ')' or a stray token.
        if self._peek().kind == "rparen":
            token = self._peek()
            raise ValueError(f"Mismatched parentheses: unexpected ')' at position {token.position}")
        if self._peek().kind != "eof":
            token = self._peek()
            raise ValueError(f"Unexpected token '{token.value}' at position {token.position}")
        return float(result)

    # ---- Tokenizer ----

    def _tokenize(self, expr: str) -> list[Token]:
        """Split *expr* into tokens, raising ValueError on invalid characters."""
        tokens: list[Token] = []
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
                    i += 1
                text = expr[start:i]
                if dot_count > 1:
                    raise ValueError(f"Invalid number '{text}' at position {start}")
                try:
                    float(text)
                except ValueError:
                    raise ValueError(f"Invalid number '{text}' at position {start}") from None
                tokens.append(Token("number", text, start))
                continue
            if ch in "+-*/":
                tokens.append(Token("op", ch, i))
                i += 1
                continue
            if ch == '(':
                tokens.append(Token("lparen", ch, i))
                i += 1
                continue
            if ch == ')':
                tokens.append(Token("rparen", ch, i))
                i += 1
                continue
            raise ValueError(f"Invalid token '{ch}' at position {i}")
        tokens.append(Token("eof", "", n))
        return tokens

    # ---- Parser helpers ----

    def _peek(self) -> Token:
        """Return the current token without consuming it."""
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        """Consume the current token and return it."""
        token = self._tokens[self._pos]
        if token.kind != "eof":
            self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse expr := term (('+' | '-') term)*  (lowest precedence)."""
        value = self._parse_term()
        while self._peek().kind == "op" and self._peek().value in ("+", "-"):
            op = self._advance().value
            rhs = self._parse_term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def _parse_term(self) -> float:
        """Parse term := factor (('*' | '/') factor)*  (higher precedence)."""
        value = self._parse_factor()
        while self._peek().kind == "op" and self._peek().value in ("*", "/"):
            op = self._advance().value
            rhs = self._parse_factor()
            if op == "*":
                value = value * rhs
            else:
                if rhs == 0:
                    raise ValueError("Division by zero")
                value = value / rhs
        return value

    def _parse_factor(self) -> float:
        """Parse factor := ('+' | '-')* primary  (handles unary operators)."""
        token = self._peek()
        if token.kind == "op" and token.value == "-":
            self._advance()
            return -self._parse_factor()
        if token.kind == "op" and token.value == "+":
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse primary := number | '(' expression ')'."""
        token = self._peek()
        if token.kind == "number":
            self._advance()
            return float(token.value)
        if token.kind == "lparen":
            self._advance()
            value = self._parse_expression()
            closing = self._peek()
            if closing.kind != "rparen":
                if closing.kind == "eof":
                    raise ValueError("Mismatched parentheses: expected ')' but reached end of expression")
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found '{closing.value}' "
                    f"at position {closing.position}"
                )
            self._advance()
            return value
        if token.kind == "rparen":
            raise ValueError(f"Mismatched parentheses: unexpected ')' at position {token.position}")
        if token.kind == "eof":
            raise ValueError("Unexpected end of expression")
        raise ValueError(f"Unexpected token '{token.value}' at position {token.position}")
```

## `test_expression_evaluator.py`

```python
"""Tests for the ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


def test_basic_arithmetic():
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3") == 5.0
    assert ev.evaluate("10 - 4") == 6.0
    assert ev.evaluate("6 * 7") == 42.0
    assert ev.evaluate("8 / 2") == 4.0
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate("0.5 + 0.25") == pytest.approx(0.75)


def test_operator_precedence():
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("2 * 3 + 4") == 10.0
    assert ev.evaluate("10 / 2 / 5") == 1.0        # left-associative
    assert ev.evaluate("100 - 50 + 10") == 60.0     # left-associative


def test_parentheses():
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("2 * (3 + 4)") == 14.0
    assert ev.evaluate("((1 + 2) * (3 + 4))") == 21.0


def test_unary_minus():
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("2 * -3") == -6.0
    assert ev.evaluate("3 - -2") == 5.0
    assert ev.evaluate("- -3") == 3.0


def test_error_cases():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(1 + 2")          # unclosed
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("1 + 2)")          # unmatched ')'
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + @")
    with pytest.raises(ValueError, match="Invalid number"):
        ev.evaluate("1.2.3")
```

## How it works

- **Tokenizer** (`_tokenize`): walks the string, emitting `number` / `op` / `lparen` / `rparen` / `eof` tokens. Invalid characters and malformed numbers (`1.2.3`, `.`) raise `ValueError` with the position.
- **Recursive descent parser**: one method per grammar rule. `_parse_expression` handles `+`/`-`, `_parse_term` handles `*`/`/`, `_parse_factor` handles unary `+`/`-`, and `_parse_primary` handles numbers and parenthesized sub-expressions. Precedence and left-associativity fall out naturally from the layered rules.
- **Errors**: all reported as `ValueError` with a descriptive message (e.g. `"Mismatched parentheses: unexpected ')' at position 4"`). Division by zero is checked explicitly since Python raises a bare `ZeroDivisionError` otherwise.

Run the tests with:

```bash
pytest test_expression_evaluator.py -v
```

One design note: like most recursive-descent parsers, deep nesting (e.g. thousands of nested parens) can hit Python's recursion limit — for typical expressions this is not a concern, but a shunting-yard/Pratt parser would be the fix if you need to support pathological input.