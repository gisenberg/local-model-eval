Here's a complete solution. I'll provide the evaluator module and a test suite using a two-pass design: a small tokenizer, then a recursive descent parser with the grammar `expression → term → factor → primary`, which gives correct precedence (`*`/`/` bind tighter than `+`/`-`) and handles parentheses naturally.

## `expression_evaluator.py`

```python
"""Recursive descent mathematical expression evaluator.

Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?|\.\d+")


@dataclass
class Token:
    """A lexical token produced by the tokenizer."""

    kind: str        # 'number' | 'op' | 'lparen' | 'rparen'
    text: str        # raw source text of the token
    position: int    # character offset of the token in the source string


class _Parser:
    """Recursive descent parser over a token stream.

    Grammar:
        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := ('-' | '+')* primary
        primary    := number | '(' expression ')'
    """

    def __init__(self, tokens: List[Token]) -> None:
        """Store the token stream and initialize the cursor."""
        self._tokens = tokens
        self._index = 0

    def at_end(self) -> bool:
        """Return True when the token stream is exhausted."""
        return self._index >= len(self._tokens)

    def peek(self) -> Optional[Token]:
        """Return the current token without consuming it (None at end of stream)."""
        if self.at_end():
            return None
        return self._tokens[self._index]

    def advance(self) -> Token:
        """Consume and return the current token."""
        token = self._tokens[self._index]
        self._index += 1
        return token

    def parse_expression(self) -> float:
        """Parse a sum/difference chain and return its value."""
        value = self.parse_term()
        token = self.peek()
        while token is not None and token.kind == "op" and token.text in ("+", "-"):
            self.advance()
            right = self.parse_term()
            value = value + right if token.text == "+" else value - right
            token = self.peek()
        return value

    def parse_term(self) -> float:
        """Parse a product/quotient chain and return its value."""
        value = self.parse_factor()
        token = self.peek()
        while token is not None and token.kind == "op" and token.text in ("*", "/"):
            self.advance()
            right = self.parse_factor()
            if token.text == "*":
                value = value * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                value = value / right
            token = self.peek()
        return value

    def parse_factor(self) -> float:
        """Parse unary sign(s) followed by a primary and return its value."""
        token = self.peek()
        if token is not None and token.kind == "op" and token.text == "-":
            self.advance()
            return -self.parse_factor()
        if token is not None and token.kind == "op" and token.text == "+":
            self.advance()
            return self.parse_factor()
        return self.parse_primary()

    def parse_primary(self) -> float:
        """Parse a number or a parenthesized expression and return its value."""
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of expression: expected a number or '('")
        if token.kind == "number":
            self.advance()
            return float(token.text)
        if token.kind == "lparen":
            self.advance()
            value = self.parse_expression()
            closing = self.peek()
            if closing is None or closing.kind != "rparen":
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self.advance()
            return value
        if token.kind == "rparen":
            raise ValueError(
                f"Mismatched parentheses: unexpected ')' at position {token.position}"
            )
        raise ValueError(f"Unexpected token '{token.text}' at position {token.position}")


class ExpressionEvaluator:
    """Evaluates arithmetic expressions (+, -, *, /, parentheses, unary minus, floats)."""

    def evaluate(self, expr: str) -> float:
        """Evaluate an arithmetic expression.

        Args:
            expr: The expression to evaluate, e.g. '3.14 * (2 + 1)'.

        Returns:
            The numeric result of the expression.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens, has
                mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise ValueError(f"Expression must be a string, got {type(expr).__name__}")
        if not expr.strip():
            raise ValueError("Expression is empty")

        tokens = self._tokenize(expr)
        parser = _Parser(tokens)
        result = parser.parse_expression()

        leftover = parser.peek()
        if leftover is not None:
            if leftover.kind == "rparen":
                raise ValueError(
                    f"Mismatched parentheses: unexpected ')' at position {leftover.position}"
                )
            raise ValueError(
                f"Unexpected token '{leftover.text}' at position {leftover.position}"
            )
        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Split a source string into tokens.

        Args:
            expr: The source expression (assumed to be non-empty).

        Returns:
            The ordered list of tokens.

        Raises:
            ValueError: If a character cannot be recognized.
        """
        tokens: List[Token] = []
        index = 0
        while index < len(expr):
            char = expr[index]
            if char.isspace():
                index += 1
                continue
            if char in "+-*/":
                tokens.append(Token("op", char, index))
                index += 1
                continue
            if char == "(":
                tokens.append(Token("lparen", char, index))
                index += 1
                continue
            if char == ")":
                tokens.append(Token("rparen", char, index))
                index += 1
                continue
            match = _NUMBER_PATTERN.match(expr, index)
            if match:
                tokens.append(Token("number", match.group(), index))
                index = match.end()
                continue
            raise ValueError(f"Invalid token '{char}' at position {index}")
        return tokens
```

## `test_expression_evaluator.py`

```python
"""Pytest tests for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


def test_basic_arithmetic() -> None:
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3") == pytest.approx(5.0)
    assert ev.evaluate("10 - 4") == pytest.approx(6.0)
    assert ev.evaluate("6 * 7") == pytest.approx(42.0)
    assert ev.evaluate("8 / 2") == pytest.approx(4.0)
    assert ev.evaluate("3.5 + 1.25") == pytest.approx(4.75)


def test_operator_precedence() -> None:
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == pytest.approx(14.0)
    assert ev.evaluate("10 - 6 / 2") == pytest.approx(7.0)
    assert ev.evaluate("2 * 3 + 4 * 5") == pytest.approx(26.0)


def test_parentheses() -> None:
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == pytest.approx(20.0)
    assert ev.evaluate("((1 + 2) * 3) - 1") == pytest.approx(8.0)
    assert ev.evaluate("2 * (3 + (4 * 5))") == pytest.approx(46.0)


def test_unary_minus() -> None:
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == pytest.approx(-3.0)
    assert ev.evaluate("-(2 + 1)") == pytest.approx(-3.0)
    assert ev.evaluate("3 + -2") == pytest.approx(1.0)
    assert ev.evaluate("--3") == pytest.approx(3.0)


@pytest.mark.parametrize(
    "expr, match",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("(3 + 4", "missing closing"),
        ("3 + 4)", r"unexpected '\)'"),
        ("3 & 4", "Invalid token"),
        ("1 / 0", "Division by zero"),
        ("3 +", "Unexpected end"),
    ],
)
def test_error_cases(expr: str, match: str) -> None:
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match=match):
        ev.evaluate(expr)
```

## Notes

- **Precedence** comes from the grammar structure: `parse_expression` handles `+`/`-`, and delegates to `parse_term` for `*`/`/`, so multiplication and division are always resolved first — no explicit precedence-climbing needed.
- **Unary minus** is handled in `parse_factor`, which recursively consumes sign(s) before calling `parse_primary` — this also supports `--3`, `-(2+1)`, and `3 + -2` for free.
- **Errors** are raised with specific messages: mismatched parens (`missing closing ')'` / `unexpected ')'`), division by zero, invalid tokens (with position), and empty expressions (caught before tokenization). Parenthesized division by zero (e.g., `(1/0)`) works because the exception propagates through the recursion.
- Floating-point results are compared with `pytest.approx` to avoid exact binary-representation mismatches.

Run with:

```bash
pytest test_expression_evaluator.py -v
```

The parametrized error test expands into 7 test items, so the suite gives 5 test functions / 11 test items total.