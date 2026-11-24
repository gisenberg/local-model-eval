Here's a complete implementation as two files: `expression_evaluator.py` and `test_expression_evaluator.py`.

## `expression_evaluator.py`

```python
"""Recursive descent mathematical expression evaluator."""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class _Token:
    """A single lexical token produced by the tokenizer."""

    kind: str       # 'number', 'operator', 'lparen', 'rparen'
    value: str      # raw text of the token
    position: int   # index in the source string


class ExpressionEvaluator:
    """Evaluates mathematical expressions via recursive descent parsing.

    Supports +, -, *, /, parentheses, unary minus, and floating-point
    numbers. Raises ValueError with a descriptive message for malformed
    input (empty expressions, invalid tokens, mismatched parentheses,
    and division by zero).
    """

    _NUMBER_RE = re.compile(r'(?:\d+(?:\.\d+)?|\.\d+)')

    def __init__(self) -> None:
        """Initialize an evaluator (all state is rebuilt on each evaluate() call)."""
        self._tokens: List[_Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate *expr* and return the result as a float.

        Raises:
            TypeError: if *expr* is not a string.
            ValueError: if *expr* is empty, contains invalid characters,
                has mismatched parentheses, is otherwise malformed, or
                divides by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"Expression must be a string, got {type(expr).__name__}")
        if not expr.strip():
            raise ValueError("Empty expression: cannot evaluate a blank string")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        remaining = self._peek()
        if remaining is not None:
            if remaining.kind == 'rparen':
                raise ValueError(
                    f"Mismatched parentheses: unexpected ')' at position {remaining.position}"
                )
            raise ValueError(
                f"Unexpected token '{remaining.value}' at position {remaining.position}"
            )

        return result

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    def _tokenize(self, expr: str) -> List[_Token]:
        """Convert the source string into a list of tokens."""
        tokens: List[_Token] = []
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or ch == '.':
                match = self._NUMBER_RE.match(expr, i)
                if match is None:
                    raise ValueError(f"Invalid number at position {i}: '{ch}'")
                tokens.append(_Token('number', match.group(), i))
                i = match.end()
                continue
            if ch in '+-*/':
                tokens.append(_Token('operator', ch, i))
                i += 1
                continue
            if ch == '(':
                tokens.append(_Token('lparen', ch, i))
                i += 1
                continue
            if ch == ')':
                tokens.append(_Token('rparen', ch, i))
                i += 1
                continue
            raise ValueError(f"Invalid character '{ch}' at position {i}")
        return tokens

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
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
            if token is not None and token.kind == 'operator' and token.value in ('+', '-'):
                self._advance()
                right = self._parse_term()
                left = left + right if token.value == '+' else left - right
            else:
                break
        return left

    def _parse_term(self) -> float:
        """Parse a multiplicative term: factor (('*' | '/') factor)*."""
        left = self._parse_factor()
        while True:
            token = self._peek()
            if token is not None and token.kind == 'operator' and token.value in ('*', '/'):
                self._advance()
                right = self._parse_factor()
                if token.value == '*':
                    left = left * right
                else:
                    if right == 0:
                        raise ValueError("Division by zero")
                    left = left / right
            else:
                break
        return left

    def _parse_factor(self) -> float:
        """Parse a unary-signed factor: ('-' | '+')* primary."""
        token = self._peek()
        if token is not None and token.kind == 'operator' and token.value == '-':
            self._advance()
            return -self._parse_factor()
        if token is not None and token.kind == 'operator' and token.value == '+':
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a number literal or a parenthesized expression."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        if token.kind == 'number':
            self._advance()
            return float(token.value)
        if token.kind == 'lparen':
            self._advance()
            value = self._parse_expression()
            closing = self._peek()
            if closing is None:
                raise ValueError("Mismatched parentheses: missing closing ')'")
            if closing.kind != 'rparen':
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found '{closing.value}'"
                )
            self._advance()
            return value
        if token.kind == 'rparen':
            raise ValueError(
                f"Mismatched parentheses: unexpected ')' at position {token.position}"
            )
        raise ValueError(f"Unexpected token '{token.value}' at position {token.position}")
```

## `test_expression_evaluator.py`

```python
"""Tests for the recursive descent ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator():
    """Provide a fresh evaluator for each test."""
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == pytest.approx(5.0)
    assert evaluator.evaluate("10 - 4") == pytest.approx(6.0)
    assert evaluator.evaluate("6 * 7") == pytest.approx(42.0)
    assert evaluator.evaluate("20 / 4") == pytest.approx(5.0)
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_operator_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == pytest.approx(14.0)
    assert evaluator.evaluate("10 - 2 * 3") == pytest.approx(4.0)
    assert evaluator.evaluate("8 / 2 * 2") == pytest.approx(8.0)   # left-associative
    assert evaluator.evaluate("100 - 50 + 10") == pytest.approx(60.0)


def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == pytest.approx(20.0)
    assert evaluator.evaluate("2 * (3 + 4)") == pytest.approx(14.0)
    assert evaluator.evaluate("((1 + 2) * 3) - 4") == pytest.approx(5.0)
    assert evaluator.evaluate("((((2))))") == pytest.approx(2.0)


def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == pytest.approx(-3.0)
    assert evaluator.evaluate("-(2 + 1)") == pytest.approx(-3.0)
    assert evaluator.evaluate("2 + -3") == pytest.approx(-1.0)
    assert evaluator.evaluate("-2 * -3") == pytest.approx(6.0)
    assert evaluator.evaluate("--5") == pytest.approx(5.0)


def test_error_cases(evaluator):
    # Empty / whitespace-only expressions
    with pytest.raises(ValueError, match="Empty"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty"):
        evaluator.evaluate("   ")
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched"):
        evaluator.evaluate("1 + 2)")
    # Division by zero
    with pytest.raises(ValueError, match="[Dd]ivision by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="[Dd]ivision by zero"):
        evaluator.evaluate("1 / (2 - 2)")
    # Invalid tokens / malformed input
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + @")
    with pytest.raises(ValueError, match="Unexpected"):
        evaluator.evaluate("2 + ")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("3 4")
```

## How it works

- **Tokenizer** (`_tokenize`) converts the string into a `List[_Token]`, rejecting unknown characters with position info and validating number literals (including `.5` style).
- **Recursive descent grammar**, mirroring precedence:
  - `expression := term (('+' | '-') term)*`
  - `term := factor (('*' | '/') factor)*`
  - `factor := ('-' | '+')* primary`
  - `primary := number | '(' expression ')'`
- Unary minus is handled by recursion in `_parse_factor`, so `--5`, `2 + -3`, and `-(2+1)` all work.
- Errors: the tokenizer raises on invalid characters; the parser raises on missing closing parens, stray closing parens, unexpected tokens, and end-of-input; division checks the right-hand side is non-zero.

Run the tests with:

```bash
python -m pytest test_expression_evaluator.py -v
```

One note: like a strict grammar, trailing garbage (e.g. `"3 4"`) and trailing operators (`"2 + "`) raise `ValueError` with "Unexpected token"/"Unexpected end of expression" messages rather than silently evaluating a prefix — this is intentional so no part of the input is silently ignored.