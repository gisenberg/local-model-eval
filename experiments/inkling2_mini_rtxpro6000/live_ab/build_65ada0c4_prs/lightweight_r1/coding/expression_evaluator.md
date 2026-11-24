Here's a complete implementation — a tokenizer plus a recursive descent parser (`expression → term → factor → primary`), with no `eval()` anywhere.

**`expression_evaluator.py`**

```python
"""Recursive descent parser for mathematical expressions."""

from dataclasses import dataclass
from typing import List


@dataclass
class _Token:
    """A single lexical token produced by the lexer."""

    kind: str   # 'number' | 'operator' | 'lparen' | 'rparen' | 'eof'
    value: str
    pos: int    # character offset in the source expression


class ExpressionEvaluator:
    """Evaluates arithmetic expressions using a recursive descent parser.

    Supports +, -, *, / with correct precedence, parentheses, unary minus,
    and floating point literals.
    """

    def __init__(self) -> None:
        self._tokens: List[_Token] = []
        self._pos: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, expr: str) -> float:
        """Evaluate an arithmetic expression string.

        Args:
            expr: The expression to evaluate (e.g. "2 + 3 * (4 - 1)").

        Returns:
            The numeric result of the expression.

        Raises:
            TypeError: If `expr` is not a string.
            ValueError: If the expression is empty, contains invalid tokens,
                has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"Expression must be a string, got {type(expr).__name__}")
        if expr.strip() == "":
            raise ValueError("Empty expression: nothing to evaluate")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        trailing = self._peek()
        if trailing.kind == 'rparen':
            raise ValueError(
                f"Mismatched parentheses: unexpected ')' at position {trailing.pos}"
            )
        if trailing.kind != 'eof':
            raise ValueError(
                f"Unexpected token {trailing.value!r} at position {trailing.pos}"
            )
        return result

    # ------------------------------------------------------------------
    # Lexer
    # ------------------------------------------------------------------

    def _tokenize(self, expr: str) -> List[_Token]:
        """Split `expr` into tokens, raising ValueError on invalid characters."""
        tokens: List[_Token] = []
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or ch == '.':
                start = i
                seen_dot = False
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if seen_dot:
                            raise ValueError(
                                f"Invalid number {expr[start:i + 1]!r} at position {start}"
                            )
                        seen_dot = True
                    i += 1
                text = expr[start:i]
                try:
                    float(text)
                except ValueError:
                    raise ValueError(f"Invalid number {text!r} at position {start}") from None
                tokens.append(_Token('number', text, start))
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
            raise ValueError(f"Invalid token {ch!r} at position {i}")
        tokens.append(_Token('eof', '', n))
        return tokens

    # ------------------------------------------------------------------
    # Parser helpers
    # ------------------------------------------------------------------

    def _peek(self) -> _Token:
        """Return the current token without consuming it."""
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        """Consume and return the current token (never advances past 'eof')."""
        tok = self._tokens[self._pos]
        if tok.kind != 'eof':
            self._pos += 1
        return tok

    # ------------------------------------------------------------------
    # Grammar rules (recursive descent)
    # ------------------------------------------------------------------

    def _parse_expression(self) -> float:
        """Parse an expression: term (('+' | '-') term)*"""
        value = self._parse_term()
        while self._peek().kind == 'operator' and self._peek().value in '+-':
            op = self._advance().value
            rhs = self._parse_term()
            value = value + rhs if op == '+' else value - rhs
        return value

    def _parse_term(self) -> float:
        """Parse a term: factor (('*' | '/') factor)*"""
        value = self._parse_factor()
        while self._peek().kind == 'operator' and self._peek().value in '*/':
            op_tok = self._advance()
            rhs = self._parse_factor()
            if op_tok.value == '/':
                if rhs == 0:
                    raise ValueError(f"Division by zero at position {op_tok.pos}")
                value = value / rhs
            else:
                value = value * rhs
        return value

    def _parse_factor(self) -> float:
        """Parse a factor, handling unary '-' and '+' operators."""
        tok = self._peek()
        if tok.kind == 'operator' and tok.value == '-':
            self._advance()
            return -self._parse_factor()
        if tok.kind == 'operator' and tok.value == '+':
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a primary: number | '(' expression ')'"""
        tok = self._peek()
        if tok.kind == 'number':
            self._advance()
            return float(tok.value)
        if tok.kind == 'lparen':
            self._advance()
            open_pos = tok.pos
            value = self._parse_expression()
            closing = self._peek()
            if closing.kind != 'rparen':
                if closing.kind == 'eof':
                    raise ValueError(
                        f"Mismatched parentheses: missing ')' to close '(' at position {open_pos}"
                    )
                raise ValueError(
                    f"Mismatched parentheses: expected ')' at position {closing.pos}, "
                    f"found {closing.value!r}"
                )
            self._advance()
            return value
        if tok.kind == 'eof':
            raise ValueError("Unexpected end of expression: expected a number or '('")
        if tok.kind == 'rparen':
            raise ValueError(f"Mismatched parentheses: unexpected ')' at position {tok.pos}")
        raise ValueError(
            f"Expected a number or '(' at position {tok.pos}, found {tok.value!r}"
        )
```

**`test_expression_evaluator.py`**

```python
"""Tests for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 7") == 21.0
    assert evaluator.evaluate("8 / 2") == 4.0
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_operator_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0


def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * 3)") == 9.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0


def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--3") == 3.0
    assert evaluator.evaluate("2 + -3") == -1.0
    assert evaluator.evaluate("-2.5 + 1") == pytest.approx(-1.5)


def test_error_cases(evaluator):
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="[Mm]ismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1 + @")
```

**Design notes**

- The grammar is layered so precedence falls out naturally: `expression` (+/−) → `term` (×/÷) → `factor` (unary −/+) → `primary` (number or parenthesized expression). Recursive descent mirrors this grammar one-to-one.
- Each rule consumes tokens via `_peek()`/`_advance()`, so parentheses are handled by re-entering the expression rule — nested groups work for free.
- Errors carry position information (e.g. `Division by zero at position 5`) and both sides of a paren mismatch are caught: unclosed `'('` (in `_parse_primary`) and extra `')'` (in `evaluate` or `_parse_primary`).

Run the tests with:

```bash
pytest test_expression_evaluator.py -v
```