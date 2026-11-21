Here's a complete implementation split into two files — the evaluator and its test suite.

**`expression_evaluator.py`**

```python
"""A recursive descent mathematical expression evaluator."""

from __future__ import annotations

from typing import Any, NamedTuple


class Token(NamedTuple):
    """A lexical token produced by the tokenizer.

    Attributes:
        kind: Token type — 'NUMBER', '+', '-', '*', '/', '(', or ')'.
        value: The float value for NUMBER tokens, otherwise the operator/paren char.
        pos: Character position in the source expression where the token begins.
    """

    kind: str
    value: Any
    pos: int


class ExpressionEvaluator:
    """Evaluate arithmetic expressions using a recursive descent parser.

    Supports ``+``, ``-``, ``*``, ``/``, parentheses, unary minus/plus,
    and floating-point numbers.

    Raises:
        ValueError: On mismatched parentheses, division by zero, invalid
            tokens, or empty expressions.
    """

    def __init__(self) -> None:
        self._tokens: list[Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression and return the result.

        Args:
            expr: The expression to evaluate, e.g. ``"3.14 * (2 + 1)"``.

        Returns:
            The numeric value of the expression.

        Raises:
            TypeError: If ``expr`` is not a string.
            ValueError: If the expression is empty, contains invalid tokens,
                has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"Expression must be a string, got {type(expr).__name__}")
        if not expr.strip():
            raise ValueError("Expression is empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            tok = self._tokens[self._pos]
            if tok.kind == ")":
                raise ValueError(f"Mismatched parentheses: unexpected closing ')' at position {tok.pos}")
            raise ValueError(f"Unexpected token '{tok.value}' at position {tok.pos}")

        return float(result)

    # ---------------- tokenizer ----------------

    def _tokenize(self, expr: str) -> list[Token]:
        """Split the expression into tokens.

        Args:
            expr: The raw expression string.

        Returns:
            A list of Token instances in source order.

        Raises:
            ValueError: If a character or number is invalid.
        """
        tokens: list[Token] = []
        i, n = 0, len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or ch == ".":
                start, dots, digits = i, 0, 0
                while i < n and (expr[i].isdigit() or expr[i] == "."):
                    if expr[i] == ".":
                        dots += 1
                    else:
                        digits += 1
                    i += 1
                raw = expr[start:i]
                if dots > 1 or digits == 0:
                    raise ValueError(f"Invalid number '{raw}' at position {start}")
                tokens.append(Token("NUMBER", float(raw), start))
                continue
            if ch in "+-*/":
                tokens.append(Token(ch, ch, i))
                i += 1
                continue
            if ch in "()":
                tokens.append(Token(ch, ch, i))
                i += 1
                continue
            raise ValueError(f"Invalid token '{ch}' at position {i}")
        return tokens

    # ---------------- parser helpers ----------------

    def _peek(self) -> Token | None:
        """Return the current token without consuming it (or None if at end)."""
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        """Consume and return the current token."""
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    # ---------------- recursive descent grammar ----------------
    # expression := term (('+' | '-') term)*
    # term       := factor (('*' | '/') factor)*
    # factor     := ('-' | '+') factor | primary
    # primary    := NUMBER | '(' expression ')'

    def _parse_expression(self) -> float:
        """Parse a sum of terms: expression := term (('+' | '-') term)*."""
        result = self._parse_term()
        while (tok := self._peek()) is not None and tok.kind in ("+", "-"):
            self._advance()
            rhs = self._parse_term()
            result = result + rhs if tok.kind == "+" else result - rhs
        return result

    def _parse_term(self) -> float:
        """Parse a product of factors: term := factor (('*' | '/') factor)*."""
        result = self._parse_factor()
        while (tok := self._peek()) is not None and tok.kind in ("*", "/"):
            self._advance()
            rhs = self._parse_factor()
            if tok.kind == "*":
                result = result * rhs
            else:
                if rhs == 0.0:
                    raise ValueError("Division by zero")
                result = result / rhs
        return result

    def _parse_factor(self) -> float:
        """Parse unary signs: factor := ('-' | '+') factor | primary."""
        tok = self._peek()
        if tok is not None and tok.kind == "-":
            self._advance()
            return -self._parse_factor()
        if tok is not None and tok.kind == "+":
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a number or a parenthesized sub-expression."""
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression: expected a number, '(', or a unary operator")
        if tok.kind == "NUMBER":
            self._advance()
            return float(tok.value)
        if tok.kind == "(":
            self._advance()
            result = self._parse_expression()
            closing = self._peek()
            if closing is None:
                raise ValueError("Mismatched parentheses: missing closing ')'")
            if closing.kind != ")":
                raise ValueError(f"Mismatched parentheses: expected ')' but found '{closing.value}' at position {closing.pos}")
            self._advance()
            return result
        if tok.kind == ")":
            raise ValueError(f"Mismatched parentheses: unexpected closing ')' at position {tok.pos}")
        raise ValueError(f"Unexpected token '{tok.value}' at position {tok.pos}")


if __name__ == "__main__":
    evaluator = ExpressionEvaluator()
    for sample in ["2 + 3 * 4", "(2 + 3) * 4", "-3.14 * 2", "1 / 0"]:
        try:
            print(f"{sample!r:>15} = {evaluator.evaluate(sample)}")
        except ValueError as exc:
            print(f"{sample!r:>15} -> ValueError: {exc}")
```

**`test_expression_evaluator.py`**

```python
"""Pytest tests for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Return a fresh evaluator for each test."""
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3") == pytest.approx(5.0)
    assert evaluator.evaluate("10 - 4") == pytest.approx(6.0)
    assert evaluator.evaluate("6 * 7") == pytest.approx(42.0)
    assert evaluator.evaluate("20 / 4") == pytest.approx(5.0)
    assert evaluator.evaluate("3.5 + 2.25") == pytest.approx(5.75)
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_operator_precedence(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3 * 4") == pytest.approx(14.0)
    assert evaluator.evaluate("10 - 2 * 3") == pytest.approx(4.0)
    assert evaluator.evaluate("100 / 10 / 2") == pytest.approx(5.0)  # left-associative
    assert evaluator.evaluate("2 * 3 + 4 * 5") == pytest.approx(26.0)


def test_parentheses(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("(2 + 3) * 4") == pytest.approx(20.0)
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == pytest.approx(21.0)
    assert evaluator.evaluate("2 + (3 * 4)") == pytest.approx(14.0)


def test_unary_minus(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("-3") == pytest.approx(-3.0)
    assert evaluator.evaluate("-(2 + 1)") == pytest.approx(-3.0)
    assert evaluator.evaluate("2 + -3") == pytest.approx(-1.0)
    assert evaluator.evaluate("3 * -2") == pytest.approx(-6.0)
    assert evaluator.evaluate("--3") == pytest.approx(3.0)


def test_error_cases(evaluator: ExpressionEvaluator) -> None:
    # Empty expressions
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate("   ")
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    # Invalid tokens / numbers
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + @")
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("1.2.3")
```

**How it works**

- **Tokenizer** (`_tokenize`): walks the string, producing `NUMBER` tokens (validating float syntax, rejecting things like `1.2.3` or `.`), operator/paren tokens, and raising `ValueError` with the position for anything else.
- **Recursive descent parser**: mirrors a standard precedence-climbing grammar — `expression` (+/-) → `term` (*//) → `factor` (unary signs, recursed so `--3` works) → `primary` (numbers and parenthesized sub-expressions). This gives `*`/`/` higher precedence and left-associativity for free.
- **Error handling**: every failure mode funnels into `ValueError` with a message naming the problem and, where useful, the character position. Division by zero is checked explicitly at the `/` node rather than relying on the raw `ZeroDivisionError`.

**Run the tests:**

```bash
python -m pytest test_expression_evaluator.py -v
```

One note: this deliberately doesn't support exponents, functions, or scientific notation (`1e5` is reported as an invalid token) — adding those would mean extending the grammar (e.g. a `power` level between `factor` and `term`), which is a straightforward next step if needed.