# Mathematical Expression Evaluator

Here's a complete implementation using a tokenizer + recursive descent parser.

## `expression_evaluator.py`

```python
"""A safe mathematical expression evaluator (no eval())."""

import re
from typing import List, Optional, Tuple

Token = Tuple[str, str]  # (kind, value); kind: 'NUMBER', 'OPERATOR', 'LPAREN', 'RPAREN'
NUMBER_RE = re.compile(r'\d+(\.\d+)?|\.\d+')


class ExpressionEvaluator:
    """Evaluates basic arithmetic expressions safely.

    Supports +, -, *, / with standard precedence, parentheses for grouping,
    unary +/- (e.g. "-3", "-(2+1)"), and floating point numbers (e.g. "3.14").
    Uses a recursive descent parser.
    """

    def evaluate(self, expr: str) -> float:
        """Evaluate an arithmetic expression.

        Args:
            expr: The expression string to evaluate.

        Returns:
            The numeric result of the expression.

        Raises:
            TypeError: If expr is not a string.
            ValueError: If the expression is empty, contains invalid
                characters, has mismatched parentheses, has unexpected
                tokens, or divides by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"expr must be a string, got {type(expr).__name__}")
        if expr.strip() == "":
            raise ValueError("Empty expression")
        tokens = self._tokenize(expr)
        parser = self._Parser(tokens)
        try:
            result = parser.parse_expression()
            parser.expect_end()
        except RecursionError as exc:
            raise ValueError("Expression nested too deeply") from exc
        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Split an expression string into a list of tokens.

        Args:
            expr: The raw expression string.

        Returns:
            A list of (kind, value) tokens.

        Raises:
            ValueError: If a character cannot be tokenized.
        """
        tokens: List[Token] = []
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            number_match = NUMBER_RE.match(expr, i)
            if number_match:
                tokens.append(('NUMBER', number_match.group()))
                i = number_match.end()
                continue
            if ch in ('+', '-', '*', '/'):
                tokens.append(('OPERATOR', ch))
                i += 1
            elif ch == '(':
                tokens.append(('LPAREN', ch))
                i += 1
            elif ch == ')':
                tokens.append(('RPAREN', ch))
                i += 1
            else:
                raise ValueError(f"Invalid character '{ch}' at position {i}")
        return tokens

    class _Parser:
        """Recursive descent parser over a token list."""

        def __init__(self, tokens: List[Token]) -> None:
            """Initialize the parser with a token list."""
            self._tokens = tokens
            self._pos = 0

        def _peek(self) -> Optional[Token]:
            """Return the current token without consuming it."""
            if self._pos < len(self._tokens):
                return self._tokens[self._pos]
            return None

        def _advance(self) -> Token:
            """Consume and return the current token."""
            token = self._tokens[self._pos]
            self._pos += 1
            return token

        def _describe(self, token: Optional[Token]) -> str:
            """Return a human-readable description of a token for error messages."""
            if token is None:
                return "end of expression"
            return f"'{token[1]}'"

        def parse_expression(self) -> float:
            """Parse an additive expression: term (('+' | '-') term)*."""
            value = self.parse_term()
            while True:
                token = self._peek()
                if token is not None and token[0] == 'OPERATOR' and token[1] in ('+', '-'):
                    self._advance()
                    rhs = self.parse_term()
                    value = value + rhs if token[1] == '+' else value - rhs
                else:
                    return value

        def parse_term(self) -> float:
            """Parse a multiplicative expression: factor (('*' | '/') factor)*."""
            value = self.parse_factor()
            while True:
                token = self._peek()
                if token is not None and token[0] == 'OPERATOR' and token[1] in ('*', '/'):
                    self._advance()
                    rhs = self.parse_factor()
                    if token[1] == '*':
                        value = value * rhs
                    else:
                        if rhs == 0:
                            raise ValueError("Division by zero")
                        value = value / rhs
                else:
                    return value

        def parse_factor(self) -> float:
            """Parse a factor with optional unary +/- sign."""
            token = self._peek()
            if token is not None and token[0] == 'OPERATOR' and token[1] == '-':
                self._advance()
                return -self.parse_factor()
            if token is not None and token[0] == 'OPERATOR' and token[1] == '+':
                self._advance()
                return self.parse_factor()
            return self.parse_primary()

        def parse_primary(self) -> float:
            """Parse a primary: a number or a parenthesized expression."""
            token = self._peek()
            if token is not None and token[0] == 'NUMBER':
                self._advance()
                return float(token[1])
            if token is not None and token[0] == 'LPAREN':
                self._advance()
                value = self.parse_expression()
                closing = self._peek()
                if closing is None or closing[0] != 'RPAREN':
                    raise ValueError(
                        "Mismatched parentheses: expected ')' "
                        f"but found {self._describe(closing)}"
                    )
                self._advance()
                return value
            raise ValueError(
                f"Expected a number or '(' but found {self._describe(token)}"
            )

        def expect_end(self) -> None:
            """Raise ValueError if tokens remain after a complete parse."""
            token = self._peek()
            if token is None:
                return
            if token[0] == 'RPAREN':
                raise ValueError("Mismatched parentheses: unexpected ')'")
            raise ValueError(f"Unexpected token {self._describe(token)}")
```

## `test_expression_evaluator.py`

```python
"""Tests for the ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Provide a fresh evaluator for each test."""
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 4") == 2.5
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert isinstance(evaluator.evaluate("1 + 1"), float)


def test_operator_precedence(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    assert evaluator.evaluate("100 / 10 / 5") == 2.0  # left associative


def test_parentheses(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((2 + 3) * 4) / 5") == 4.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("(2 + 3) * (4 - 1)") == 15.0


def test_unary_minus(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 + -2") == 1.0
    assert evaluator.evaluate("--4") == 4.0
    assert evaluator.evaluate("-(-2)") == 2.0
    assert evaluator.evaluate("-3.5 * 2") == -7.0


def test_empty_and_invalid_tokens(evaluator: ExpressionEvaluator) -> None:
    with pytest.raises(ValueError, match="Empty"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 & 3")
    with pytest.raises(TypeError):
        evaluator.evaluate(42)  # type: ignore[arg-type]


def test_mismatched_parentheses(evaluator: ExpressionEvaluator) -> None:
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("((1 + 2)")


def test_division_by_zero(evaluator: ExpressionEvaluator) -> None:
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("(2 + 3) / (5 - 5)")
```

## How it works

The grammar mirrors the parser structure, which gives precedence for free:

- `parse_expression` → additive (`+`, `-`), lowest precedence
- `parse_term` → multiplicative (`*`, `/`)
- `parse_factor` → unary `+`/`-`
- `parse_primary` → number literal or parenthesized sub-expression

Error handling is layered: the tokenizer reports invalid characters (with position), `parse_primary` reports malformed primaries (e.g., trailing `+`), the parser reports unclosed parens and unmatched closing parens, and `parse_term` raises `ValueError("Division by zero")`. Anything left over after a complete parse is reported as an unexpected token.

Run the tests with:

```bash
python -m pytest test_expression_evaluator.py -v
```

One note: the evaluator supports plain numbers (`3.14`, `.5`) but not scientific notation (`1e5`) or functions — adding those would mean extending the tokenizer and `parse_primary`. Happy to add them if you want.