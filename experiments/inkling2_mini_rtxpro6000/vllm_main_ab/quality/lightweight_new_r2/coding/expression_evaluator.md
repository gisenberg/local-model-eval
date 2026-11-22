Here's a complete implementation: a recursive descent parser (no `eval()`), plus a pytest suite.

**`expression_evaluator.py`**

```python
"""Recursive descent mathematical expression evaluator."""

from typing import List, Optional, Tuple, cast

# Token = (type, value, position)
#   type: 'NUMBER', '+', '-', '*', '/', 'LPAREN', 'RPAREN'
#   value: float for NUMBER tokens, otherwise the literal character
Token = Tuple[str, object, int]


class ExpressionEvaluator:
    """Evaluate arithmetic expressions with +, -, *, /, parentheses,
    unary minus, and floating point numbers.

    Uses a recursive descent parser (no eval()).
    """

    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate *expr* and return the result as a float.

        Args:
            expr: The arithmetic expression to evaluate.

        Returns:
            The computed value.

        Raises:
            TypeError: If expr is not a string.
            ValueError: If the expression is empty, contains invalid tokens,
                has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise TypeError(f"Expression must be a string, got {type(expr).__name__}")
        if expr.strip() == "":
            raise ValueError("Expression is empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0
        result = self._parse_expr()

        if self._pos < len(self._tokens):
            leftover = self._tokens[self._pos]
            if leftover[0] == 'RPAREN':
                raise ValueError(
                    f"Mismatched parentheses: unexpected ')' at position {leftover[2]}"
                )
            raise ValueError(
                f"Unexpected token '{leftover[1]}' at position {leftover[2]}"
            )
        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Split *expr* into a list of tokens.

        Args:
            expr: The raw expression string.

        Returns:
            A list of Token tuples.

        Raises:
            ValueError: If a character or number is not recognized.
        """
        tokens: List[Token] = []
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or ch == '.':
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                text = expr[start:i]
                try:
                    value = float(text)
                except ValueError:
                    raise ValueError(
                        f"Invalid number '{text}' at position {start}"
                    ) from None
                tokens.append(('NUMBER', value, start))
                continue
            if ch in '+-*/':
                tokens.append((ch, ch, i))
                i += 1
                continue
            if ch == '(':
                tokens.append(('LPAREN', ch, i))
                i += 1
                continue
            if ch == ')':
                tokens.append(('RPAREN', ch, i))
                i += 1
                continue
            raise ValueError(f"Invalid token '{ch}' at position {i}")
        return tokens

    def _current(self) -> Optional[Token]:
        """Return the token at the cursor, or None if the token stream is exhausted."""
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        """Consume and return the current token."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _parse_expr(self) -> float:
        """Parse an expression: term (('+' | '-') term)*."""
        value = self._parse_term()
        while True:
            tok = self._current()
            if tok is not None and tok[0] in ('+', '-'):
                self._advance()
                rhs = self._parse_term()
                value = value + rhs if tok[0] == '+' else value - rhs
            else:
                return value

    def _parse_term(self) -> float:
        """Parse a term: unary (('*' | '/') unary)*."""
        value = self._parse_unary()
        while True:
            tok = self._current()
            if tok is not None and tok[0] in ('*', '/'):
                self._advance()
                rhs = self._parse_unary()
                if tok[0] == '*':
                    value = value * rhs
                else:
                    if rhs == 0:
                        raise ValueError("Division by zero")
                    value = value / rhs
            else:
                return value

    def _parse_unary(self) -> float:
        """Parse a unary-signed factor: ('+' | '-')* factor."""
        tok = self._current()
        if tok is not None and tok[0] == '-':
            self._advance()
            return -self._parse_unary()
        if tok is not None and tok[0] == '+':
            self._advance()
            return self._parse_unary()
        return self._parse_factor()

    def _parse_factor(self) -> float:
        """Parse a factor: number | '(' expression ')'.

        Raises:
            ValueError: If the token is not a number or an opening parenthesis,
                or if parentheses are mismatched.
        """
        tok = self._current()
        if tok is None:
            raise ValueError("Unexpected end of expression: expected a number or '('")
        if tok[0] == 'NUMBER':
            self._advance()
            return cast(float, tok[1])
        if tok[0] == 'LPAREN':
            self._advance()
            value = self._parse_expr()
            closing = self._current()
            if closing is None:
                raise ValueError(
                    "Mismatched parentheses: expected ')' but reached end of expression"
                )
            if closing[0] != 'RPAREN':
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found '{closing[1]}' "
                    f"at position {closing[2]}"
                )
            self._advance()
            return value
        if tok[0] == 'RPAREN':
            raise ValueError(f"Mismatched parentheses: unexpected ')' at position {tok[2]}")
        raise ValueError(f"Unexpected token '{tok[1]}' at position {tok[2]}")


if __name__ == "__main__":
    evaluator = ExpressionEvaluator()
    print(evaluator.evaluate("2 + 3 * 4"))       # 14.0
    print(evaluator.evaluate("(2 + 3) * 4"))     # 20.0
    print(evaluator.evaluate("-(2 + 1)"))        # -3.0
    print(evaluator.evaluate("3.14 * 2"))        # 6.28
```

**`test_expression_evaluator.py`**

```python
"""Tests for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Provide a fresh evaluator instance per test."""
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3") == 5
    assert evaluator.evaluate("10 - 4") == 6
    assert evaluator.evaluate("2 * 3") == 6
    assert evaluator.evaluate("8 / 2") == 4
    assert evaluator.evaluate("3.14 + 0.01") == pytest.approx(3.15)


def test_precedence(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3 * 4") == 14
    assert evaluator.evaluate("10 / 2 - 1") == 4
    assert evaluator.evaluate("2 * 3 + 4") == 10
    assert evaluator.evaluate("10 - 2 * 3") == 4


def test_parentheses(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("(2 + 3) * 4") == 20
    assert evaluator.evaluate("2 * (3 + 4)") == 14
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == 15
    assert evaluator.evaluate("3 + (4 * 2)") == 11


def test_unary_minus(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("-3") == -3
    assert evaluator.evaluate("-(2 + 1)") == -3
    assert evaluator.evaluate("3 - -2") == 5
    assert evaluator.evaluate("--5") == 5
    assert evaluator.evaluate("-3.14") == pytest.approx(-3.14)


@pytest.mark.parametrize(
    "expr, match",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("2 + * 3", "Unexpected token"),
        ("(3 + 4", "Mismatched parentheses"),
        ("3 + 4)", "Mismatched parentheses"),
        ("1 / 0", "Division by zero"),
        ("10 / (2 - 2)", "Division by zero"),
        ("3.14.1", "Invalid number"),
        ("2 $ 3", "Invalid token"),
    ],
)
def test_error_cases(evaluator: ExpressionEvaluator, expr: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        evaluator.evaluate(expr)
```

**How it works**

The grammar mirrors the classic precedence hierarchy, implemented as mutually recursive methods:

```
expression → term (('+' | '-') term)*
term       → unary (('*' | '/') unary)*
unary      → ('+' | '-')* factor
factor     → NUMBER | '(' expression ')'
```

Each level handles one precedence tier, so `*` binds tighter than `+` automatically. Unary minus is handled at its own level (so it binds tighter than `*`, e.g. `-2 * 3 == -6`). Errors are raised with descriptive `ValueError` messages:

- **Empty**: whitespace-only or empty strings are rejected before parsing
- **Mismatched parens**: unclosed `(`, extra `)`, or `)` appearing where an operand is expected
- **Invalid tokens**: unrecognized characters, malformed numbers like `3.14.1`, or unexpected tokens
- **Division by zero**: checked before each `/`, including zero produced by a parenthesized subexpression

Run the tests with:

```bash
pytest test_expression_evaluator.py -v
```

One note: the parser keeps its cursor state on the instance, but `evaluate()` resets it at the start of every call, so an `ExpressionEvaluator` can be reused safely (just not concurrently from multiple threads).