Here's a complete solution. The parser is a recursive descent parser with a Pratt-style grammar: `expression → term → factor → primary`, which gives correct precedence (`*`/`/` bind tighter than `+`/`-`) for free.

## `expression_evaluator.py`

```python
"""Recursive descent parser for basic arithmetic expressions."""

from typing import List, NamedTuple, Union, cast

TokenValue = Union[str, float, None]


class _Token(NamedTuple):
    """A lexical token produced by the tokenizer."""

    kind: str          # 'NUMBER', '+', '-', '*', '/', '(', ')', 'EOF'
    value: TokenValue  # float for numbers, the symbol otherwise, None for EOF
    position: int      # index in the source string where the token starts


class _Parser:
    """Recursive descent parser; one instance is created per evaluate() call."""

    def __init__(self, tokens: List[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # -- cursor helpers -----------------------------------------------------

    @property
    def _current(self) -> _Token:
        return self._tokens[self._pos]

    def _advance(self) -> None:
        """Move the cursor past the current token (never past EOF)."""
        if self._current.kind != 'EOF':
            self._pos += 1

    def _describe(self, token: _Token) -> str:
        """Return a human-readable description of a token for error messages."""
        if token.kind == 'EOF':
            return 'end of expression'
        if token.kind == 'NUMBER':
            return f"number '{token.value:g}'"
        return f"'{token.value}'"

    # -- grammar rules ------------------------------------------------------

    def parse_expression(self) -> float:
        """expression := term (('+' | '-') term)*"""
        value = self.parse_term()
        while self._current.kind in ('+', '-'):
            op = self._current.kind
            self._advance()
            rhs = self.parse_term()
            value = value + rhs if op == '+' else value - rhs
        return value

    def parse_term(self) -> float:
        """term := factor (('*' | '/') factor)*"""
        value = self.parse_factor()
        while self._current.kind in ('*', '/'):
            op = self._current.kind
            op_pos = self._current.position
            self._advance()
            rhs = self.parse_factor()
            if op == '*':
                value = value * rhs
            else:
                if rhs == 0:
                    raise ValueError(f"Division by zero at position {op_pos}")
                value = value / rhs
        return value

    def parse_factor(self) -> float:
        """factor := ('-' | '+')* primary   (unary minus / plus)"""
        if self._current.kind == '-':
            self._advance()
            return -self.parse_factor()
        if self._current.kind == '+':
            self._advance()
            return self.parse_factor()
        return self.parse_primary()

    def parse_primary(self) -> float:
        """primary := NUMBER | '(' expression ')'"""
        token = self._current
        if token.kind == 'NUMBER':
            self._advance()
            return cast(float, token.value)
        if token.kind == '(':
            self._advance()
            value = self.parse_expression()
            if self._current.kind != ')':
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found "
                    f"{self._describe(self._current)} at position {self._current.position}"
                )
            self._advance()
            return value
        if token.kind == 'EOF':
            raise ValueError("Unexpected end of expression: expected a number or '('")
        raise ValueError(
            f"Unexpected token {self._describe(token)} at position {token.position}: "
            "expected a number or '('"
        )

    def ensure_end(self) -> None:
        """Raise ValueError if any tokens remain after a valid expression."""
        token = self._current
        if token.kind == 'EOF':
            return
        if token.kind == ')':
            raise ValueError(f"Mismatched parentheses: unexpected ')' at position {token.position}")
        raise ValueError(f"Unexpected token {self._describe(token)} at position {token.position}")


class ExpressionEvaluator:
    """Evaluates arithmetic expressions via a recursive descent parser."""

    def evaluate(self, expr: str) -> float:
        """Evaluate an arithmetic expression and return its numeric value.

        Supports +, -, *, /, parentheses, unary minus, and floating-point numbers.

        Args:
            expr: The expression string to evaluate.

        Returns:
            The value of the expression as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens or
                numbers, has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise ValueError(f"Expression must be a string, got {type(expr).__name__}")
        if not expr.strip():
            raise ValueError("Empty expression")
        tokens = self._tokenize(expr)
        parser = _Parser(tokens)
        result = parser.parse_expression()
        parser.ensure_end()
        return float(result)

    def _tokenize(self, expr: str) -> List[_Token]:
        """Convert the expression string into a list of tokens.

        Raises:
            ValueError: If a character or number literal is invalid.
        """
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
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                text = expr[start:i]
                try:
                    number = float(text)
                except ValueError:
                    raise ValueError(f"Invalid number '{text}' at position {start}") from None
                tokens.append(_Token('NUMBER', number, start))
                continue
            if ch in '+-*/()':
                tokens.append(_Token(ch, ch, i))
                i += 1
                continue
            raise ValueError(f"Invalid token '{ch}' at position {i}")
        tokens.append(_Token('EOF', None, n))
        return tokens
```

## `test_expression_evaluator.py`

```python
"""Tests for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("6 / 3") == 2.0
    assert evaluator.evaluate("3.14 + 1.86") == pytest.approx(5.0)


def test_operator_precedence(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    assert evaluator.evaluate("100 / 10 / 5") == 2.0  # left associativity


def test_parentheses(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 - 1))") == 6.0
    assert evaluator.evaluate("2 * (3 + (4 * 5))") == 46.0


def test_unary_minus(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 - -2") == 5.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-2 * 3") == -6.0


def test_error_cases(evaluator: ExpressionEvaluator) -> None:
    # empty expressions
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

    # mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")   # missing closing paren
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")   # unexpected closing paren

    # division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")

    # invalid tokens / numbers
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + $")
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("1.2.3")
```

## Notes

- **Precedence & associativity** come from the grammar structure: `expression` (lowest) → `term` → `factor` → `primary`. The `while` loops inside `parse_expression` and `parse_term` produce left-associative behavior, so `100 / 10 / 5` = 2.0 (not 10.0).
- **Unary minus** lives in `parse_factor`, which recurses, so `-(2+1)`, `--5`, and `3 - -2` all work and unary binds tighter than `*`/`/` (matching Python semantics).
- **Errors** are raised with source positions (e.g., `Invalid token '$' at position 4`), which makes debugging much easier.
- **Thread-safety**: each `evaluate()` call creates a fresh `_Parser`, so the `ExpressionEvaluator` instance itself is stateless and reusable.
- The number tokenizer also accepts forms like `.5` and `3.`, and rejects malformed ones like `1.2.3` or `..` with a descriptive `Invalid number` error.

Run with:

```bash
pytest test_expression_evaluator.py -v
```

One limitation to be aware of: expressions nested more than ~100 levels deep (e.g., 1000 open parens) will hit Python's recursion limit with a `RecursionError` rather than a `ValueError`. If you need to support pathological nesting, that would need an iterative parser or a depth cap — happy to add that if you want.