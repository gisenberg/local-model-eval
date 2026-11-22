Here's a complete implementation split into a library module and a test module.

## `expression_evaluator.py`

```python
"""A mathematical expression evaluator using a recursive descent parser.

Supports +, -, *, / with correct precedence, parentheses, unary minus,
and floating point numbers. No eval() or ast.literal_eval() is used.
"""

from typing import List, NamedTuple


class _Token(NamedTuple):
    """A single lexical token produced by the tokenizer."""

    kind: str  # 'NUMBER' | '+' | '-' | '*' | '/' | 'LPAREN' | 'RPAREN' | 'EOF'
    value: str
    position: int


class ExpressionEvaluator:
    """Recursive descent parser/evaluator for arithmetic expressions.

    Grammar (informal):

        expr    := term (('+' | '-') term)*
        term    := factor (('*' | '/') factor)*
        factor  := '-' factor | primary
        primary := NUMBER | '(' expr ')'
    """

    def __init__(self) -> None:
        """Initialize the evaluator's internal parser state."""
        self._tokens: List[_Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate an arithmetic expression.

        Args:
            expr: The expression string to evaluate (e.g. '3.14 * (2 + 1)').

        Returns:
            The numeric result of the expression as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                has mismatched parentheses, or attempts division by zero.
        """
        if not isinstance(expr, str):
            raise ValueError(f"Expression must be a string, got {type(expr).__name__}")
        if expr.strip() == "":
            raise ValueError("Expression is empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        if self._peek().kind != 'EOF':
            token = self._peek()
            if token.kind == 'RPAREN':
                raise ValueError(
                    f"Mismatched parentheses: unexpected ')' at position {token.position}"
                )
            raise ValueError(f"Unexpected token '{token.value}' at position {token.position}")

        return float(result)

    def _tokenize(self, expr: str) -> List[_Token]:
        """Convert a raw expression string into a list of tokens.

        Args:
            expr: The raw expression string.

        Returns:
            A list of tokens terminated by an 'EOF' token.

        Raises:
            ValueError: If an invalid character or malformed number is found.
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
                    float(text)
                except ValueError:
                    raise ValueError(f"Invalid number '{text}' at position {start}") from None
                tokens.append(_Token('NUMBER', text, start))
                continue
            if ch in ('+', '-', '*', '/'):
                tokens.append(_Token(ch, ch, i))
                i += 1
                continue
            if ch == '(':
                tokens.append(_Token('LPAREN', ch, i))
                i += 1
                continue
            if ch == ')':
                tokens.append(_Token('RPAREN', ch, i))
                i += 1
                continue
            raise ValueError(f"Invalid token '{ch}' at position {i}")
        tokens.append(_Token('EOF', '', n))
        return tokens

    def _peek(self) -> _Token:
        """Return the current token without consuming it."""
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        """Consume and return the current token."""
        token = self._tokens[self._pos]
        if token.kind != 'EOF':
            self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse expr := term (('+' | '-') term)*."""
        left = self._parse_term()
        while self._peek().kind in ('+', '-'):
            op = self._advance().kind
            right = self._parse_term()
            left = left + right if op == '+' else left - right
        return left

    def _parse_term(self) -> float:
        """Parse term := factor (('*' | '/') factor)*."""
        left = self._parse_factor()
        while self._peek().kind in ('*', '/'):
            op = self._advance().kind
            right = self._parse_factor()
            if op == '*':
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
        return left

    def _parse_factor(self) -> float:
        """Parse factor := '-' factor | primary (unary minus)."""
        if self._peek().kind == '-':
            self._advance()
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse primary := NUMBER | '(' expr ')'."""
        token = self._peek()
        if token.kind == 'NUMBER':
            self._advance()
            return float(token.value)
        if token.kind == 'LPAREN':
            self._advance()
            result = self._parse_expression()
            closing = self._peek()
            if closing.kind != 'RPAREN':
                raise ValueError("Mismatched parentheses: expected ')'")
            self._advance()
            return result
        if token.kind == 'EOF':
            raise ValueError("Unexpected end of expression: expected a number or '('")
        if token.kind == 'RPAREN':
            raise ValueError(
                f"Mismatched parentheses: unexpected ')' at position {token.position}"
            )
        if token.kind == '+':
            raise ValueError(
                f"Unexpected token '+' at position {token.position}: unary '+' is not supported"
            )
        raise ValueError(
            f"Expected a number or '(' but found '{token.value}' at position {token.position}"
        )


if __name__ == "__main__":
    evaluator = ExpressionEvaluator()
    print(evaluator.evaluate("3.14 * (2 + 1)"))
```

## `test_expression_evaluator.py`

```python
"""Pytest tests for the recursive descent ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


def test_basic_arithmetic():
    """Basic +, -, *, / operations and floating point results."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == pytest.approx(5.0)
    assert evaluator.evaluate("10 - 4") == pytest.approx(6.0)
    assert evaluator.evaluate("3 * 7") == pytest.approx(21.0)
    assert evaluator.evaluate("9 / 2") == pytest.approx(4.5)
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_operator_precedence():
    """Multiplication/division bind tighter than addition/subtraction."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == pytest.approx(14.0)
    assert evaluator.evaluate("10 - 2 * 3") == pytest.approx(4.0)
    assert evaluator.evaluate("2 * 3 + 4 * 5") == pytest.approx(26.0)


def test_parentheses():
    """Parentheses override operator precedence."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == pytest.approx(20.0)
    assert evaluator.evaluate("(10 - 4) / 3") == pytest.approx(2.0)
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == pytest.approx(15.0)


def test_unary_minus():
    """Unary minus applies to literals and parenthesized groups."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == pytest.approx(-3.0)
    assert evaluator.evaluate("-(2 + 1)") == pytest.approx(-3.0)
    assert evaluator.evaluate("2 + -3") == pytest.approx(-1.0)
    assert evaluator.evaluate("-3 * 2") == pytest.approx(-6.0)


def test_error_cases():
    """Invalid expressions raise ValueError with descriptive messages."""
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Mismatched"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched"):
        evaluator.evaluate("1 + 2)")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 @ 2")
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("3.14.15")
```

## Design notes

- **Tokenizer**: converts the raw string into a `List[_Token]`, raising `ValueError` with a position for invalid characters or malformed numbers (e.g. `3.14.15`).
- **Recursive descent grammar**: `expr` (additive) → `term` (multiplicative) → `factor` (unary minus) → `primary` (number or parenthesized group). This gives correct precedence for free and handles unary minus via the recursive `factor` rule.
- **Error reporting**: `evaluate()` catches trailing tokens after a successful parse (e.g. `1 + 2)`) and reports mismatched parentheses, while the grammar level reports unterminated groups and invalid structure — all as `ValueError` with specific messages.
- **State reset**: `evaluate()` reinitializes `_tokens`/`_pos` each call, so one evaluator instance can be reused.

Run the tests with:

```bash
pytest test_expression_evaluator.py -v
```

One note: unary `+` is intentionally not supported (per spec, only unary `-`), so `"+3"` raises a descriptive `ValueError`. Easy to add if you want it — just handle the `'+'` case in `_parse_factor` the same way as `'-'`.