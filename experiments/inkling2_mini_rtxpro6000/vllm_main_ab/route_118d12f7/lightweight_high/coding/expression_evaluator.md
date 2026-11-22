Here's a complete implementation using a recursive descent parser (no `eval()`/`ast`), plus a pytest test suite.

## `expression_evaluator.py`

```python
"""
Recursive descent parser for a simple arithmetic expression evaluator.

Grammar (lowest to highest precedence):
    expression := term (('+' | '-') term)*
    term       := factor (('*' | '/') factor)*
    factor     := ('-' | '+')* primary
    primary    := NUMBER | '(' expression ')'
"""

from typing import List, Optional, Tuple


class ExpressionEvaluator:
    """Evaluate arithmetic expressions with +, -, *, /, parentheses, and unary minus."""

    def __init__(self) -> None:
        """Initialize empty parser state (tokens are re-built on every evaluate call)."""
        self._tokens: List[Tuple[str, str]] = []
        self._pos: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self, expr: str) -> float:
        """
        Evaluate the arithmetic expression `expr`.

        Args:
            expr: The expression string, e.g. "3.14 * (2 + 1)".

        Returns:
            The numeric result of the expression as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise ValueError(f"Expression must be a string, got {type(expr).__name__}")
        if expr.strip() == "":
            raise ValueError("Expression cannot be empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0
        result = self._parse_expression()

        if self._pos < len(self._tokens):
            kind, text = self._tokens[self._pos]
            if kind == "RPAREN":
                raise ValueError("Mismatched parentheses: unexpected ')'")
            raise ValueError(f"Unexpected token '{text}' after a complete expression")
        return result

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    def _tokenize(self, expr: str) -> List[Tuple[str, str]]:
        """
        Split `expr` into (kind, text) tokens.

        Token kinds: 'NUMBER', 'OP', 'LPAREN', 'RPAREN'. Whitespace is skipped.
        Raises ValueError for unrecognized characters or malformed numbers.
        """
        tokens: List[Tuple[str, str]] = []
        i, n = 0, len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or ch == ".":
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == "."):
                    i += 1
                text = expr[start:i]
                try:
                    float(text)
                except ValueError:
                    raise ValueError(f"Invalid number '{text}'") from None
                tokens.append(("NUMBER", text))
                continue
            if ch in "+-*/":
                tokens.append(("OP", ch))
                i += 1
                continue
            if ch == "(":
                tokens.append(("LPAREN", ch))
                i += 1
                continue
            if ch == ")":
                tokens.append(("RPAREN", ch))
                i += 1
                continue
            raise ValueError(f"Invalid token '{ch}' at position {i}")
        return tokens

    # ------------------------------------------------------------------
    # Parser helpers
    # ------------------------------------------------------------------
    def _peek(self) -> Optional[Tuple[str, str]]:
        """Return the current token without consuming it, or None if at end of input."""
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos]

    def _advance(self) -> Tuple[str, str]:
        """Consume and return the current token."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    # ------------------------------------------------------------------
    # Recursive descent grammar
    # ------------------------------------------------------------------
    def _parse_expression(self) -> float:
        """Parse an additive expression: term (('+' | '-') term)*."""
        left = self._parse_term()
        while True:
            tok = self._peek()
            if tok is None or tok[0] != "OP" or tok[1] not in "+-":
                break
            op = self._advance()[1]
            right = self._parse_term()
            left = left + right if op == "+" else left - right
        return left

    def _parse_term(self) -> float:
        """Parse a multiplicative term: factor (('*' | '/') factor)*."""
        left = self._parse_factor()
        while True:
            tok = self._peek()
            if tok is None or tok[0] != "OP" or tok[1] not in "*/":
                break
            op = self._advance()[1]
            right = self._parse_factor()
            if op == "*":
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
        return left

    def _parse_factor(self) -> float:
        """Parse a factor: unary plus/minus followed by a primary."""
        tok = self._peek()
        if tok is not None and tok[0] == "OP" and tok[1] == "-":
            self._advance()
            return -self._parse_factor()
        if tok is not None and tok[0] == "OP" and tok[1] == "+":
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a primary: a number or a parenthesized expression."""
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression — expected a number or '('")
        if tok[0] == "NUMBER":
            self._advance()
            return float(tok[1])
        if tok[0] == "LPAREN":
            self._advance()
            value = self._parse_expression()
            closing = self._peek()
            if closing is None:
                raise ValueError("Mismatched parentheses: missing closing ')'")
            if closing[0] != "RPAREN":
                raise ValueError(f"Mismatched parentheses: expected ')', found '{closing[1]}'")
            self._advance()
            return value
        if tok[0] == "RPAREN":
            raise ValueError("Mismatched parentheses: unexpected ')'")
        raise ValueError(f"Unexpected token '{tok[1]}' — expected a number or '('")
```

## `test_expression_evaluator.py`

```python
import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator():
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("9 / 2") == 4.5


def test_operator_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    assert evaluator.evaluate("100 / 10 / 2") == 5.0  # left associativity


def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("1 + 2 * (3 + 4)") == 15.0
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == 15.0


def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 - -2") == 5.0
    assert evaluator.evaluate("-2 * 3") == -6.0


def test_floating_point_numbers(evaluator):
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("0.5 + 0.25") == pytest.approx(0.75)
    assert evaluator.evaluate("-1.5 + 2.25") == pytest.approx(0.75)


@pytest.mark.parametrize("bad_expr, match", [
    ("", "empty"),
    ("   ", "empty"),
    ("(1 + 2", "Mismatched"),
    ("1 + 2)", "Mismatched"),
    (")", "Mismatched"),
    ("1 / 0", "Division by zero"),
    ("hello", "Invalid token"),
    ("2 @ 3", "Invalid token"),
    ("3 + ", "Unexpected end"),
    ("1.2.3", "Invalid number"),
])
def test_error_cases(evaluator, bad_expr, match):
    with pytest.raises(ValueError, match=match):
        evaluator.evaluate(bad_expr)
```

## How it works

- **Tokenizer** splits the input into `(kind, text)` tokens — numbers, `+ - * /`, `(`, `)` — skipping whitespace and rejecting anything else with a descriptive message. Malformed numbers like `1.2.3` are caught by a `float()` sanity check.
- **Recursive descent grammar** encodes precedence structurally: `_parse_expression` handles `+`/`-`, calls `_parse_term` for `*`/`/`, calls `_parse_factor` for unary minus/plus, and finally `_parse_primary` for numbers or parenthesized sub-expressions. Left-associativity comes naturally from the loop-based parsing.
- **Errors** all surface as `ValueError`: empty/whitespace input, invalid tokens/numbers, mismatched parens (missing, extra, or inside a group), division by zero, and truncated expressions (`"3 +"`).
- The evaluator is stateful only within a single `evaluate()` call, so one instance can be reused.

Run with:

```bash
pytest test_expression_evaluator.py -v
```

This gives you the 5 required test areas (plus an extra floating-point test and a parametrized error-case test covering 10 distinct failure modes).