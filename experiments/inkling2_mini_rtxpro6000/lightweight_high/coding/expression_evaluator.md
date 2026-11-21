Here's a complete implementation — a tokenizer plus a recursive descent parser (grammar: `expression → term → unary → primary`), with all errors raised as `ValueError`.

## `expression_evaluator.py`

```python
"""Recursive descent mathematical expression evaluator."""

from typing import List, Optional, Tuple


class _UnexpectedEnd(Exception):
    """Internal signal: the token stream ran out mid-expression."""


class ExpressionEvaluator:
    """Evaluate mathematical expressions with a recursive descent parser.

    Supports +, -, *, /, parentheses, unary minus and floating-point numbers.
    Does NOT use eval() or ast.literal_eval().
    """

    def __init__(self) -> None:
        """Initialise parser state."""
        self._tokens: List[Tuple[str, str, int]] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate an expression string and return the result.

        Args:
            expr: The expression to evaluate, e.g. '3.14 * (2 + 1)'.

        Returns:
            The result of the expression as a float.

        Raises:
            ValueError: For empty expressions, invalid tokens/numbers, mismatched
                parentheses, unexpected trailing tokens, or division by zero.
        """
        if not isinstance(expr, str):
            raise ValueError(f"Expression must be a string, got {type(expr).__name__}")
        if not expr.strip():
            raise ValueError("Expression cannot be empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        try:
            result = self._parse_expression()
        except _UnexpectedEnd:
            raise ValueError("Unexpected end of expression") from None

        if self._pos < len(self._tokens):
            tok = self._tokens[self._pos]
            if tok[0] == 'RPAREN':
                raise ValueError(f"Mismatched parentheses: unexpected ')' at position {tok[2]}")
            raise ValueError(f"Unexpected token '{tok[1]}' at position {tok[2]}")

        return float(result)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _tokenize(self, expr: str) -> List[Tuple[str, str, int]]:
        """Split the expression into (kind, text, position) tokens.

        Raises:
            ValueError: For invalid numbers or unrecognised characters.
        """
        tokens: List[Tuple[str, str, int]] = []
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
                tokens.append(('NUMBER', text, start))
                continue
            if ch in ('+', '-', '*', '/'):
                tokens.append(('OP', ch, i))
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

    # ------------------------------------------------------------------
    # Parser helpers
    # ------------------------------------------------------------------

    def _peek(self) -> Optional[Tuple[str, str, int]]:
        """Return the current token, or None if the stream is exhausted."""
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos]

    def _advance(self) -> Tuple[str, str, int]:
        """Consume and return the current token."""
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _at_op(self, op: str) -> bool:
        """Return True if the current token is the given operator."""
        tok = self._peek()
        return tok is not None and tok[0] == 'OP' and tok[1] == op

    # ------------------------------------------------------------------
    # Grammar: expression -> term (('+' | '-') term)*
    #          term       -> unary (('*' | '/') unary)*
    #          unary      -> ('-' | '+')* primary
    #          primary    -> NUMBER | '(' expression ')'
    # ------------------------------------------------------------------

    def _parse_expression(self) -> float:
        """Parse an additive expression: term (('+' | '-') term)*."""
        result = self._parse_term()
        while self._at_op('+') or self._at_op('-'):
            op = self._advance()[1]
            rhs = self._parse_term()
            result = result + rhs if op == '+' else result - rhs
        return result

    def _parse_term(self) -> float:
        """Parse a multiplicative expression: unary (('*' | '/') unary)*."""
        result = self._parse_unary()
        while self._at_op('*') or self._at_op('/'):
            op_tok = self._advance()
            op = op_tok[1]
            rhs = self._parse_unary()
            if op == '*':
                result = result * rhs
            else:
                if rhs == 0:
                    raise ValueError(f"Division by zero at position {op_tok[2]}")
                result = result / rhs
        return result

    def _parse_unary(self) -> float:
        """Parse a unary expression: ('-' | '+')* primary."""
        if self._at_op('-'):
            self._advance()
            return -self._parse_unary()
        if self._at_op('+'):
            self._advance()
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a primary expression: NUMBER | '(' expression ')'."""
        tok = self._peek()
        if tok is None:
            raise _UnexpectedEnd()
        if tok[0] == 'NUMBER':
            self._advance()
            return float(tok[1])
        if tok[0] == 'LPAREN':
            self._advance()
            try:
                result = self._parse_expression()
            except _UnexpectedEnd:
                raise ValueError("Mismatched parentheses: missing closing ')'") from None
            closing = self._peek()
            if closing is None:
                raise ValueError("Mismatched parentheses: missing closing ')'")
            if closing[0] != 'RPAREN':
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found '{closing[1]}' "
                    f"at position {closing[2]}"
                )
            self._advance()
            return result
        if tok[0] == 'RPAREN':
            raise ValueError(f"Mismatched parentheses: unexpected ')' at position {tok[2]}")
        raise ValueError(f"Unexpected token '{tok[1]}' at position {tok[2]}")
```

## `test_expression_evaluator.py`

```python
"""Tests for the recursive descent expression evaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


def test_basic_arithmetic() -> None:
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3") == 5.0
    assert ev.evaluate("10 - 4") == 6.0
    assert ev.evaluate("7 * 8") == 56.0
    assert ev.evaluate("9 / 2") == 4.5
    assert ev.evaluate("0.1 + 0.2") == pytest.approx(0.3)
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_operator_precedence() -> None:
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0
    assert ev.evaluate("10 / 2 - 3") == 2.0
    assert ev.evaluate("20 / 5 * 2") == 8.0


def test_parentheses() -> None:
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("((1 + 2) * 3)") == 9.0
    assert ev.evaluate("2 * (3 + 4)") == 14.0
    assert ev.evaluate("((2 + 3) * (4 - 1))") == 15.0


def test_unary_minus() -> None:
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("2 * -3") == -6.0
    assert ev.evaluate("-2 + -3") == -5.0
    assert ev.evaluate("--5") == 5.0


def test_error_cases() -> None:
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="empty"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="empty"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("1 + 2)")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 @ 3")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / (2 - 2)")
```

## Run it

```bash
pytest test_expression_evaluator.py -v
```

## Design notes

- **Precedence via grammar layers**: `expression` (additive) → `term` (multiplicative) → `unary` → `primary`. Lower-precedence operations call higher-precedence ones, which gives standard precedence and left-associativity for free.
- **Unary minus** is handled in `_parse_unary`, recursively, so `-3`, `--5`, and `-(2 + 1)` all work.
- **Errors**: the tokenizer reports invalid characters/numbers with position, the parser reports mismatched parentheses explicitly (missing `)` vs. unexpected `)`), division by zero is caught at the `/` node, and a final trailing-token check catches junk like `2 3` or `2)`.
- **Results** are always returned as `float`, and every method carries a type hint and docstring as required. The evaluator is re-entrant-safe since `evaluate()` fully resets its state on each call.