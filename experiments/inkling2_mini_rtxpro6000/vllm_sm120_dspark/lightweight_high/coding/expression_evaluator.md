Here's a complete implementation using a recursive descent parser (no `eval`/`ast.literal_eval`), split into two files.

## `expression_evaluator.py`

```python
"""Recursive descent parser and evaluator for arithmetic expressions."""

from typing import List, Optional, Tuple


class ExpressionEvaluator:
    """Evaluate arithmetic expressions with +, -, *, /, parentheses and unary minus."""

    def __init__(self) -> None:
        """Initialize the evaluator's internal token-stream state."""
        self._tokens: List[Tuple[str, str, int]] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate `expr` and return the result as a float.

        Args:
            expr: The arithmetic expression to evaluate.

        Returns:
            The numeric result of the expression.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise ValueError(f"Expression must be a string, got {type(expr).__name__}")
        if not expr.strip():
            raise ValueError("Expression is empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            leftover = self._tokens[self._pos]
            if leftover[0] == 'RPAREN':
                raise ValueError(
                    f"Mismatched parentheses: unexpected closing parenthesis "
                    f"at position {leftover[2]}"
                )
            raise ValueError(
                f"Unexpected token '{leftover[1]}' at position {leftover[2]}"
            )
        return float(result)

    def _tokenize(self, expr: str) -> List[Tuple[str, str, int]]:
        """Convert a raw expression string into a list of (kind, text, position) tokens."""
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
                dots = 0
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        dots += 1
                    i += 1
                text = expr[start:i]
                if dots > 1 or text == '.':
                    raise ValueError(f"Invalid number '{text}' at position {start}")
                tokens.append(('NUMBER', text, start))
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

    def _peek(self) -> Optional[Tuple[str, str, int]]:
        """Return the current token without consuming it, or None at end of input."""
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos]

    def _advance(self) -> Tuple[str, str, int]:
        """Consume and return the current token."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse a left-associative sequence of '+'/'-' terms (lowest precedence)."""
        left = self._parse_term()
        token = self._peek()
        while token is not None and token[0] in ('+', '-'):
            op = self._advance()[0]
            right = self._parse_term()
            left = left + right if op == '+' else left - right
            token = self._peek()
        return left

    def _parse_term(self) -> float:
        """Parse a left-associative sequence of '*'/'/' factors."""
        left = self._parse_unary()
        token = self._peek()
        while token is not None and token[0] in ('*', '/'):
            op = self._advance()[0]
            right = self._parse_unary()
            if op == '*':
                left = left * right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                left = left / right
            token = self._peek()
        return left

    def _parse_unary(self) -> float:
        """Parse an optional unary minus/plus followed by a primary expression."""
        token = self._peek()
        if token is not None and token[0] == '-':
            self._advance()
            return -self._parse_unary()
        if token is not None and token[0] == '+':
            self._advance()
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a number literal or a parenthesized expression (highest precedence)."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression: expected a number or '('")
        kind = token[0]
        if kind == 'NUMBER':
            self._advance()
            return float(token[1])
        if kind == 'LPAREN':
            self._advance()
            result = self._parse_expression()
            closing = self._peek()
            if closing is None:
                raise ValueError("Mismatched parentheses: missing closing parenthesis")
            if closing[0] != 'RPAREN':
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found '{closing[1]}' "
                    f"at position {closing[2]}"
                )
            self._advance()
            return result
        if kind == 'RPAREN':
            raise ValueError(
                f"Mismatched parentheses: unexpected closing parenthesis at position {token[2]}"
            )
        raise ValueError(f"Unexpected token '{token[1]}' at position {token[2]}")
```

## `test_expression_evaluator.py`

```python
"""Tests for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator():
    """Return a fresh evaluator instance for each test."""
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator):
    """Basic +, -, *, / operations."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("8 / 2") == 4.0
    assert evaluator.evaluate("3.14 + 1.86") == pytest.approx(5.0)


def test_operator_precedence(evaluator):
    """Multiplication and division bind tighter than addition and subtraction."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 / 2") == 8.0
    assert evaluator.evaluate("100 / 10 / 2") == 5.0  # left associativity


def test_parentheses(evaluator):
    """Parentheses override operator precedence."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * (10 - 7))") == 15.0
    assert evaluator.evaluate("((1 + 2) * 3) - 4") == 5.0


def test_unary_minus(evaluator):
    """Unary minus applies to literals, expressions, and after binary operators."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0
    assert evaluator.evaluate("--3") == 3.0
    assert evaluator.evaluate("-3.14") == pytest.approx(-3.14)


def test_error_cases(evaluator):
    """Invalid input raises ValueError with a descriptive message."""
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("1 + 2)")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + @")
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate("   ")
```

## Run the tests

```bash
python -m pytest test_expression_evaluator.py -v
```

## Design notes

- **Grammar** (recursive descent, lowest → highest precedence): `expression → term (('+'|'-') term)*` → `term → unary (('*'|'/') unary)*` → `unary → '-' unary | '+' unary | primary` → `primary → NUMBER | '(' expression ')'`. This gives correct precedence and left associativity for free.
- **Tokenizer** handles whitespace, floats (`3.14`, `.5`, `3.`), and reports the source position of invalid tokens/numbers.
- **Errors** are raised as `ValueError` at the point of detection: mismatched parens are detected both inside `_parse_primary` (missing closing) and in `evaluate` (stray closing), and division-by-zero is checked explicitly in `_parse_term` so it fires even inside nested expressions like `2 / (3 - 3)`.