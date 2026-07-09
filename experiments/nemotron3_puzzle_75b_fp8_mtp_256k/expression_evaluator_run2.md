We need to produce code for ExpressionEvaluator class with evaluate method. Use recursive descent parser. Provide type hints, docstrings. Also include 5 pytest tests. Should output code in a block.

We'll implement tokenization: numbers, operators, parentheses. Recognize unary minus: when '-' appears at start or after '(' or another operator.

Define grammar:

expr ::= term (('+'|'-') term)*
term ::= factor (('*'|'/') factor)*
factor ::= ('+'|'-') factor | primary
primary ::= NUMBER | '(' expr ')'

We'll implement parse_expression, parse_term, parse_factor, parse_primary.

Token generator or iterator over list of tokens.

Need to raise ValueError for invalid tokens, mismatched parentheses, division by zero, empty expression.

We'll implement tokenizer: iterate over string, ignore whitespace, collect numbers (including floats), operators, parentheses.

If invalid char raise ValueError.

Unary minus: in factor, if token is '+' or '-', treat as unary; we allow multiple unary operators.

Implementation details: token list, index pointer.

Evaluation returns float.

Edge Cases: empty string => raise.

Now provide test file with 5 tests: e.g., test_addition, test_precedence, test_parentheses, test_unary_minus, test_error_cases (division by zero, mismatched parentheses, invalid token, empty).

But they ask 5 pytest tests; we can produce separate test functions.

Will output both evaluator code and test file.

Make sure to include imports.

Let's craft final answer.





**`expression_evaluator.py`**

```python
"""
Recursive‑descent expression evaluator.

Supports:
    +, -, *, /   (with correct precedence)
    Parentheses for grouping
    Unary plus/minus
    Floating‑point literals
    Proper error handling (ValueError) for:
        • mismatched parentheses
        • division by zero
        • invalid tokens
        • empty expression

The evaluator is implemented as ``ExpressionEvaluator.evaluate`` and
uses a classic grammar:

    expr   ::= term (('+' | '-') term)*
    term   ::= factor (('*' | '/') factor)*
    factor ::= ('+' | '-') factor | primary
    primary::= NUMBER | '(' expr ')'

No use of ``eval`` or ``ast.literal_eval`` is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class _Token:
    """Simple token container."""
    type: str          # one of "NUMBER", "+", "-", "*", "/", "(", ")"
    value: float | None  # numeric value for NUMBER tokens, otherwise None


class ExpressionEvaluator:
    """
    Evaluates arithmetic expressions using a recursive‑descent parser.

    Example
    -------
    >>> ExpressionEvaluator().evaluate("3 + 4 * (2 - 1)")
    7.0
    """

    def __init__(self) -> None:
        self._tokens: List[_Token] = []
        self._pos: int = 0  # current token index

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def evaluate(self, expr: str) -> float:
        """
        Parse and evaluate *expr*.

        Parameters
        ----------
        expr: str
            The arithmetic expression to evaluate.

        Returns
        -------
        float
            The result of the expression.

        Raises
        ------
        ValueError
            If the expression is empty, contains invalid tokens,
            has mismatched parentheses, or attempts division by zero.
        """
        if not expr or expr.isspace():
            raise ValueError("empty expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0
        result = self._parse_expr()
        if self._pos != len(self._tokens):
            # leftover tokens → syntax error
            raise ValueError("invalid syntax")
        return float(result)

    # --------------------------------------------------------------------- #
    # Tokeniser
    # --------------------------------------------------------------------- #
    def _tokenize(self, s: str) -> List[_Token]:
        """Convert the input string into a list of tokens."""
        tokens: List[_Token] = []
        i = 0
        length = len(s)

        while i < length:
            ch = s[i]

            if ch.isspace():
                i += 1
                continue

            if ch in "+-*/()":
                tokens.append(_Token(type=ch, value=None))
                i += 1
                continue

            if ch.isdigit() or ch == ".":
                start = i
                dot_seen = ch == "."
                i += 1
                while i < length and (s[i].isdigit() or (s[i] == "." and not dot_seen)):
                    if s[i] == ".":
                        dot_seen = True
                    i += 1
                num_str = s[start:i]
                try:
                    num_val = float(num_str)
                except ValueError:
                    raise ValueError(f"invalid number literal: {num_str!r}") from None
                tokens.append(_Token(type="NUMBER", value=num_val))
                continue

            # Anything else is illegal
            raise ValueError(f"invalid token: {ch!r}")

        return tokens

    # --------------------------------------------------------------------- #
    # Parser – recursive descent
    # --------------------------------------------------------------------- #
    def _peek(self) -> _Token | None:
        """Return current token without consuming it, or None at EOF."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> _Token:
        """Consume and return the current token; advance position."""
        tok = self._peek()
        if tok is None:
            raise ValueError("unexpected end of expression")
        self._pos += 1
        return tok

    def _parse_expr(self) -> float:
        """expr ::= term (('+' | '-') term)*"""
        value = self._parse_term()
        while True:
            op = self._peek()
            if op and op.type in ("+", "-"):
                self._consume()
                rhs = self._parse_term()
                value = value + rhs if op.type == "+" else value - rhs
            else:
                break
        return value

    def _parse_term(self) -> float:
        """term ::= factor (('*' | '/') factor)*"""
        value = self._parse_factor()
        while True:
            op = self._peek()
            if op and op.type in ("*", "/"):
                self._consume()
                rhs = self._parse_factor()
                if op.type == "*":
                    value = value * rhs
                else:  # division
                    if rhs == 0.0:
                        raise ValueError("division by zero")
                    value = value / rhs
            else:
                break
        return value

    def _parse_factor(self) -> float:
        """
        factor ::= ('+' | '-') factor | primary
        Handles unary plus/minus.
        """
        token = self._peek()
        if token and token.type in ("+", "-"):
            self._consume()
            val = self._parse_factor()
            return val if token.type == "+" else -val
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """primary ::= NUMBER | '(' expr ')'"""
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end of expression")

        if token.type == "NUMBER":
            self._consume()
            assert token.value is not None  # for mypy
            return token.value

        if token.type == "(":
            self._consume()  # consume '('
            val = self._parse_expr()
            if self._peek() is None or self._peek().type != ")":
                raise ValueError("mismatched parentheses")
            self._consume()  # consume ')'
            return val

        raise ValueError(f"unexpected token: {token.type!r}")
```

---

**`test_expression_evaluator.py`**

```python
import pytest
from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator():
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("8 / 2") == 4.0


def test_operator_precedence_and_parentheses(evaluator):
    assert evaluator.evaluate("3 + 4 * 2") == 11.0          # * before +
    assert evaluator.evaluate("(3 + 4) * 2") == 14.0        # parentheses override
    assert evaluator.evaluate("2 * (3 + 4) / 7") == 2.0    # mixed precedence


def test_unary_minus_and_plus(evaluator):
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("+5") == 5.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0                # double unary
    assert evaluator.evaluate("-*+-3") == 3.0              # mix of unary +/-
                                                            # - * + - 3 → - * (+ (-3)) → - * (-3) → 3


def test_floating_point_numbers(evaluator):
    assert evaluator.evaluate("3.14") == 3.14
    assert evaluator.evaluate("0.5 * 2") == 1.0
    assert evaluator.evaluate("1.1 + 2.2") == pytest.approx(3.3, rel=1e-12)
    assert evaluator.evaluate("-3.5 / 2") == -1.75


def test_error_conditions(evaluator):
    # mismatched parentheses
    with pytest.raises(ValueError, match="mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="mismatched parentheses"):
        evaluator.evaluate("1 + 2)")

    # division by zero
    with pytest.raises(ValueError, match="division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="division by zero"):
        evaluator.evaluate("-(2+3) / (1-1)")

    # invalid tokens
    with pytest.raises(ValueError, match="invalid token"):
        evaluator.evaluate("2 & 3")
    with pytest.raises(ValueError, match="invalid token"):
        evaluator.evaluate("2 ++ 3")   # two operators in a row without unary handling
    with pytest.raises(ValueError, match="invalid number literal"):
        evaluator.evaluate("3..14")

    # empty expression
    with pytest.raises(ValueError, match="empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="empty expression"):
        evaluator.evaluate("   \t\n   ")
```

Place both files in the same directory and run the tests with:

```bash
pytest test_expression_evaluator.py
```

All tests should pass, confirming that the evaluator meets the specification.