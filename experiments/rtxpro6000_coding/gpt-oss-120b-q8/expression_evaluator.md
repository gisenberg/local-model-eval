**`expression_evaluator.py`**

```python
"""Expression evaluator based on a recursive‑descent parser.

The evaluator supports:
* binary operators ``+ - * /`` with the usual precedence,
* parentheses for explicit grouping,
* unary minus (e.g. ``-3`` or ``-(2+1)``),
* floating‑point literals (e.g. ``3.14``).

Invalid input raises ``ValueError`` with a clear description:
* mismatched parentheses,
* division by zero,
* unknown/illegal tokens,
* empty expression.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Union


Token = Tuple[str, str]  # (type, value)


class ExpressionEvaluator:
    """Parse and evaluate a mathematical expression.

    The public API consists of a single method ``evaluate`` which returns the
    floating‑point result of the supplied expression string.
    """

    # --------------------------------------------------------------------- #
    # Tokenisation
    # --------------------------------------------------------------------- #
    _TOKEN_SPEC = [
        ("NUMBER",   r"\d+(?:\.\d*)?|\.\d+"),   # integer or decimal number
        ("PLUS",     r"\+"),
        ("MINUS",    r"-"),
        ("TIMES",    r"\*"),
        ("DIVIDE",   r"/"),
        ("LPAREN",   r"\("),
        ("RPAREN",   r"\)"),
        ("SKIP",     r"[ \t\r\n]+"),            # whitespace – ignored
        ("MISMATCH", r"."),                    # any other character
    ]
    _TOKEN_REGEX = re.compile("|".join(f"(?P<{name}>{pattern})"
                                      for name, pattern in _TOKEN_SPEC))

    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pos: int = 0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def evaluate(self, expr: str) -> float:
        """Return the numeric value of *expr*.

        Parameters
        ----------
        expr: str
            The arithmetic expression to evaluate.

        Returns
        -------
        float
            The evaluated result.

        Raises
        ------
        ValueError
            If the expression is empty, contains illegal characters,
            has mismatched parentheses, or attempts division by zero.
        """
        expr = expr.strip()
        if not expr:
            raise ValueError("empty expression")

        self._tokens = self._tokenise(expr)
        self._pos = 0
        result = self._parse_expression()

        if self._pos != len(self._tokens):
            # Something left over that could not be parsed
            token_type, token_val = self._tokens[self._pos]
            raise ValueError(f"invalid token '{token_val}' at position {self._pos}")

        return result

    # --------------------------------------------------------------------- #
    # Tokeniser
    # --------------------------------------------------------------------- #
    def _tokenise(self, text: str) -> List[Token]:
        """Convert *text* into a list of (type, value) tokens.

        Raises
        ------
        ValueError
            If an illegal character is encountered.
        """
        tokens: List[Token] = []
        for mo in self._TOKEN_REGEX.finditer(text):
            kind = mo.lastgroup
            value = mo.group()
            if kind == "NUMBER":
                tokens.append((kind, value))
            elif kind in {"PLUS", "MINUS", "TIMES", "DIVIDE", "LPAREN", "RPAREN"}:
                tokens.append((kind, value))
            elif kind == "SKIP":
                continue
            else:  # MISMATCH
                raise ValueError(f"invalid token '{value}' at position {mo.start()}")
        return tokens

    # --------------------------------------------------------------------- #
    # Recursive‑descent parsing helpers
    # --------------------------------------------------------------------- #
    def _peek(self) -> Union[Token, None]:
        """Return the next token without consuming it, or ``None`` if at EOF."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self, expected_type: str) -> Token:
        """Consume the next token if it matches *expected_type*; otherwise raise."""
        token = self._peek()
        if token is None or token[0] != expected_type:
            raise ValueError(f"expected token type {expected_type!r} but got "
                             f"{token[0] if token else 'EOF'}")
        self._pos += 1
        return token

    # Grammar (EBNF):
    #
    # expression ::= term ((PLUS | MINUS) term)*
    # term       ::= factor ((TIMES | DIVIDE) factor)*
    # factor     ::= (PLUS | MINUS) factor | primary
    # primary    ::= NUMBER | LPAREN expression RPAREN
    #
    # The grammar respects the usual precedence and left‑associativity.

    def _parse_expression(self) -> float:
        """Parse ``expression`` and return its value."""
        value = self._parse_term()
        while True:
            token = self._peek()
            if token and token[0] in {"PLUS", "MINUS"}:
                op_type, _ = self._consume(token[0])
                right = self._parse_term()
                if op_type == "PLUS":
                    value += right
                else:
                    value -= right
            else:
                break
        return value

    def _parse_term(self) -> float:
        """Parse ``term`` and return its value."""
        value = self._parse_factor()
        while True:
            token = self._peek()
            if token and token[0] in {"TIMES", "DIVIDE"}:
                op_type, _ = self._consume(token[0])
                right = self._parse_factor()
                if op_type == "TIMES":
                    value *= right
                else:
                    if right == 0.0:
                        raise ValueError("division by zero")
                    value /= right
            else:
                break
        return value

    def _parse_factor(self) -> float:
        """Parse ``factor`` (handles unary plus/minus) and return its value."""
        token = self._peek()
        if token and token[0] in {"PLUS", "MINUS"}:
            op_type, _ = self._consume(token[0])
            operand = self._parse_factor()
            return +operand if op_type == "PLUS" else -operand
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse ``primary`` – a number or a parenthesised expression."""
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end of expression")

        if token[0] == "NUMBER":
            _, value = self._consume("NUMBER")
            return float(value)
        if token[0] == "LPAREN":
            self._consume("LPAREN")
            inner = self._parse_expression()
            if self._peek() is None or self._peek()[0] != "RPAREN":
                raise ValueError("mismatched parentheses")
            self._consume("RPAREN")
            return inner

        raise ValueError(f"invalid token '{token[1]}' at position {self._pos}")
```

---

**`test_expression_evaluator.py`**

```python
import pytest
from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("7 - 4") == 3.0
    assert evaluator.evaluate("6 * 5") == 30.0
    assert evaluator.evaluate("8 / 2") == 4.0


def test_operator_precedence(evaluator: ExpressionEvaluator) -> None:
    # * and / have higher precedence than + and -
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("2 + 6 / 3") == 4.0
    # left‑associative
    assert evaluator.evaluate("8 - 4 - 1") == 3.0   # (8-4)-1
    assert evaluator.evaluate("8 / 4 / 2") == 1.0   # (8/4)/2


def test_parentheses_grouping(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4) / (1 + 1)") == 7.0
    # nested parentheses
    assert evaluator.evaluate("-( (1 + 2) * (3 - 4) )") == 3.0


def test_unary_minus(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("- (2 + 1)") == -3.0
    assert evaluator.evaluate("5 + -2") == 3.0
    assert evaluator.evaluate("5 * -2") == -10.0
    # double unary minus -> positive
    assert evaluator.evaluate("--5") == 5.0


def test_error_cases(evaluator: ExpressionEvaluator) -> None:
    # empty expression
    with pytest.raises(ValueError, match="empty expression"):
        evaluator.evaluate("   ")

    # mismatched parentheses
    with pytest.raises(ValueError, match="mismatched parentheses"):
        evaluator.evaluate("(1 + 2")

    # division by zero
    with pytest.raises(ValueError, match="division by zero"):
        evaluator.evaluate("4 / (2 - 2)")

    # invalid token
    with pytest.raises(ValueError, match="invalid token '@'"):
        evaluator.evaluate("2 + @")
```

**Explanation**

* `ExpressionEvaluator` implements a classic recursive‑descent parser:
  * `expression → term ((+|-) term)*`
  * `term → factor ((*|/) factor)*`
  * `factor → (+|-) factor | primary`
  * `primary → NUMBER | '(' expression ')'`
* Tokenisation is performed with a single regular expression; any character that does not match the defined token set raises a `ValueError`.
* Division by zero is detected at runtime and reported with a clear message.
* Parentheses are checked both when opening and when expecting the closing token; a missing `)` triggers a “mismatched parentheses” error.
* The public `evaluate` method strips whitespace, validates emptiness, tokenises, parses, and finally ensures that no stray tokens remain.
* The accompanying pytest suite covers:
  1. Basic binary operations,
  2. Operator precedence and associativity,
  3. Grouping with parentheses (including nesting),
  4. Unary minus handling,
  5. All required error conditions.