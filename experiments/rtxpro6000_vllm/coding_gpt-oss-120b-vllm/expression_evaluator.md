**`expression_evaluator.py`**

```python
"""Expression evaluator based on a recursive‑descent parser.

The evaluator supports:
* binary operators ``+ - * /`` with the usual precedence,
* parentheses for explicit grouping,
* unary minus (e.g. ``-3`` or ``-(2+1)``),
* floating‑point literals (e.g. ``3.14``).

Invalid input raises ``ValueError`` with a clear message:
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

    The public API consists of a single method ``evaluate`` that receives a
    string and returns the computed ``float`` value.
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

    def _tokenise(self, expr: str) -> List[Token]:
        """Convert the input string into a list of (type, value) tokens.

        Raises:
            ValueError: If an illegal character is found.
        """
        tokens: List[Token] = []
        for mo in self._TOKEN_REGEX.finditer(expr):
            kind = mo.lastgroup
            value = mo.group()
            if kind == "NUMBER":
                tokens.append((kind, value))
            elif kind in {"PLUS", "MINUS", "TIMES", "DIVIDE",
                          "LPAREN", "RPAREN"}:
                tokens.append((kind, value))
            elif kind == "SKIP":
                continue
            else:  # MISMATCH
                raise ValueError(f"Invalid token '{value}' in expression")
        return tokens

    # --------------------------------------------------------------------- #
    # Recursive‑descent parser
    # --------------------------------------------------------------------- #
    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Parse *expr* and return its numeric value.

        Args:
            expr: The arithmetic expression to evaluate.

        Returns:
            The result as a ``float``.

        Raises:
            ValueError: For empty input, mismatched parentheses,
                division by zero, or any lexical/syntactic error.
        """
        if not expr or expr.strip() == "":
            raise ValueError("Empty expression")

        self._tokens = self._tokenise(expr)
        self._pos = 0
        result = self._parse_expression()

        if self._pos != len(self._tokens):
            # Something left over that could not be parsed
            leftover = self._tokens[self._pos][1]
            raise ValueError(f"Invalid token '{leftover}' after complete parse")

        return result

    # --------------------------------------------------------------------- #
    # Grammar helpers
    # --------------------------------------------------------------------- #
    # expression ::= term ((PLUS | MINUS) term)*
    def _parse_expression(self) -> float:
        """Parse an expression handling ``+`` and ``-``."""
        value = self._parse_term()
        while self._current_token_type() in {"PLUS", "MINUS"}:
            op = self._consume()[0]
            right = self._parse_term()
            if op == "PLUS":
                value += right
            else:
                value -= right
        return value

    # term ::= factor ((TIMES | DIVIDE) factor)*
    def _parse_term(self) -> float:
        """Parse a term handling ``*`` and ``/``."""
        value = self._parse_factor()
        while self._current_token_type() in {"TIMES", "DIVIDE"}:
            op = self._consume()[0]
            right = self._parse_factor()
            if op == "TIMES":
                value *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                value /= right
        return value

    # factor ::= (PLUS | MINUS) factor | primary
    def _parse_factor(self) -> float:
        """Parse a factor, supporting unary ``+``/``-``."""
        token_type = self._current_token_type()
        if token_type == "PLUS":
            self._consume()
            return self._parse_factor()
        if token_type == "MINUS":
            self._consume()
            return -self._parse_factor()
        return self._parse_primary()

    # primary ::= NUMBER | LPAREN expression RPAREN
    def _parse_primary(self) -> float:
        """Parse a primary expression: a number or a parenthesised sub‑expression."""
        token_type = self._current_token_type()
        if token_type == "NUMBER":
            value_str = self._consume()[1]
            return float(value_str)
        if token_type == "LPAREN":
            self._consume()                     # '('
            inner = self._parse_expression()
            if self._current_token_type() != "RPAREN":
                raise ValueError("Mismatched parentheses")
            self._consume()                     # ')'
            return inner
        raise ValueError(f"Unexpected token '{self._current_token_value()}'")

    # --------------------------------------------------------------------- #
    # Utility helpers
    # --------------------------------------------------------------------- #
    def _current_token_type(self) -> Union[str, None]:
        """Return the type of the current token or ``None`` if at end."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos][0]
        return None

    def _current_token_value(self) -> Union[str, None]:
        """Return the raw value of the current token or ``None`` if at end."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos][1]
        return None

    def _consume(self) -> Token:
        """Consume the current token and advance the cursor.

        Returns:
            The consumed token.

        Raises:
            ValueError: If there is no token to consume.
        """
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression")
        token = self._tokens[self._pos]
        self._pos += 1
        return token
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
    assert evaluator.evaluate("2+3*4") == 14.0          # 2 + (3*4)
    assert evaluator.evaluate("10-5/5") == 9.0        # 10 - (5/5)


def test_precedence_and_parentheses(evaluator: ExpressionEvaluator) -> None:
    # Parentheses override precedence
    assert evaluator.evaluate("(2+3)*4") == 20.0
    # Nested parentheses
    assert evaluator.evaluate("((1+2)+3)*2") == 12.0


def test_unary_minus_and_floats(evaluator: ExpressionEvaluator) -> None:
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("- (2 + 1) * 2") == -6.0
    assert evaluator.evaluate("3.5 * 2") == 7.0
    assert evaluator.evaluate("-0.5 * -4") == 2.0


@pytest.mark.parametrize(
    "expr, message",
    [
        ("(1+2", "Mismatched parentheses"),
        ("4/0", "Division by zero"),
        ("2 & 3", "Invalid token"),
        ("", "Empty expression"),
    ],
)
def test_error_cases(evaluator: ExpressionEvaluator, expr: str, message: str) -> None:
    with pytest.raises(ValueError) as exc:
        evaluator.evaluate(expr)
    assert message in str(exc.value)
```

**Explanation of the tests**

| Test | What it checks |
|------|----------------|
| `test_basic_arithmetic` | Simple binary operators and left‑to‑right associativity. |
| `test_precedence_and_parentheses` | Correct precedence (`*` over `+`) and that parentheses change the order of evaluation. |
| `test_unary_minus_and_floats` | Unary minus, combination of unary and binary operators, and handling of floating‑point literals. |
| `test_error_cases` (parametrised) | Detects mismatched parentheses, division by zero, illegal characters, and empty input, each raising a descriptive `ValueError`. |

Run the tests with:

```bash
pytest test_expression_evaluator.py
```

All tests should pass, confirming that the `ExpressionEvaluator` meets the specified requirements.