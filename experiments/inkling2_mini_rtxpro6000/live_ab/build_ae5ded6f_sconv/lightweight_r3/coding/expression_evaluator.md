Here's a complete implementation using a recursive descent parser (grammar: expression → term → factor → primary), plus the pytest suite.

## `expression_evaluator.py`

```python
"""Recursive descent parser/evaluator for basic arithmetic expressions."""

from typing import List, Optional


class ExpressionEvaluator:
    """Evaluates arithmetic expressions with +, -, *, /, parentheses,
    unary minus, and floating-point literals.

    Uses a recursive descent parser; never calls eval() or ast.literal_eval().
    State is reset on every evaluate() call, so the same instance is reusable.
    """

    def __init__(self) -> None:
        """Initialize the evaluator with empty parser state."""
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate an arithmetic expression.

        Args:
            expr: The expression string, e.g. "3.14 * (2 + 1)".

        Returns:
            The result of the expression as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                has mismatched parentheses or leftover tokens, or divides by zero.
        """
        if not isinstance(expr, str):
            raise ValueError(f"Expression must be a string, got {type(expr).__name__}")
        if expr.strip() == "":
            raise ValueError("Expression is empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            leftover = self._tokens[self._pos]
            if leftover == ')':
                raise ValueError("Mismatched parentheses: unexpected ')'")
            raise ValueError(f"Unexpected token '{leftover}' at position {self._pos}")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """Split the expression string into number and operator tokens.

        Args:
            expr: The raw expression string.

        Returns:
            A list of token strings (e.g. ["3", ".", "14"] -> ["3.14"]).

        Raises:
            ValueError: If an invalid character or malformed number is found.
        """
        tokens: List[str] = []
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
                token = expr[start:i]
                try:
                    float(token)
                except ValueError:
                    raise ValueError(f"Invalid number '{token}' at position {start}") from None
                tokens.append(token)
                continue
            if ch in "+-*/()":
                tokens.append(ch)
                i += 1
                continue
            raise ValueError(f"Invalid token '{ch}' at position {i}")
        return tokens

    def _peek(self) -> Optional[str]:
        """Return the current token without consuming it, or None at end of input."""
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos]

    def _advance(self) -> str:
        """Consume and return the current token."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse expression := term (('+' | '-') term)*  (lowest precedence)."""
        result = self._parse_term()
        while self._peek() in ("+", "-"):
            op = self._advance()
            rhs = self._parse_term()
            result = result + rhs if op == "+" else result - rhs
        return result

    def _parse_term(self) -> float:
        """Parse term := factor (('*' | '/') factor)*  (higher precedence)."""
        result = self._parse_factor()
        while self._peek() in ("*", "/"):
            op = self._advance()
            rhs = self._parse_factor()
            if op == "*":
                result = result * rhs
            else:
                if rhs == 0:
                    raise ValueError("Division by zero")
                result = result / rhs
        return result

    def _parse_factor(self) -> float:
        """Parse factor := ('-' | '+') factor | primary  (unary operators)."""
        token = self._peek()
        if token == "-":
            self._advance()
            return -self._parse_factor()
        if token == "+":
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse primary := number | '(' expression ')'."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression: expected a number or '('")
        if token == "(":
            self._advance()
            result = self._parse_expression()
            if self._peek() != ")":
                found = self._peek() or "end of expression"
                raise ValueError(f"Mismatched parentheses: expected ')' but found {found!r}")
            self._advance()
            return result
        try:
            value = float(token)
        except ValueError:
            raise ValueError(f"Unexpected token '{token}': expected a number or '('") from None
        self._advance()
        return value


if __name__ == "__main__":
    evaluator = ExpressionEvaluator()
    for expression in ["3 + 4 * 2", "(3 + 4) * 2", "-3.14 * 2", "10 / 4 / 2"]:
        print(f"{expression!r} = {evaluator.evaluate(expression)}")
```

## `test_expression_evaluator.py`

```python
"""Pytest suite for ExpressionEvaluator."""

import pytest

from expression_evaluator import ExpressionEvaluator


def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 2") == 4.0
    assert evaluator.evaluate("10 - 3") == 7.0
    assert evaluator.evaluate("4 * 5") == 20.0
    assert evaluator.evaluate("20 / 4") == 5.0
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_operator_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0          # * binds tighter than +
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    assert evaluator.evaluate("10 / 2 / 5") == pytest.approx(1.0)  # left-associative


def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0


def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 + -2") == 1.0
    assert evaluator.evaluate("--5") == 5.0


def test_error_cases():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="[Mm]ismatched"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="[Mm]ismatched"):
        evaluator.evaluate("1 + 2)")
    with pytest.raises(ValueError, match="[Dd]ivision by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 & 3")
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("1.2.3")
    with pytest.raises(ValueError, match="must be a string"):
        evaluator.evaluate(123)
```

## How it works

- **Tokenizer** splits the input into numbers, operators, and parentheses, validating number format (rejecting e.g. `1.2.3`) and rejecting unknown characters with a position-tagged error.
- **Parser** (recursive descent, four levels):
  - `expression` handles `+`/`-` (lowest precedence, left-associative)
  - `term` handles `*`/`/` (higher precedence, left-associative)
  - `factor` handles unary `-`/`+`
  - `primary` parses a number or a parenthesized sub-expression
- **Error handling**: empty input, invalid characters/numbers, mismatched parentheses (both `(1 + 2` and `1 + 2)`), trailing tokens, and division by zero all raise `ValueError` with descriptive messages.

Run the tests with:

```bash
python -m pytest test_expression_evaluator.py -v
```

Notes: the evaluator intentionally supports only the required operators, `+`/`-`/`*`/`/`, parentheses, unary minus, and decimal literals — no exponentiation, functions, or variables. All results are returned as `float`, including for expressions with integer-valued results.