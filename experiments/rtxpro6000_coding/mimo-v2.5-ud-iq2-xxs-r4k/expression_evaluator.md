Here's the complete implementation:

```python
"""
Mathematical Expression Evaluator
=================================
A recursive descent parser that evaluates arithmetic expressions with
operator precedence, parentheses, unary minus, and floating point numbers.
"""

import re
from typing import List, Optional


class ExpressionEvaluator:
    """
    Evaluates mathematical expressions using a recursive descent parser.

    Supports +, -, *, / with correct operator precedence,
    parentheses for grouping, unary minus, and floating point numbers.
    """

    def __init__(self) -> None:
        """Initialize the evaluator with an empty token list and index."""
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.

        Args:
            expr: The expression string to evaluate.

        Returns:
            The result of the expression as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr.strip():
            raise ValueError("Empty expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(
                f"Unexpected token '{self._tokens[self._pos]}' at position {self._pos}"
            )

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert an expression string into a list of tokens.

        Args:
            expr: The expression string to tokenize.

        Returns:
            A list of tokens (numbers, operators, parentheses).

        Raises:
            ValueError: If the expression contains invalid characters.
        """
        tokens: List[str] = []
        i = 0
        while i < len(expr):
            char = expr[i]
            if char.isspace():
                i += 1
                continue
            if char in "+-*/()":
                tokens.append(char)
                i += 1
            elif char.isdigit() or char == '.':
                start = i
                has_dot = False
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(
                                f"Invalid number at position {start}: "
                                f"'{expr[start:i+1]}'"
                            )
                        has_dot = True
                    i += 1
                num_str = expr[start:i]
                if num_str == '.':
                    raise ValueError(
                        f"Invalid number at position {start}: '{num_str}'"
                    )
                tokens.append(num_str)
            else:
                raise ValueError(
                    f"Invalid character '{char}' at position {i}"
                )
        return tokens

    def _parse_expression(self) -> float:
        """
        Parse an expression: term (('+' | '-') term)*

        Returns:
            The result of the expression.

        Raises:
            ValueError: If the expression is invalid.
        """
        result = self._parse_term()
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ("+", "-"):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_term()
            if op == "+":
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """
        Parse a term: factor (('*' | '/') factor)*

        Returns:
            The result of the term.

        Raises:
            ValueError: If division by zero is attempted.
        """
        result = self._parse_factor()
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ("*", "/"):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_factor()
            if op == "*":
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """
        Parse a factor: '-' factor | number | '(' expression ')'

        Returns:
            The result of the factor.

        Raises:
            ValueError: If the expression is invalid or has mismatched parentheses.
        """
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression")

        token = self._tokens[self._pos]

        if token == "-":
            self._pos += 1
            return -self._parse_factor()

        if token == "(":
            self._pos += 1
            result = self._parse_expression()
            if self._pos >= len(self._tokens) or self._tokens[self._pos] != ")":
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._pos += 1
            return result

        if token == "+":
            self._pos += 1
            return self._parse_factor()

        if token == ")":
            raise ValueError("Mismatched parentheses: unexpected ')'")

        try:
            value = float(token)
            self._pos += 1
            return value
        except ValueError:
            raise ValueError(f"Invalid token '{token}'")


# ─────────────────────────────────────────────────────────────────────────────
# Pytest Tests
# ─────────────────────────────────────────────────────────────────────────────

import pytest


class TestExpressionEvaluator:
    """Test suite for ExpressionEvaluator."""

    def setup_method(self) -> None:
        """Set up the evaluator before each test."""
        self.evaluator = ExpressionEvaluator()

    def test_basic_arithmetic(self) -> None:
        """Test basic arithmetic operations: +, -, *, /."""
        assert self.evaluator.evaluate("2 + 3") == 5.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("3 * 7") == 21.0
        assert self.evaluator.evaluate("15 / 3") == 5.0
        assert self.evaluator.evaluate("3.14 + 2.86") == 6.0

    def test_operator_precedence(self) -> None:
        """Test that * and / have higher precedence than + and -."""
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
        assert self.evaluator.evaluate("10 - 2 * 3") == 4.0
        assert self.evaluator.evaluate("6 / 2 + 1") == 4.0
        assert self.evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
        assert self.evaluator.evaluate("1 + 2 * 3 - 4 / 2") == 5.0

    def test_parentheses(self) -> None:
        """Test parentheses for grouping."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert self.evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert self.evaluator.evaluate("(1 + 2) * (3 + 4)") == 21.0
        assert self.evaluator.evaluate("((2 + 3) * (4 - 1))") == 15.0
        assert self.evaluator.evaluate("(10 - (2 + 3)) * 2") == 10.0

    def test_unary_minus(self) -> None:
        """Test unary minus."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-(2 + 1)") == -3.0
        assert self.evaluator.evaluate("-3 + 5") == 2.0
        assert self.evaluator.evaluate("2 * -3") == -6.0
        assert self.evaluator.evaluate("-(3 + 2) * 2") == -10.0

    def test_error_cases(self) -> None:
        """Test error cases: empty expression, mismatched parentheses, division by zero, invalid tokens."""
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("")
        with pytest.raises(ValueError, match="Empty expression"):
            self.evaluator.evaluate("   ")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("2 + 3)")
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("1 / 0")
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("2 & 3")
        with pytest.raises(ValueError, match="Invalid token"):
            self.evaluator.evaluate("2 + abc")
        with pytest.raises(ValueError, match="Unexpected token"):
            self.evaluator.evaluate("2 3")
```

**How it works:**

1. **Tokenizer** (`_tokenize`): Scans the expression string character by character, producing tokens (numbers, operators, parentheses). It validates that numbers are well-formed and raises errors for invalid characters.

2. **Recursive Descent Parser**: Uses four methods that mirror the grammar:
   - `_parse_expression`: Handles `+` and `-` (lowest precedence)
   - `_parse_term`: Handles `*` and `/` (higher precedence)
   - `_parse_factor`: Handles unary minus, parenthesized sub-expressions, and numbers

3. **Precedence**: The grammar naturally enforces precedence — `expression` calls `term`, which calls `factor`. This means `*` and `/` are evaluated before `+` and `-`.

4. **Unary minus**: Handled in `_parse_factor` by recursively calling itself after consuming the `-` token.

5. **Error handling**: Each method checks for invalid states and raises `ValueError` with descriptive messages.

**To run the tests:**
```bash
pytest test_expression_evaluator.py -v
```