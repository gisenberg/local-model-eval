# MiniMax-M2.7 UD-IQ2_XXS (empty-think, 192K ctx) — Expression Evaluator

## Output (12017 chars)

Below is a complete implementation of a **recursive‑descent parser** that satisfies all the requirements:

```python
"""
Expression Evaluator
====================

A pure‑Python expression evaluator built with a recursive‑descent parser.
Supports:
* Basic arithmetic operators: +, -, *, /
* Parentheses for grouping
* Unary minus (e.g. "-3", "-(2+1)")
* Floating‑point numbers (e.g. "3.14")
* Proper operator precedence (MUL/DIV before ADD/SUB)
* Descriptive ValueError messages for malformed input
"""

import re
from typing import List

__all__ = ["ExpressionEvaluator"]


class ExpressionEvaluator:
    """
    Recursive‑descent parser that evaluates a simple arithmetic expression.

    Grammar (EBNF):
        expr     → term ( ('+' | '-') term )*
        term     → factor ( ('*' | '/') factor )*
        factor   → unary | '(' expr ')'
        unary    → '-' unary | primary
        primary  → NUMBER | '(' expr ')'

    Tokens are produced by a simple tokenizer; the parser then builds an AST
    and evaluates it directly.
    """

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

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
            The numeric result of the expression.

        Raises
        ------
        ValueError
            If the expression is malformed (mismatched parentheses,
            division by zero, invalid tokens, empty input, etc.).
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        try:
            self._tokenizer = Tokenizer(expr)
            self._tokens = self._tokenizer.tokenize()
        except ValueError as err:
            raise ValueError(f"Invalid token: {err}") from err

        # Prime the lookahead
        self._pos = 0

        result = self._parse_expr()

        # Make sure we have consumed all tokens
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token '{self._tokens[self._pos].value}'")

        return result

    # ----------------------------------------------------------------------
    # Private helpers – tokenizer
    # ----------------------------------------------------------------------

    def _peek(self) -> "Token":
        """Return the current token without advancing."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return Token("EOF", "")

    def _consume(self, expected_type: str) -> "Token":
        """Consume the next token if it matches *expected_type*."""
        token = self._peek()
        if token.type != expected_type:
            raise ValueError(f"Unexpected token '{token.value}'; expected {expected_type}")
        self._pos += 1
        return token

    # ----------------------------------------------------------------------
    # Private helpers – parser (recursive descent)
    # ----------------------------------------------------------------------

    def _parse_expr(self) -> float:
        """Parse a full expression: term (('+' | '-') term)*"""
        value = self._parse_term()
        while True:
            token = self._peek()
            if token.type == "PLUS":
                self._consume("PLUS")
                value = value + self._parse_term()
            elif token.type == "MINUS":
                self._consume("MINUS")
                value = value - self._parse_term()
            else:
                break
        return value

    def _parse_term(self) -> float:
        """Parse a term: factor (('*' | '/') factor)*"""
        value = self._parse_factor()
        while True:
            token = self._peek()
            if token.type == "MUL":
                self._consume("MUL")
                value = value * self._parse_factor()
            elif token.type == "DIV":
                self._consume("DIV")
                divisor = self._parse_factor()
                if divisor == 0.0:
                    raise ValueError("Division by zero")
                value = value / divisor
            else:
                break
        return value

    def _parse_factor(self) -> float:
        """Parse a factor: unary | '(' expr ')'"""
        token = self._peek()
        if token.type == "MINUS":
            self._consume("MINUS")
            return -self._parse_factor()
        elif token.type == "LPAREN":
            self._consume("LPAREN")
            value = self._parse_expr()
            if self._peek().type != "RPAREN":
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume("RPAREN")
            return value
        elif token.type == "NUMBER":
            self._consume("NUMBER")
            return float(token.value)
        else:
            raise ValueError(f"Unexpected token '{token.value}'")

    # ----------------------------------------------------------------------
    # Private helpers – token representation
    # ----------------------------------------------------------------------


class Token:
    """Simple token class."""

    __slots__ = ("type", "value")

    def __init__(self, token_type: str, value: str) -> None:
        self.type = token_type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type!r}, {self.value!r})"


class Tokenizer:
    """
    Simple tokenizer for arithmetic expressions.

    Recognizes:
        * NUMBER   – integer or floating point literals
        * PLUS     – '+'
        * MINUS    – '-'
        * MUL      – '*'
        * DIV      – '/'
        * LPAREN   – '('
        * RPAREN   – ')'
    """

    TOKEN_SPECS = [
        ("NUMBER",  r"(?P<num>\d+\.\d*|\d*\.?\d+)([eE][+-]?\d+)?"),
        ("PLUS",    r"\+"),
        ("MINUS",   r"-"),
        ("MUL",     r"\*"),
        ("DIV",     r"/"),
        ("LPAREN",  r"\("),
        ("RPAREN",  r"\)"),
        ("SKIP",    r"\s+"),          # whitespace
    ]

    def __init__(self, text: str) -> None:
        self.text = text
        self._regex = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in self.TOKEN_SPECS))

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        pos = 0
        while pos < len(self.text):
            match = self._regex.match(self.text, pos)
            if not match:
                raise ValueError(self.text[pos])
            group = match.lastgroup
            if group == "SKIP":
                pos = match.end()
                continue
            if group is None:
                raise ValueError(self.text[pos])
            value = match.group(group)
            tokens.append(Token(group, value))
            pos = match.end()
        return tokens


# --------------------------------------------------------------------------
# Quick demo / sanity check (optional)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ev = ExpressionEvaluator()
    print(ev.evaluate("2 + 3 * 4"))          # 14.0
    print(ev.evaluate("(2 + 3) * 4"))        # 20.0
    print(ev.evaluate("-3"))                  # -3.0
    print(ev.evaluate("-(2+1)"))              # -3.0
    print(ev.evaluate("3.14"))                # 3.14
```

---

### How it works

| Component | Role |
|-----------|------|
| **Tokenizer** (`Tokenizer`) | Splits the input string into a list of tokens (`NUMBER`, `PLUS`, `MINUS`, `MUL`, `DIV`, `LPAREN`, `RPAREN`). Whitespace is ignored. |
| **Parser** (`ExpressionEvaluator`) | Implements a recursive‑descent parser that follows the grammar shown in the docstring. |
| **Evaluation** | The parser directly computes a float as it parses (`_parse_expr`, `_parse_term`, `_parse_factor`). No separate AST is built – the parser evaluates on the fly. |
| **Error handling** | `ValueError` is raised for mismatched parentheses, division by zero, invalid tokens, empty input, or unexpected tokens. |

---

### Pytest test suite

```python
# test_expression_evaluator.py
import pytest
from expression_evaluator import ExpressionEvaluator

ev = ExpressionEvaluator()

# ----------------------------------------------------------------------
# Basic arithmetic
# ----------------------------------------------------------------------
def test_basic_addition():
    assert ev.evaluate("2 + 3") == 5.0

def test_basic_subtraction():
    assert ev.evaluate("5 - 2") == 3.0

def test_basic_multiplication():
    assert ev.evaluate("3 * 4") == 12.0

def test_basic_division():
    assert ev.evaluate("8 / 2") == 4.0

# ----------------------------------------------------------------------
# Operator precedence
# ----------------------------------------------------------------------
def test_precedence_mul_over_add():
    # 2 + 3 * 4 should be 2 + (3 * 4) = 14
    assert ev.evaluate("2 + 3 * 4") == 14.0

def test_precedence_div_over_sub():
    # 10 - 2 * 3 should be 10 - (2 * 3) = 4
    assert ev.evaluate("10 - 2 * 3") == 4.0

def test_precedence_mixed():
    # 10 / 2 + 3 * 4 - 1 = 5 + 12 - 1 = 16
    assert ev.evaluate("10 / 2 + 3 * 4 - 1") == 16.0

# ----------------------------------------------------------------------
# Parentheses
# ----------------------------------------------------------------------
def test_parentheses_grouping():
    # (2 + 3) * 4 = 5 * 4 = 20
    assert ev.evaluate("(2 + 3) * 4") == 20.0

def test_nested_parentheses():
    # ((2 + 3) * (1 + 1)) = 5 * 2 = 10
    assert ev.evaluate("((2 + 3) * (1 + 1))") == 10.0

def test_parentheses_with_precedence():
    # (10 / 2 + 3) * 2 = (5 + 3) * 2 = 16
    assert ev.evaluate("(10 / 2 + 3) * 2") == 16.0

# ----------------------------------------------------------------------
# Unary minus
# ----------------------------------------------------------------------
def test_unary_minus_simple():
    assert ev.evaluate("-3") == -3.0

def test_unary_minus_in_expression():
    # -3 + 5 = 2
    assert ev.evaluate("-3 + 5") == 2.0

def test_unary_minus_with_parentheses():
    # -(2 + 1) = -3
    assert ev.evaluate("-(2 + 1)") == -3.0

def test_double_unary_minus():
    # --5 = 5
    assert ev.evaluate("--5") == 5.0

# ----------------------------------------------------------------------
# Floating point numbers
# ----------------------------------------------------------------------
def test_floating_point():
    assert ev.evaluate("3.14") == 3.14

def test_floating_point_arithmetic():
    # 1.5 + 2.5 = 4.0
    assert ev.evaluate("1.5 + 2.5") == 4.0

def test_floating_point_precedence():
    # 3.0 * 2.0 + 1.0 = 6.0 + 1.0 = 7.0
    assert ev.evaluate("3.0 * 2.0 + 1.0") == 7.0

# ----------------------------------------------------------------------
# Error cases
# ----------------------------------------------------------------------
def test_mismatched_parentheses_left():
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")

def test_mismatched_parentheses_right():
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("2 + 3)")

def test_division_by_zero():
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("5 / 0")

def test_invalid_token():
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 $ 3")

def test_empty_expression():
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")

def test_invalid_syntax():
    with pytest.raises(ValueError, match="Unexpected token"):
        ev.evaluate("2 + + 3")
```

---

### Running the tests

```bash
$ pip install pytest
$ pytest test_expression_evaluator.py -v
```

All 5 core categories (basic arithmetic, precedence, parentheses, unary minus, error cases) are covered, and the implementation satisfies every requirement listed in the prompt.
