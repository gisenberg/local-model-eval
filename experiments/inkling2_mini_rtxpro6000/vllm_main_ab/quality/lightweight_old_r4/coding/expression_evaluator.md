# Mathematical Expression Evaluator

Here's a complete solution using a recursive descent parser (no `eval()`/`ast`), split into a module and a pytest file.

## `expression_evaluator.py`

```python
"""A safe recursive-descent mathematical expression evaluator.

Supports +, -, *, / with standard precedence, parentheses, unary minus,
and floating-point literals. Never uses eval() or ast.literal_eval().
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class _Token:
    """A single lexical token produced by the tokenizer."""

    kind: str        # 'number' | 'operator' | 'lparen' | 'rparen'
    text: str        # raw source text
    value: float     # numeric value for 'number' tokens (0.0 otherwise)
    position: int    # index of the token's first character in the source


class ExpressionEvaluator:
    """Evaluate arithmetic expressions via recursive descent.

    Example:
        >>> ExpressionEvaluator().evaluate("2 + 3 * 4")
        14.0
    """

    def __init__(self) -> None:
        """Initialize the evaluator with an empty token buffer."""
        self._tokens: List[_Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Parse and evaluate an arithmetic expression.

        Args:
            expr: The expression string, e.g. "3.14 * (2 + 1)".

        Returns:
            The result of the expression.

        Raises:
            ValueError: If expr is empty, contains invalid characters/tokens,
                has mismatched parentheses, or divides by zero.
        """
        if not isinstance(expr, str):
            raise ValueError(f"Expression must be a string, got {type(expr).__name__}")

        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Empty expression: expected a number, '(', or an operator")

        self._tokens = tokens
        self._pos = 0
        try:
            result = self._parse_expression()
            self._expect_end()
            return result
        finally:
            self._tokens = []
            self._pos = 0

    # ------------------------- Tokenizer -------------------------

    def _tokenize(self, expr: str) -> List[_Token]:
        """Split the source into tokens.

        Raises:
            ValueError: On an invalid character or an invalid number literal.
        """
        tokens: List[_Token] = []
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
            elif ch.isdigit() or ch == ".":
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == "."):
                    i += 1
                text = expr[start:i]
                try:
                    value = float(text)
                except ValueError:
                    raise ValueError(f"Invalid number '{text}' at position {start}") from None
                tokens.append(_Token("number", text, value, start))
            elif ch in "+-*/":
                tokens.append(_Token("operator", ch, 0.0, i))
                i += 1
            elif ch == "(":
                tokens.append(_Token("lparen", ch, 0.0, i))
                i += 1
            elif ch == ")":
                tokens.append(_Token("rparen", ch, 0.0, i))
                i += 1
            else:
                raise ValueError(f"Invalid character '{ch}' at position {i}")
        return tokens

    # --------------------- Recursive descent parser ---------------------
    #
    # Grammar (precedence low -> high):
    #   expression := term (('+' | '-') term)*
    #   term       := factor (('*' | '/') factor)*
    #   factor     := ('-' | '+') factor | primary
    #   primary    := number | '(' expression ')'

    def _peek(self) -> Optional[_Token]:
        """Return the current token without consuming it, or None at the end."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> _Token:
        """Consume and return the current token."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _at_operator(self, ops: Tuple[str, ...]) -> bool:
        """Return True if the current token is one of the given operator texts."""
        token = self._peek()
        return token is not None and token.kind == "operator" and token.text in ops

    def _expect_end(self) -> None:
        """Raise ValueError if any tokens remain after the top-level parse."""
        token = self._peek()
        if token is not None:
            if token.kind == "rparen":
                raise ValueError(f"Mismatched parentheses: unexpected ')' at position {token.position}")
            raise ValueError(f"Invalid token '{token.text}' at position {token.position}: unexpected trailing input")

    def _parse_expression(self) -> float:
        """Parse an expression: term (('+' | '-') term)*."""
        left = self._parse_term()
        while self._at_operator(("+", "-")):
            op = self._advance().text
            right = self._parse_term()
            left = left + right if op == "+" else left - right
        return left

    def _parse_term(self) -> float:
        """Parse a term: factor (('*' | '/') factor)*."""
        left = self._parse_factor()
        while self._at_operator(("*", "/")):
            op_token = self._advance()
            right = self._parse_factor()
            if op_token.text == "*":
                left = left * right
            else:
                if right == 0.0:
                    raise ValueError(f"Division by zero at position {op_token.position}")
                left = left / right
        return left

    def _parse_factor(self) -> float:
        """Parse a factor: ('-' | '+') factor | primary."""
        if self._at_operator(("+", "-")):
            op = self._advance().text
            value = self._parse_factor()
            return -value if op == "-" else value
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a primary: number | '(' expression ')'."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression: expected a number or '('")
        if token.kind == "number":
            self._advance()
            return token.value
        if token.kind == "lparen":
            self._advance()
            value = self._parse_expression()
            closing = self._peek()
            if closing is None:
                raise ValueError("Mismatched parentheses: expected ')' at end of expression")
            if closing.kind != "rparen":
                raise ValueError(
                    f"Mismatched parentheses: expected ')' but found '{closing.text}' "
                    f"at position {closing.position}"
                )
            self._advance()
            return value
        if token.kind == "rparen":
            raise ValueError(f"Mismatched parentheses: unexpected ')' at position {token.position}")
        raise ValueError(f"Unexpected token '{token.text}' at position {token.position}")


if __name__ == "__main__":
    evaluator = ExpressionEvaluator()
    for expr in ["2 + 3 * 4", "(2 + 3) * 4", "-(2 + 1)", "3.14 * 2", "10 / 4"]:
        print(f"{expr}  =  {evaluator.evaluate(expr)}")
```

## `test_expression_evaluator.py`

```python
"""Pytest suite for ExpressionEvaluator (5 tests)."""

import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator():
    """Return a fresh evaluator for each test."""
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator):
    """Basic +, -, *, / operations and float literals."""
    assert evaluator.evaluate("2 + 3") == 5
    assert evaluator.evaluate("10 - 4") == 6
    assert evaluator.evaluate("6 * 7") == 42
    assert evaluator.evaluate("8 / 2") == 4
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)


def test_operator_precedence(evaluator):
    """Multiplication/division bind tighter than addition/subtraction."""
    assert evaluator.evaluate("2 + 3 * 4") == 14
    assert evaluator.evaluate("10 - 2 * 3") == 4
    assert evaluator.evaluate("8 / 2 / 4") == 1.0   # left associativity
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26


def test_parentheses(evaluator):
    """Parenthesized expressions override precedence."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20
    assert evaluator.evaluate("2 * (3 + 4)") == 14
    assert evaluator.evaluate("((1 + 2) * 3)") == 9
    assert evaluator.evaluate("3 + (4 * 2) - 5") == 10


def test_unary_minus(evaluator):
    """Unary minus applies to numbers, groups, and nested factors."""
    assert evaluator.evaluate("-3") == -3
    assert evaluator.evaluate("-(2 + 1)") == -3
    assert evaluator.evaluate("--5") == 5
    assert evaluator.evaluate("3 * -2") == -6
    assert evaluator.evaluate("2 + -1.5") == 0.5


def test_error_cases(evaluator):
    """Empty expressions, mismatched parentheses, division by zero, invalid tokens."""
    # Empty expressions
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("1 + 2)")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("((1 + 2)")
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("(2 + 3) / (4 - 4)")
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("3 @ 4")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 3")
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("1.2.3")
```

## Running the tests

```bash
python -m pytest test_expression_evaluator.py -v
```

## Design notes

- **Grammar** (standard precedence, left-associative): `expression → term → factor → primary`. `factor` handles unary `-`/`+`, and each level is a direct recursive-descent method that evaluates as it parses.
- **Error handling**: every failure path raises `ValueError` with a message that includes the offending text and source position — e.g. `Division by zero at position 2` or `Invalid number '1.2.3' at position 0`.
- **Floating point**: literals are parsed with `float()`, so `3.14`, `.5`, `3.`, and integer literals all work and return `float`.
- **Scope limits**: scientific notation (`1e5`) and named functions are intentionally not supported — adding them means extending the tokenizer/grammar, which is a straightforward next step.