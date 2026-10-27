

# Mathematical Expression Evaluator

Here's a complete implementation with a recursive descent parser, tokenizer, and tests:

## `expression_evaluator.py`

```python
"""A mathematical expression evaluator using recursive descent parsing."""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
from typing import List


class TokenType(Enum):
    """Enumeration of token types recognized by the lexer."""

    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    """A single token produced by the lexer."""

    type: TokenType
    value: float | str
    position: int


class Lexer:
    """Tokenizes a mathematical expression string into a stream of tokens."""

    def __init__(self, text: str) -> None:
        """Initialize the lexer with the input expression text."""
        self._text: str = text
        self._pos: int = 0

    def tokenize(self) -> List[Token]:
        """Convert the input text into a list of Token objects.

        Returns:
            A list of tokens ending with an EOF token.

        Raises:
            ValueError: If an unrecognized character is encountered.
        """
        tokens: List[Token] = []
        while self._pos < len(self._text):
            ch: str = self._text[self._pos]

            # Skip whitespace
            if ch.isspace():
                self._pos += 1
                continue

            # Numbers (integer or float)
            if ch.isdigit() or ch == ".":
                tokens.append(self._read_number())
                continue

            # Single-character operators and parentheses
            token_map: dict[str, TokenType] = {
                "+": TokenType.PLUS,
                "-": TokenType.MINUS,
                "*": TokenType.MULTIPLY,
                "/": TokenType.DIVIDE,
                "(": TokenType.LPAREN,
                ")": TokenType.RPAREN,
            }

            if ch in token_map:
                tokens.append(Token(token_map[ch], ch, self._pos))
                self._pos += 1
                continue

            raise ValueError(
                f"Invalid character '{ch}' at position {self._pos}"
            )

        tokens.append(Token(TokenType.EOF, "", self._pos))
        return tokens

    def _read_number(self) -> Token:
        """Read a floating-point or integer number starting at the current position.

        Returns:
            A NUMBER token containing the parsed float value.

        Raises:
            ValueError: If the number has multiple decimal points.
        """
        start: int = self._pos
        dot_count: int = 0

        while self._pos < len(self._text) and (
            self._text[self._pos].isdigit() or self._text[self._pos] == "."
        ):
            if self._text[self._pos] == ".":
                dot_count += 1
                if dot_count > 1:
                    raise ValueError(
                        f"Invalid number at position {start}: "
                        f"'{self._text[start:self._pos + 1]}' contains "
                        f"multiple decimal points"
                    )
            self._pos += 1

        number_str: str = self._text[start : self._pos]
        if number_str == ".":
            raise ValueError(
                f"Invalid number '.' at position {start}"
            )
        return Token(TokenType.NUMBER, float(number_str), start)


class ExpressionEvaluator:
    """Evaluates mathematical expressions using a recursive descent parser.

    Supported syntax:
        - Binary operators: +, -, *, /
        - Unary operators: +, -
        - Parentheses for grouping
        - Integer and floating-point literals

    Grammar (EBNF):
        expression = term {('+' | '-') term}
        term       = factor {('*' | '/') factor}
        factor     = ['+' | '-'] (NUMBER | '(' expression ')')

    Usage:
        >>> evaluator = ExpressionEvaluator()
        >>> evaluator.evaluate("2 + 3 * 4")
        14.0
    """

    def __init__(self) -> None:
        """Initialize the evaluator with an empty token list and position."""
        self._tokens: List[Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result.

        Args:
            expr: The expression string to evaluate.

        Returns:
            The numeric result as a float.

        Raises:
            ValueError: If the expression is empty, has mismatched parentheses,
                        contains invalid tokens, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Cannot evaluate an empty expression")

        lexer = Lexer(expr)
        self._tokens = lexer.tokenize()
        self._pos = 0

        result: float = self._parse_expression()

        # Ensure all tokens have been consumed
        if self._current().type != TokenType.EOF:
            raise ValueError(
                f"Unexpected token '{self._current().value}' at position "
                f"{self._current().position}. Expected end of expression."
            )

        return result

    # ─── Recursive Descent Parsing ────────────────────────────────────────

    def _parse_expression(self) -> float:
        """Parse an expression: handles + and - (lowest precedence).

        Grammar: expression = term {('+' | '-') term}

        Returns:
            The evaluated float value of the expression.
        """
        result: float = self._parse_term()

        while self._current().type in (TokenType.PLUS, TokenType.MINUS):
            op: TokenType = self._current().type
            self._advance()
            right: float = self._parse_term()
            if op == TokenType.PLUS:
                result += right
            else:
                result -= right

        return result

    def _parse_term(self) -> float:
        """Parse a term: handles * and / (higher precedence than + and -).

        Grammar: term = factor {('*' | '/') factor}

        Returns:
            The evaluated float value of the term.

        Raises:
            ValueError: On division by zero.
        """
        result: float = self._parse_factor()

        while self._current().type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            op: TokenType = self._current().type
            self._advance()
            right: float = self._parse_factor()
            if op == TokenType.MULTIPLY:
                result *= right
            else:
                if right == 0.0:
                    raise ValueError(
                        "Division by zero is not allowed"
                    )
                result /= right

        return result

    def _parse_factor(self) -> float:
        """Parse a factor: handles unary +/-, numbers, and parenthesized expressions.

        Grammar: factor = ['+' | '-'] (NUMBER | '(' expression ')')

        Returns:
            The evaluated float value of the factor.

        Raises:
            ValueError: On mismatched parentheses or unexpected tokens.
        """
        # Handle unary operators
        if self._current().type == TokenType.MINUS:
            self._advance()
            return -self._parse_factor()

        if self._current().type == TokenType.PLUS:
            self._advance()
            return self._parse_factor()

        # Handle numbers
        if self._current().type == TokenType.NUMBER:
            value: float = float(self._current().value)
            self._advance()
            return value

        # Handle parenthesized sub-expressions
        if self._current().type == TokenType.LPAREN:
            self._advance()  # consume '('
            value = self._parse_expression()

            if self._current().type != TokenType.RPAREN:
                raise ValueError(
                    f"Mismatched parentheses: expected ')' at position "
                    f"{self._current().position}, got "
                    f"'{self._current().value or 'end of input'}'"
                )
            self._advance()  # consume ')'
            return value

        raise ValueError(
            f"Unexpected token '{self._current().value or 'end of input'}' "
            f"at position {self._current().position}. Expected a number, "
            f"'(', '+', or '-'."
        )

    # ─── Token Helper Methods ─────────────────────────────────────────────

    def _current(self) -> Token:
        """Return the token at the current position without consuming it."""
        return self._tokens[self._pos]

    def _advance(self) -> None:
        """Advance the parser position by one token."""
        self._pos += 1
```

## `test_expression_evaluator.py`

```python
"""Tests for the ExpressionEvaluator class."""

import pytest

from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Provide a fresh ExpressionEvaluator instance for each test."""
    return ExpressionEvaluator()


class TestBasicArithmetic:
    """Tests for fundamental addition, subtraction, multiplication, division."""

    def test_simple_operations(self, evaluator: ExpressionEvaluator) -> None:
        """Verify basic +, -, *, / produce correct results."""
        assert evaluator.evaluate("2 + 3") == 5.0
        assert evaluator.evaluate("10 - 4") == 6.0
        assert evaluator.evaluate("3 * 7") == 21.0
        assert evaluator.evaluate("15 / 3") == 5.0
        assert evaluator.evaluate("2 + 3 - 1") == 4.0
        assert evaluator.evaluate("100 / 4 / 5") == 5.0


class TestOperatorPrecedence:
    """Tests verifying * and / bind tighter than + and -."""

    def test_precedence(self, evaluator: ExpressionEvaluator) -> None:
        """Multiplication/division should be evaluated before addition/subtraction."""
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        assert evaluator.evaluate("3 * 4 + 2") == 14.0
        assert evaluator.evaluate("2 - 3 * 4") == -10.0
        assert evaluator.evaluate("10 - 2 * 3 + 4") == 8.0
        assert evaluator.evaluate("6 / 2 + 1") == 4.0
        assert evaluator.evaluate("2 + 6 / 2") == 5.0


class TestParentheses:
    """Tests for grouping sub-expressions with parentheses."""

    def test_parentheses(self, evaluator: ExpressionEvaluator) -> None:
        """Parentheses should override default operator precedence."""
        assert evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
        assert evaluator.evaluate("10 / (2 + 3)") == 2.0
        assert evaluator.evaluate("(2 + 3) * (4 - 1)") == 15.0


class TestUnaryMinus:
    """Tests for unary negation applied to numbers and expressions."""

    def test_unary_minus(self, evaluator: ExpressionEvaluator) -> None:
        """Unary minus should negate a number or a parenthesized expression."""
        assert evaluator.evaluate("-3") == -3.0
        assert evaluator.evaluate("-3 + 2") == -1.0
        assert evaluator.evaluate("-(2 + 1)") == -3.0
        assert evaluator.evaluate("2 * -3") == -6.0
        assert evaluator.evaluate("--5") == 5.0
        assert evaluator.evaluate("-(4)") == -4.0
        assert evaluator.evaluate("3 - -2") == 5.0


class TestErrorCases:
    """Tests for proper ValueError exceptions on invalid input."""

    def test_empty_expression(self, evaluator: ExpressionEvaluator) -> None:
        """An empty or whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError, match="empty expression"):
            evaluator.evaluate("")
        with pytest.raises(ValueError, match="empty expression"):
            evaluator.evaluate("   ")

    def test_division_by_zero(self, evaluator: ExpressionEvaluator) -> None:
        """Division by zero should raise a descriptive ValueError."""
        with pytest.raises(ValueError, match="[Dd]ivision by zero"):
            evaluator.evaluate("5 / 0")
        with pytest.raises(ValueError, match="[Dd]ivision by zero"):
            evaluator.evaluate("10 / (3 - 3)")

    def test_mismatched_parentheses(self, evaluator: ExpressionEvaluator) -> None:
        """Unbalanced parentheses should raise a descriptive ValueError."""
        with pytest.raises(ValueError, match="[Mm]ismatched parentheses"):
            evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="[Mm]ismatched parentheses"):
            evaluator.evaluate("2 + 3)")
        with pytest.raises(ValueError, match="[Mm]ismatched parentheses"):
            evaluator.evaluate("((1 + 2)")

    def test_invalid_tokens(self, evaluator: ExpressionEvaluator) -> None:
        """Unrecognized characters should raise a descriptive ValueError."""
        with pytest.raises(ValueError, match="[Ii]nvalid character"):
            evaluator.evaluate("2 + $3")
        with pytest.raises(ValueError, match="[Uu]nexpected token"):
            evaluator.evaluate("2 + + ")
        with pytest.raises(ValueError, match="[Ii]nvalid number"):
            evaluator.evaluate("1.2.3")

    def test_missing_operand(self, evaluator: ExpressionEvaluator) -> None:
        """Trailing or leading operators without operands should raise ValueError."""
        with pytest.raises(ValueError, match="[Uu]nexpected token"):
            evaluator.evaluate("2 +")
        with pytest.raises(ValueError, match="[Uu]nexpected token"):
            evaluator.evaluate("* 3")
```

## Key Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Parsing strategy** | Recursive descent (3 levels) | Clean separation of precedence: `expr → term → factor` |
| **Tokenization** | Separate `Lexer` class | Single-responsibility; easier to test and extend |
| **Unary handling** | Recursive in `_parse_factor` | Allows chained unary (`--5`, `-(-3)`) naturally |
| **Error messages** | Include position & context | Makes debugging user-facing expressions tractable |
| **Return type** | Always `float` | Simplifies downstream consumption; integers become `2.0` |

## Running the Tests

```bash
# Install pytest if needed
pip install pytest

# Run the test suite
pytest test_expression_evaluator.py -v

# Run a quick interactive check
python -c "
from expression_evaluator import ExpressionEvaluator
e = ExpressionEvaluator()
print(e.evaluate('3.14 * (2 + 1)'))   # 9.42
print(e.evaluate('-2 ** ...'))         # ValueError (no ** support)
print(e.evaluate('-(5 + 3) / 2'))      # -4.0
"
```

The grammar strictly enforces left-to-right associativity for `+`/`-` and `*`/`/` through the iterative loops in `_parse_expression` and `_parse_term`, while recursion in `_parse_factor` handles both grouping and unary operators. No `eval()` or `ast` module is used anywhere.