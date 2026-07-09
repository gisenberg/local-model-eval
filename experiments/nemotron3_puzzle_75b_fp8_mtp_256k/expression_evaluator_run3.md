We need to produce code for ExpressionEvaluator class with recursive descent parser, supporting operations, parentheses, unary minus, floats, raising ValueError for errors. Also include type hints, docstrings. Then provide 5 pytest tests. Provide final answer with code and tests. Ensure we adhere to requirements.

Will implement tokenizer scanning numbers (including decimal point). Also handle minus as unary or binary based on context.

Recursive descent grammar: expression = term ((+|-) term)* ; term = factor ((*|/) factor)* ; factor = (+|-) factor | number | '(' expression ')'.

Number: sequence of digits possibly with one '.'.

We need to handle errors: mismatched parentheses (if expecting '(' not found, or leftover after parse), division by zero, invalid tokens (any char not part of number or operator or parentheses or whitespace), empty expression (after stripping whitespace yields empty). We'll raise ValueError with appropriate message.

Implementation details:

class ExpressionEvaluator:
    def __init__(self):
        self.expr = ''
        self.pos = 0
        self.tokens = [] maybe not needed; we can parse on the fly with index into string.

Better to tokenize first to simplify, but tokenizing can be done on the fly while parsing.

I'll implement a simple tokenizer that yields tokens: numbers (float), '+', '-', '*', '/', '(', ')'. Whitespace ignored. Raise ValueError for invalid char.

But need to differentiate unary minus: during parsing factor we can check if token is '-' and next token is something that can start factor.

Implement parse_expression returning float.

Pseudo:

class ExpressionEvaluator:
    def __init__(self):
        self.tokens = []
        self.pos = 0

    def tokenize(self, s: str):
        i=0; n=len(s)
        while i < n:
            ch=s[i]
            if ch.isspace():
                i+=1
                continue
            if ch in '+-*/()':
                self.tokens.append(ch)
                i+=1
            elif ch.isdigit() or ch=='.':
                # parse number
                start=i
                dot_seen=False
                while i < n and (s[i].isdigit() or s[i]=='.'):
                    if s[i]=='.':
                        if dot_seen:
                            raise ValueError("Invalid number format")
                        dot_seen=True
                    i+=1
                num_str=s[start:i]
                # Ensure not just '.' or empty
                if num_str=='.' or num_str.count('.')>1:
                    raise ValueError("Invalid number format")
                try:
                    val=float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")
                self.tokens.append(val)
            else:
                raise ValueError(f"Invalid character: {ch}")

    def evaluate(self, expr: str) -> float:
        if not expr or expr.strip()=='':
            raise ValueError("Empty expression")
        self.tokenize(expr)
        self.pos=0
        if not self.tokens:
            raise ValueError("Empty expression")
        result=self._parse_expression()
        if self.pos != len(self.tokens):
            raise ValueError("Mismatched parentheses or extra tokens")
        return result

    def _parse_expression(self) -> float:
        left=self._parse_term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op=self.tokens[self.pos]
            self.pos+=1
            right=self._parse_term()
            if op=='+':
                left+=right
            else:
                left-=right
        return left

    def _parse_term(self) -> float:
        left=self._parse_factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op=self.tokens[self.pos]
            self.pos+=1
            right=self._parse_factor()
            if op=='*':
                left*=right
            else:
                if right==0:
                    raise ValueError("Division by zero")
                left/=right
        return left

    def _parse_factor(self) -> float:
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        token=self.tokens[self.pos]
        # unary plus/minus
        if token == '+':
            self.pos+=1
            return self._parse_factor()
        if token == '-':
            self.pos+=1
            return -self._parse_factor()
        if token == '(':
            self.pos+=1
            val=self._parse_expression()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos+=1
            return val
        if isinstance(token, (int, float)):
            self.pos+=1
            return float(token)
        raise ValueError(f"Invalid token: {token}")

Add docstrings etc.

Now tests: We'll write 5 pytest functions.

Test cases:
1. Basic precedence: "2+3*4" => 14
2. Parentheses: "(2+3)*4" => 20
3. Unary minus: "-3+2*(-1)" => -3 + -2 = -5? Actually 2*(-1) = -2, -3 + -2 = -5. So -5.
4. Floating point: "3.5 + 2.1 * 2" => 3.5 + 4.2 = 7.7
5. Error handling: division by zero "5/0" => ValueError
6. Mismatched parentheses: "(1+2" => ValueError
7. Invalid token: "2 ++ 3" maybe raises due to invalid token? Actually tokenization yields '2', '+', '+', '3' -> parse factor sees '+' unary then factor expects something; after consuming unary '+', parse factor expects number or '(' etc. Next token is '+', which is allowed as unary again? Actually after consuming first '+', we are at second '+', which is unary allowed? The grammar would allow ++3? That would parse as unary + then factor: unary + again then number 3 => 3, then expression sees left 2 + (result of ++3) which is 2+3 =5. That might be okay; but requirement says raise ValueError for invalid tokens, not for ++? Might be okay. Better test invalid token like "2 @ 3". So test that.

We'll provide 5 tests covering various aspects.

Implementation code will be in a file, with class and tests.

Make sure to include import typing for type hints.

Let's craft final answer.





**Solution Overview**

We implement a tiny recursive‑descent parser that follows the grammar

```
expression ::= term (('+' | '-') term)*
term       ::= factor (('*' | '/') factor)*
factor     ::= ('+' | '-') factor | NUMBER | '(' expression ')'
NUMBER     ::= digits ('.' digits)?
```

* Whitespace is ignored.
* Numbers are converted to `float`; a malformed literal (e.g. `"..5"` or `"."`) raises `ValueError`.
* Unary plus/minus are handled in `factor`.
* Parentheses are checked for matching; a missing `)` or extra tokens after parsing raise `ValueError`.
* Division by zero is detected in `term`.
* Any character that is not part of a number, an operator, a parenthesis or whitespace causes an `Invalid token` error.

The public API is the class `ExpressionEvaluator` with a single method
`evaluate(expr: str) -> float`.

---

### Code

```python
"""ExpressionEvaluator – a tiny recursive‑descent math expression evaluator.

Supports +, -, *, /, parentheses, unary minus and floating point numbers.
Raises ValueError for mismatched parentheses, division by zero,
invalid tokens or empty input.
"""

from __future__ import annotations

from typing import List, Union


class ExpressionEvaluator:
    """Evaluates arithmetic expressions using a recursive‑descent parser."""

    def __init__(self) -> None:
        self._tokens: List[Union[float, str]] = []
        self._pos: int = 0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def evaluate(self, expr: str) -> float:
        """Parse and evaluate *expr*.

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
            has mismatched parentheses or attempts division by zero.
        """
        if not expr or expr.strip() == "":
            raise ValueError("Empty expression")

        self._tokenize(expr)
        self._pos = 0
        if not self._tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()
        if self._pos != len(self._tokens):
            # Leftover tokens -> mismatched parentheses or stray symbols
            raise ValueError("Mismatched parentheses or extra tokens")
        return result

    # --------------------------------------------------------------------- #
    # Tokeniser
    # --------------------------------------------------------------------- #
    def _tokenize(self, s: str) -> None:
        """Convert *s* into a list of tokens (numbers, operators, parentheses)."""
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch.isspace():
                i += 1
                continue
            if ch in "+-*/()":
                self._tokens.append(ch)
                i += 1
                continue
            if ch.isdigit() or ch == ".":
                start = i
                dot_seen = False
                while i < n and (s[i].isdigit() or s[i] == "."):
                    if s[i] == ".":
                        if dot_seen:
                            raise ValueError("Invalid number format")
                        dot_seen = True
                    i += 1
                num_str = s[start:i]
                # Reject "." or multiple dots
                if num_str == "." or num_str.count(".") > 1:
                    raise ValueError("Invalid number format")
                try:
                    value = float(num_str)
                except ValueError as exc:
                    raise ValueError(f"Invalid number: {num_str}") from exc
                self._tokens.append(value)
                continue
            raise ValueError(f"Invalid character: {ch!r}")

    # --------------------------------------------------------------------- #
    # Recursive‑descent parsing
    # --------------------------------------------------------------------- #
    def _parse_expression(self) -> float:
        """expression ::= term (('+' | '-') term)*"""
        left = self._parse_term()
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ("+", "-"):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_term()
            left = left + right if op == "+" else left - right
        return left

    def _parse_term(self) -> float:
        """term ::= factor (('*' | '/') factor)*"""
        left = self._parse_factor()
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ("*", "/"):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_factor()
            if op == "*":
                left = left * right
            else:  # division
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
        return left

    def _parse_factor(self) -> float:
        """factor ::= ('+' | '-') factor | NUMBER | '(' expression ')'"""
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression")

        token = self._tokens[self._pos]

        # Unary plus
        if token == "+":
            self._pos += 1
            return self._parse_factor()

        # Unary minus
        if token == "-":
            self._pos += 1
            return -self._parse_factor()

        # Parenthesised sub‑expression
        if token == "(":
            self._pos += 1
            val = self._parse_expression()
            if self._pos >= len(self._tokens) or self._tokens[self._pos] != ")":
                raise ValueError("Mismatched parentheses")
            self._pos += 1  # consume ')'
            return val

        # Number
        if isinstance(token, (int, float)):
            self._pos += 1
            return float(token)

        raise ValueError(f"Invalid token: {token!r}")
```

---

### Pytest test suite

```python
import pytest
from expression_evaluator import ExpressionEvaluator  # assuming the class is in this file


@pytest.fixture
def evaluator():
    return ExpressionEvaluator()


def test_basic_precedence(evaluator):
    """2 + 3 * 4 = 14 (multiplication before addition)."""
    assert evaluator.evaluate("2 + 3 * 4") == pytest.approx(14.0)


def test_parentheses(evaluator):
    """(2 + 3) * 4 = 20."""
    assert evaluator.evaluate("(2 + 3) * 4") == pytest.approx(20.0)


def test_unary_minus(evaluator):
    """-3 + 2 * (-1) = -5."""
    assert evaluator.evaluate("-3 + 2 * (-1)") == pytest.approx(-5.0)


def test_floating_point(evaluator):
    """3.5 + 2.1 * 2 = 7.7."""
    assert evaluator.evaluate("3.5 + 2.1 * 2") == pytest.approx(7.7)


def test_division_by_zero(evaluator):
    """Division by zero must raise ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")


def test_mismatched_parentheses(evaluator):
    """Missing closing parenthesis -> ValueError."""
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")


def test_invalid_token(evaluator):
    """Any unsupported character triggers ValueError."""
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 @ 3")


def test_empty_expression(evaluator):
    """Empty or whitespace‑only strings raise ValueError."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
```

*The test file contains **more than five** tests to demonstrate thorough coverage; any five of them satisfy the requirement.*

---

**How it works**

1. **Tokenisation** turns the input string into a list of numbers (`float`) and single‑character tokens (`+ - * / ( )`).
   Invalid characters or malformed numbers raise `ValueError` immediately.

2. **Recursive‑descent parsing** follows the grammar above, consuming tokens with an index (`self._pos`).
   Each parsing method returns a `float` and advances the index as it consumes tokens.

3. **Error handling** is performed at the points where the grammar expects something specific:
   * Missing `)` → “Mismatched parentheses”.
   * Extra tokens after a complete parse → “Mismatched parentheses or extra tokens”.
   * Division by zero → explicit check in `_parse_term`.
   * Empty input or only whitespace → caught in `evaluate`.

The implementation uses only primitive Python constructs, no `eval` or `ast`, and is fully type‑hinted and documented. Enjoy!