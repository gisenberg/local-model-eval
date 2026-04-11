# Nemotron-3-Super-120B-A12B Q4_K_M (bartowski) — KV f16 (baseline) — Expression Evaluator

## Thinking (6553 chars)

We need to produce code for ExpressionEvaluator class with recursive descent parser, supporting + - * /, parentheses, unary minus, floats. Provide type hints, docstrings. Also write 5 pytest tests covering basic arithmetic, precedence, parentheses, unary minus, error cases.

We need to output only the code? Probably include both class and test file. Provide as a single code block? Could separate sections. Provide class definition and test functions.

Implement parser: tokenization: numbers (including floats), operators + - * /, parentheses, maybe whitespace ignored.

Recursive descent grammar:

expr   ::= term ((+|-) term)*
term   ::= factor ((*|/) factor)*
factor ::= (+|-) factor | number | '(' expr ')'

But need to handle unary minus: factor can be ('+'|'-') factor.

Number: float literal.

Implement token generator.

Raise ValueError for mismatched parentheses, division by zero, invalid token, empty expression.

Implementation details:

class ExpressionEvaluator:
    def __init__(self):
        self.tokens = []
        self.pos = 0

    def evaluate(self, expr: str) -> float:
        self.tokens = self._tokenize(expr)
        if not self.tokens:
            raise ValueError("Empty expression")
        self.pos = 0
        result = self._parse_expr()
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
        return result

Tokenization: iterate over string, skip whitespace. Recognize numbers: digits and optional dot. Could also support scientific? Not required.

If encountering invalid char, raise ValueError.

Parentheses: '(' and ')'.

Operators: + - * /.

Unary plus/minus handled in factor.

Implement parse methods:

def _parse_expr(self) -> float:
    result = self._parse_term()
    while self._peek() in ('+', '-'):
        op = self._consume()
        right = self._parse_term()
        if op == '+':
            result += right
        else:
            result -= right
    return result

def _parse_term(self) -> float:
    result = self._parse_factor()
    while self._peek() in ('*', '/'):
        op = self._consume()
        right = self._parse_factor()
        if op == '*':
            result *= right
        else:
            if right == 0:
                raise ValueError("Division by zero")
            result /= right
    return result

def _parse_factor(self) -> float:
    token = self._peek()
    if token == '+':
        self._consume()
        return self._parse_factor()
    if token == '-':
        self._consume()
        return -self._parse_factor()
    if token == '(':
        self._consume()
        val = self._parse_expr()
        if self._peek() != ')':
            raise ValueError("Mismatched parentheses")
        self._consume()  # consume ')'
        return val
    # number
    if token is None:
        raise ValueError("Unexpected end of expression")
    # token should be a number string
    try:
        num = float(token)
    except ValueError:
        raise ValueError(f"Invalid token: {token}")
    self._consume()
    return num

Helper methods:

def _peek(self):
    if self.pos < len(self.tokens):
        return self.tokens[self.pos]
    return None

def _consume(self):
    tok = self._peek()
    if tok is None:
        raise ValueError("Unexpected end of expression")
    self.pos += 1
    return tok

Tokenization: produce list of strings: numbers as strings, operators, parentheses.

Implementation:

def _tokenize(self, expr: str) -> List[str]:
    tokens = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in '+-*/()':
            tokens.append(ch)
            i += 1
        elif ch.isdigit() or ch == '.':
            # start number
            start = i
            dot_seen = False
            while i < n and (expr[i].isdigit() or expr[i] == '.'):
                if expr[i] == '.':
                    if dot_seen:
                        raise ValueError("Invalid number format")
                    dot_seen = True
                i += 1
            tokens.append(expr[start:i])
        else:
            raise ValueError(f"Invalid character: {ch}")
    return tokens

Edge Cases: expression like ".5" should be allowed? Our number parsing allows leading dot; we accept '.' as start, but need at least one digit somewhere. Could just rely on float conversion; ".5" works. However ".." would be caught by dot_seen check.

Empty expression: after tokenization, if tokens empty -> raise.

Now tests: using pytest.

Write test file:

def test_basic_arithmetic():
    ev = ExpressionEvaluator()
    assert ev.evaluate("1+2") == 3.0
    assert ev.evaluate("5-3") == 2.0
    assert ev.evaluate("2*3") == 6.0
    assert ev.evaluate("8/4") == 2.0

def test_precedence():
    ev = ExpressionEvaluator()
    assert ev.evaluate("1+2*3") == 7.0
    assert ev.evaluate("1*2+3") == 5.0
    assert ev.evaluate("1+2+3*4") == 15.0
    assert ev.evaluate("10/2-3") == 2.0

def test_parentheses():
    ev = ExpressionEvaluator()
    assert ev.evaluate("(1+2)*3") == 9.0
    assert ev.evaluate("2*(3+4)") == 14.0
    assert ev.evaluate("((1+2))") == 3.0
    assert ev.evaluate("-(2+3)") == -5.0  # unary minus with parentheses

def test_unary_minus():
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("+5") == 5.0
    assert ev.evaluate("-(-2)") == 2.0
    assert ev.evaluate("-(2+1)*3") == -9.0

def test_errors():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1/0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(1+2")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("1++2")
    with pytest.raises(ValueError, match="Invalid character"):
        ev.evaluate("1$a")

We need to ensure messages match. We'll raise specific messages.

In _tokenize, raise ValueError for invalid char.

In evaluate, raise ValueError for empty expression.

In _parse_factor, raise ValueError for mismatched parentheses.

In _parse_term division, raise ValueError for division by zero.

In _parse_expr after parsing, if leftover tokens raise ValueError for unexpected token.

Now produce final answer: Provide class code and test code in separate blocks or combined.

We'll include imports: from typing import List, Optional.

Add docstrings.

Let's craft final.



## Output

**ExpressionEvaluator implementation**

```python
from __future__ import annotations
from typing import List, Optional


class ExpressionEvaluator:
    """
    Recursive‑descent parser for arithmetic expressions.

    Supports:
        * binary operators +, -, *, / with correct precedence
        * parentheses for grouping
        * unary plus/minus (e.g. "-3", "-(2+1)")
        * floating point numbers (e.g. "3.14")
    Raises ValueError with a descriptive message for:
        * mismatched parentheses
        * division by zero
        * invalid tokens / characters
        * empty expression
    """

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
        self._tokens: List[str] = self._tokenize(expr)
        if not self._tokens:
            raise ValueError("Empty expression")

        self._pos: int = 0
        result = self._parse_expr()
        if self._pos != len(self._tokens):
            # Something left over → unexpected token
            raise ValueError(f"Unexpected token: {self._tokens[self._pos]}")
        return result

    # -----------------------------------------------------------------
    # Tokeniser
    # -----------------------------------------------------------------
    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert the input string into a list of tokens.

        Tokens are numbers (as strings), '+', '-', '*', '/', '(', ')'.

        Raises
        ------
        ValueError
            If an invalid character is encountered or a number contains
            more than one decimal point.
        """
        tokens: List[str] = []
        i = 0
        n = len(expr)

        while i < n:
            ch = expr[i]

            if ch.isspace():
                i += 1
                continue

            if ch in "+-*/()":
                tokens.append(ch)
                i += 1
                continue

            if ch.isdigit() or ch == ".":
                start = i
                dot_seen = False
                while i < n and (expr[i].isdigit() or expr[i] == "."):
                    if expr[i] == ".":
                        if dot_seen:
                            raise ValueError("Invalid number format")
                        dot_seen = True
                    i += 1
                tokens.append(expr[start:i])
                continue

            # Anything else is illegal
            raise ValueError(f"Invalid character: {ch}")

        return tokens

    # -----------------------------------------------------------------
    # Parser helpers
    # -----------------------------------------------------------------
    def _peek(self) -> Optional[str]:
        """Return the current token without consuming it, or None at EOF."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self) -> str:
        """Consume and return the current token; raise if at EOF."""
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
        return tok

    # -----------------------------------------------------------------
    # Grammar:
    #   expr   ::= term (('+' | '-') term)*
    #   term   ::= factor (('*' | '/') factor)*
    #   factor ::= ('+' | '-') factor | number | '(' expr ')'
    # -----------------------------------------------------------------
    def _parse_expr(self) -> float:
        """Parse addition and subtraction."""
        value = self._parse_term()
        while self._peek() in ("+", "-"):
            op = self._consume()
            right = self._parse_term()
            if op == "+":
                value += right
            else:  # op == "-"
                value -= right
        return value

    def _parse_term(self) -> float:
        """Parse multiplication and division."""
        value = self._parse_factor()
        while self._peek() in ("*", "/"):
            op = self._consume()
            right = self._parse_factor()
            if op == "*":
                value *= right
            else:  # op == "/"
                if right == 0.0:
                    raise ValueError("Division by zero")
                value /= right
        return value

    def _parse_factor(self) -> float:
        """Parse a factor: unary +/- , number, or parenthesised expression."""
        token = self._peek()

        # Unary plus/minus
        if token == "+":
            self._consume()
            return self._parse_factor()
        if token == "-":
            self._consume()
            return -self._parse_factor()

        # Parenthesised sub‑expression
        if token == "(":
            self._consume()  # consume '('
            inner = self._parse_expr()
            if self._peek() != ")":
                raise ValueError("Mismatched parentheses")
            self._consume()  # consume ')'
            return inner

        # Number literal
        if token is None:
            raise ValueError("Unexpected end of expression")
        try:
            num = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid token: {token}") from exc
        self._consume()
        return num
```

---

**Pytest test suite**

```python
import pytest
from expression_evaluator import ExpressionEvaluator  # adjust import if needed


def test_basic_arithmetic():
    ev = ExpressionEvaluator()
    assert ev.evaluate("1+2") == 3.0
    assert ev.evaluate("5-3") == 2.0
    assert ev.evaluate("2*3") == 6.0
    assert ev.evaluate("8/4") == 2.0


def test_precedence():
    ev = ExpressionEvaluator()
    assert ev.evaluate("1+2*3") == 7.0      # * before +
    assert ev.evaluate("1*2+3") == 5.0      # * before +
    assert ev.evaluate("1+2+3*4") == 15.0   # * before +
    assert ev.evaluate("10/2-3") == 2.0     # / before -


def test_parentheses():
    ev = ExpressionEvaluator()
    assert ev.evaluate("(1+2)*3") == 9.0
    assert ev.evaluate("2*(3+4)") == 14.0
    assert ev.evaluate("((1+2))") == 3.0
    assert ev.evaluate("-(2+3)") == -5.0   # unary minus with parentheses


def test_unary_minus():
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("+5") == 5.0
    assert ev.evaluate("-(-2)") == 2.0
    assert ev.evaluate("-(2+1)*3") == -9.0  # unary minus applies to the parenthesised sum


def test_error_cases():
    ev = ExpressionEvaluator()

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1/0")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(1+2")

    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")

    # Invalid token (two operators in a row)
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("1++2")

    # Invalid character
    with pytest.raises(ValueError, match="Invalid character"):
        ev.evaluate("1$a")
```

Save the class in a file named `expression_evaluator.py` and the tests in a file such as `test_expression_evaluator.py`. Run the tests with `pytest`. All tests should pass, confirming correct handling of arithmetic, precedence, parentheses, unary minus, and error conditions.
