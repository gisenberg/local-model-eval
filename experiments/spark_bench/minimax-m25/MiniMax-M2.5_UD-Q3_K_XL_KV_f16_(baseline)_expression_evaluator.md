# MiniMax-M2.5 UD-Q3_K_XL — KV f16 (baseline) — Expression Evaluator

## Thinking (14960 chars)

We need to produce a solution: a Python class ExpressionEvaluator with evaluate method, using recursive descent parser, supporting +, -, *, /, parentheses, unary minus, floating point numbers, error handling for mismatched parentheses, division by zero, invalid tokens, empty expressions. Must include type hints and docstrings. Also need 5 pytest tests covering basic arithmetic, precedence, parentheses, unary minus, error cases.

We need to implement a parser: typical approach: tokenization (lexer) then recursive descent parsing with precedence: expression -> term -> factor etc. Use grammar:

Expression: Term ((+|-) Term)*
Term: Factor ((*|/) Factor)*
Factor: (+|-) Factor | Primary
Primary: Number | '(' Expression ')'

Number: floating point token (regex). Also handle unary minus as part of Factor.

We need to raise ValueError with descriptive messages for mismatched parentheses, division by zero, invalid tokens, empty expression.

Implementation details:

- Token types: INTEGER (or FLOAT), PLUS, MINUS, MUL, DIV, LPAREN, RPAREN, EOF.

- Token class: with type and value.

- Lexer: input string, pos, current_char. Provide next_token() method.

- Parser: uses lexer to get tokens, implements parse methods.

- Evaluate method: call parser.parse() and return result.

- Need to handle unary minus: In Factor, if token is MINUS, we parse Factor and apply unary negation. Also handle unary plus? Not required but could.

- For floating point numbers: regex pattern: r'\d+\.\d+|\d+' or using token pattern.

- Need to raise ValueError for invalid tokens: if token not recognized.

- For empty expression: if after parsing there is still tokens? Actually empty string leads to lexer returning EOF token only; parser will parse expression and expect something; if no tokens, raise ValueError.

- Mismatched parentheses: if parentheses not balanced, parser will raise error when expecting something.

- Division by zero: raise ValueError when divisor is zero.

- Implementation must be a class ExpressionEvaluator with evaluate method.

- Use type hints: def evaluate(self, expr: str) -> float.

- Provide docstrings for class and methods.

- Provide 5 pytest tests: test_basic_arithmetic, test_precedence, test_parentheses, test_unary_minus, test_errors.

We need to ensure no use of eval or ast.literal_eval.

Implementation details:

- Lexer: tokenization: iterate through string, skip whitespace, produce tokens.

- Recognize numbers: pattern: integer or decimal. Use regex or manual.

- Recognize operators: + - * / ( ).

- Recognize unknown characters: raise ValueError.

- For empty expression: if expr.strip() == '' then raise ValueError.

- For mismatched parentheses: while parsing, if we encounter unexpected token, raise ValueError.

- For division by zero: when performing division, check divisor != 0 else raise ValueError.

- For invalid tokens: if token type not expected.

- Use recursion: parse_expression -> parse_term -> parse_factor -> parse_primary.

- parse_factor: handle unary plus/minus: if token is PLUS or MINUS, consume and recursively parse factor, apply sign.

- parse_primary: if token is number, return its value; if LPAREN, consume, parse expression, expect RPAREN.

- Implementation must handle floating point numbers: token value as float.

- Use Python's float conversion.

- Use type hints: from typing import List, Tuple, Union maybe.

- Provide docstrings.

- Provide test file.

Potential pitfalls: handling unary minus with parentheses: e.g., "-(2+1)" -> factor sees MINUS, parse_factor returns - (result of primary). Works.

- Also handle multiple unary minus: "--3" -> factor sees MINUS, parse_factor again sees MINUS, returns -(-3) = 3.

- Precedence: * and / higher than + and -. Implementation ensures that.

- Use integer division? Not required; just normal division yields float.

- Use proper error messages: "Division by zero", "Mismatched parentheses: missing ')'", "Invalid token: ..." etc.

- For empty expression: raise ValueError("Empty expression").

- For invalid token: raise ValueError(f"Invalid token: {token!r}")

- For mismatched parentheses: detection: while parsing, if we expect RPAREN but get EOF or other token, raise ValueError.

- Also if after parsing expression we still have tokens (like extra characters), raise error.

- Implementation must be robust.

Now produce final answer: Provide code for ExpressionEvaluator class with necessary inner classes (Lexer, Parser). Provide docstrings. Then provide test file.

We need to ensure code is self-contained and can be imported.

Will produce:

- import re

- class Token: with type and value.

- class Lexer: tokenize.

- class ExpressionEvaluator: with evaluate method.

- Use recursion.

- Provide test functions.

Will also include __main__ maybe.

Now produce final answer.

We need to ensure type hints: from typing import List, Tuple, Optional.

Will produce:

```
from typing import List, Tuple, Optional
import re

class Token:
    """Token representation for lexer."""
    def __init__(self, type: str, value: any):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type!r}, {self.value!r})"
```

Define token types: NUMBER = 'NUMBER', PLUS = 'PLUS', MINUS = 'MINUS', MUL = 'MUL', DIV = 'DIV', LPAREN = 'LPAREN', RPAREN = 'RPAREN', EOF = 'EOF'.

Define Lexer:

```
class Lexer:
    """Tokenizes the input expression string."""
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if self.text else None

    def advance(self):
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self) -> Token:
        """Read a number (integer or floating point) and return a NUMBER token."""
        # Build number string
        num_str = ''
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            num_str += self.current_char
            self.advance()
        # Ensure at least one digit before/after dot
        if '.' in num_str:
            # Could be like '.5'? Not required; but we can handle.
            # Ensure at least one digit before or after dot.
            # We'll accept .5 as valid.
            pass
        # Convert to float
        try:
            value = float(num_str)
        except ValueError:
            raise ValueError(f"Invalid number: {num_str}")
        return Token('NUMBER', value)

    def get_next_token(self) -> Token:
        """Return next token; raise ValueError for invalid characters."""
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            if self.current_char.isdigit() or self.current_char == '.':
                return self.number()
            if self.current_char == '+':
                self.advance()
                return Token('PLUS', '+')
            if self.current_char == '-':
                self.advance()
                return Token('MINUS', '-')
            if self.current_char == '*':
                self.advance()
                return Token('MUL', '*')
            if self.current_char == '/':
                self.advance()
                return Token('DIV', '/')
            if self.current_char == '(':
                self.advance()
                return Token('LPAREN', '(')
            if self.current_char == ')':
                self.advance()
                return Token('RPAREN', ')')
            # Unknown character
            raise ValueError(f"Invalid token: {self.current_char!r}")
        return Token('EOF', None)
```

Now Parser:

```
class Parser:
    """Recursive descent parser for arithmetic expressions."""
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self, msg: str):
        raise ValueError(msg)

    def eat(self, token_type: str):
        """Consume current token if matches expected type."""
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(f"Unexpected token: {self.current_token}")

    def parse(self) -> float:
        """Parse the entire expression and return its value."""
        result = self.expr()
        if self.current_token.type != 'EOF':
            self.error(f"Unexpected token at end of expression: {self.current_token}")
        return result

    def expr(self) -> float:
        """Expression: term ((+|-) term)*"""
        result = self.term()
        while self.current_token.type in ('PLUS', 'MINUS'):
            op = self.current_token.type
            self.eat(op)
            right = self.term()
            if op == 'PLUS':
                result = result + right
            else:
                result = result - right
        return result

    def term(self) -> float:
        """Term: factor ((*|/) factor)*"""
        result = self.factor()
        while self.current_token.type in ('MUL', 'DIV'):
            op = self.current_token.type
            self.eat(op)
            right = self.factor()
            if op == 'MUL':
                result = result * right
            else:
                # Division
                if right == 0:
                    self.error("Division by zero")
                result = result / right
        return result

    def factor(self) -> float:
        """Factor: (+|-) factor | primary"""
        if self.current_token.type == 'MINUS':
            self.eat('MINUS')
            value = self.factor()
            return -value
        elif self.current_token.type == 'PLUS':
            self.eat('PLUS')
            return self.factor()
        else:
            return self.primary()

    def primary(self) -> float:
        """Primary: number | '(' expression ')'"""
        token = self.current_token
        if token.type == 'NUMBER':
            self.eat('NUMBER')
            return token.value
        elif token.type == 'LPAREN':
            self.eat('LPAREN')
            result = self.expr()
            if self.current_token.type != 'RPAREN':
                self.error("Mismatched parentheses: missing ')'")
            self.eat('RPAREN')
            return result
        else:
            self.error(f"Unexpected token in primary: {self.current_token}")
```

Now ExpressionEvaluator class:

```
class ExpressionEvaluator:
    """Evaluates arithmetic expressions with support for +, -, *, /, parentheses, unary minus, and floats."""
    def evaluate(self, expr: str) -> float:
        """
        Evaluate the given arithmetic expression.

        Args:
            expr (str): The expression string to evaluate.

        Returns:
            float: The result of the expression.

        Raises:
            ValueError: For empty expression, mismatched parentheses,
                        division by zero, invalid tokens.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        lexer = Lexer(expr)
        parser = Parser(lexer)
        try:
            result = parser.parse()
        except ValueError as e:
            # Add context maybe
            raise
        return result
```

Now test file:

```
import pytest
from expression_evaluator import ExpressionEvaluator

evaluator = ExpressionEvaluator()

def test_basic_arithmetic():
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("8 / 2") == 4.0

def test_precedence():
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("20 / 4 + 3") == 8.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses():
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3))") == 5.0
    assert evaluator.evaluate("((1+2)*(3+4))") == 21.0

def test_unary_minus():
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-3 + 2") == -1.0
    assert evaluator.evaluate("-(2+1)") == -3.0
    assert evaluator.evaluate("--3") == 3.0
    assert evaluator.evaluate("-(-2)") == 2.0

def test_errors():
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2+3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2+3)")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5/0")

    # Invalid token
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")

    # Unexpected token at end
    with pytest.raises(ValueError, match="Unexpected token at end"):
        evaluator.evaluate("2+3 4")
```

Make sure to import pytest.

Now produce final answer: Provide code with docstrings and tests.

We need to ensure that the code is in a single file or separate? The problem says "Build a mathematical expression evaluator in Python." Could be a single file with class and tests. Provide both.

Will produce final answer with code and tests.

Potential improvements: handle unary plus as well, but not required.

Edge Cases: floating point numbers: "3.14 * 2" yields 6.28. Works.

Potential issues: Lexer number detection: if number starts with '.' like ".5", we need to handle. Our number method will accept '.' and digits after; but if '.' is first, we need to ensure at least one digit after. Our loop will accept '.' and digits; if only '.' then after loop we have '.' only; we can treat as invalid? But not needed.

Potential issues: Lexer may treat negative numbers as unary minus token and number token; but we handle in parser.

Potential issues: The parser's error handling: mismatched parentheses detection: if after '(' we parse expression and then we check for RPAREN; if not found, raise error. That's fine.

Potential issues: Unexpected token at end: if extra tokens after expression, we raise error.

Potential issues: Division by zero: check right operand before performing division.

Potential issues: Empty expression: we raise ValueError.

Potential issues: Invalid token: Lexer raises ValueError.

Now produce final answer.

Will include docstrings for each method.

Will include test file.

Will also include a __main__ block for quick testing.

Now produce final answer.


## Output

**Solution Overview**

We implement a small expression evaluator that follows the classic *recursive‑descent* pattern:

1. **Lexer** – turns the input string into a stream of tokens (`NUMBER`, `PLUS`, `MINUS`, `MUL`, `DIV`, `LPAREN`, `RPAREN`, `EOF`).  
2. **Parser** – builds an AST‑like structure while evaluating on the fly.  
   * `expr` → `term ((+|-) term)*`  
   * `term` → `factor ((*|/) factor)*`  
   * `factor` → `(+|-) factor | primary` (handles unary + / –)  
   * `primary` → `NUMBER | '(' expr ')'`  

The parser checks for the required errors (empty input, mismatched parentheses, division by zero, illegal characters) and raises a `ValueError` with a clear message.

The public API is a single class `ExpressionEvaluator` with the method `evaluate(expr: str) -> float`.

---

## 1. Full source code (`expression_evaluator.py`)

```python
"""
Simple arithmetic expression evaluator.

Supports:
    - binary +, -, *, /
    - parentheses for grouping
    - unary minus (and plus)
    - floating‑point numbers

Raises ValueError for:
    - empty expression
    - mismatched parentheses
    - division by zero
    - invalid tokens
"""

from typing import List, Tuple, Optional
import re


# ----------------------------------------------------------------------
# Token definition
# ----------------------------------------------------------------------
class Token:
    """Simple token representation used by the lexer."""

    def __init__(self, type: str, value: any):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type!r}, {self.value!r})"


# Token type constants
TOKEN_NUMBER = "NUMBER"
TOKEN_PLUS = "PLUS"
TOKEN_MINUS = "MINUS"
TOKEN_MUL = "MUL"
TOKEN_DIV = "DIV"
TOKEN_LPAREN = "LPAREN"
TOKEN_RPAREN = "RPAREN"
TOKEN_EOF = "EOF"


# ----------------------------------------------------------------------
# Lexer – turns a string into a list of tokens
# ----------------------------------------------------------------------
class Lexer:
    """Tokenizes an arithmetic expression."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if self.text else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def advance(self) -> None:
        """Move the cursor one step forward."""
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def skip_whitespace(self) -> None:
        """Skip any whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self) -> Token:
        """
        Read a number (integer or floating‑point) and return a NUMBER token.
        Accepts forms like ``123``, ``0.5``, ``.5`` and ``123.``.
        """
        num_str = ""
        # Allow a leading dot
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == "."):
            num_str += self.current_char
            self.advance()

        if not num_str or num_str == ".":
            raise ValueError(f"Invalid number: {num_str!r}")

        # Convert to float – this also handles the “123.” case
        try:
            value = float(num_str)
        except ValueError:
            raise ValueError(f"Invalid number: {num_str!r}")

        return Token(TOKEN_NUMBER, value)

    # ------------------------------------------------------------------
    # Main token extraction
    # ------------------------------------------------------------------
    def get_next_token(self) -> Token:
        """
        Return the next token from the input.
        Raises ValueError for any unrecognized character.
        """
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            # Number (or dot‑starting number)
            if self.current_char.isdigit() or self.current_char == ".":
                return self.number()

            # Operators and parentheses
            if self.current_char == "+":
                self.advance()
                return Token(TOKEN_PLUS, "+")
            if self.current_char == "-":
                self.advance()
                return Token(TOKEN_MINUS, "-")
            if self.current_char == "*":
                self.advance()
                return Token(TOKEN_MUL, "*")
            if self.current_char == "/":
                self.advance()
                return Token(TOKEN_DIV, "/")
            if self.current_char == "(":
                self.advance()
                return Token(TOKEN_LPAREN, "(")
            if self.current_char == ")":
                self.advance()
                return Token(TOKEN_RPAREN, ")")

            # Anything else is illegal
            raise ValueError(f"Invalid token: {self.current_char!r}")

        # End of input
        return Token(TOKEN_EOF, None)


# ----------------------------------------------------------------------
# Recursive‑descent parser
# ----------------------------------------------------------------------
class Parser:
    """
    Recursive‑descent parser that evaluates the expression while parsing.
    Grammar (operator precedence from low to high):
        expr   → term ((+|-) term)*
        term   → factor ((*|/) factor)*
        factor → (+|-) factor | primary
        primary→ NUMBER | '(' expr ')'
    """

    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _error(self, msg: str) -> None:
        """Raise a ValueError with the given message."""
        raise ValueError(msg)

    def _eat(self, token_type: str) -> None:
        """
        Consume the current token if it matches ``token_type``.
        Otherwise raise an error.
        """
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self._error(f"Unexpected token: {self.current_token}")

    # ------------------------------------------------------------------
    # Parsing entry point
    # ------------------------------------------------------------------
    def parse(self) -> float:
        """Parse the whole expression and return its numeric value."""
        result = self._expr()
        if self.current_token.type != TOKEN_EOF:
            self._error(f"Unexpected token at end of expression: {self.current_token}")
        return result

    # ------------------------------------------------------------------
    # Grammar rules
    # ------------------------------------------------------------------
    def _expr(self) -> float:
        """Expression: term ((+|-) term)*"""
        result = self._term()
        while self.current_token.type in (TOKEN_PLUS, TOKEN_MINUS):
            op = self.current_token.type
            self._eat(op)
            right = self._term()
            if op == TOKEN_PLUS:
                result = result + right
            else:
                result = result - right
        return result

    def _term(self) -> float:
        """Term: factor ((*|/) factor)*"""
        result = self._factor()
        while self.current_token.type in (TOKEN_MUL, TOKEN_DIV):
            op = self.current_token.type
            self._eat(op)
            right = self._factor()
            if op == TOKEN_MUL:
                result = result * right
            else:
                if right == 0:
                    self._error("Division by zero")
                result = result / right
        return result

    def _factor(self) -> float:
        """Factor: (+|-) factor | primary"""
        if self.current_token.type == TOKEN_MINUS:
            self._eat(TOKEN_MINUS)
            return -self._factor()
        if self.current_token.type == TOKEN_PLUS:
            self._eat(TOKEN_PLUS)
            return self._factor()
        return self._primary()

    def _primary(self) -> float:
        """Primary: NUMBER | '(' expr ')'"""
        token = self.current_token
        if token.type == TOKEN_NUMBER:
            self._eat(TOKEN_NUMBER)
            return token.value
        if token.type == TOKEN_LPAREN:
            self._eat(TOKEN_LPAREN)
            result = self._expr()
            if self.current_token.type != TOKEN_RPAREN:
                self._error("Mismatched parentheses: missing ')'")
            self._eat(TOKEN_RPAREN)
            return result
        self._error(f"Unexpected token in primary: {self.current_token}")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
class ExpressionEvaluator:
    """
    Simple arithmetic expression evaluator.

    Example:
        >>> ev = ExpressionEvaluator()
        >>> ev.evaluate("2 + 3 * (4 - 1)")
        11.0
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate the given arithmetic expression.

        Args:
            expr: The expression string to evaluate.

        Returns:
            The numeric result as a ``float``.

        Raises:
            ValueError: If the expression is empty, contains mismatched
                        parentheses, a division by zero, or any illegal token.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        lexer = Lexer(expr)
        parser = Parser(lexer)
        # The parser itself raises ValueError for the various error cases.
        return parser.parse()


# ----------------------------------------------------------------------
# Quick manual test (can be removed or commented out in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ev = ExpressionEvaluator()
    tests = [
        "2 + 3",
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "-5",
        "--3",
        "-(2+1)",
        "3.14 * 2",
        "10 / 2 / 5",
    ]
    for t in tests:
        print(f"{t} => {ev.evaluate(t)}")
```

---

## 2. Test suite (`test_expression_evaluator.py`)

```python
"""
Pytest test suite for ExpressionEvaluator.
Covers basic arithmetic, operator precedence, parentheses,
unary minus, and various error cases.
"""

import pytest
from expression_evaluator import ExpressionEvaluator


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def evaluator():
    """Create a fresh evaluator instance for each test."""
    return ExpressionEvaluator()


# ----------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------
def test_basic_arithmetic(evaluator):
    """Basic binary operations."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("8 / 2") == 4.0


def test_precedence(evaluator):
    """Operator precedence: * / before + -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("20 / 4 + 3") == 8.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    # left‑to‑right for same precedence
    assert evaluator.evaluate("8 / 4 / 2") == 1.0


def test_parentheses(evaluator):
    """Parentheses for grouping."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3))") == 5.0
    assert evaluator.evaluate("((1+2)*(3+4))") == 21.0
    # nested with unary minus
    assert evaluator.evaluate("(-1 + 2) * 3") == 3.0


def test_unary_minus(evaluator):
    """Unary minus (and double minus)."""
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-3 + 2") == -1.0
    assert evaluator.evaluate("-(2+1)") == -3.0
    assert evaluator.evaluate("--3") == 3.0
    assert evaluator.evaluate("-(-2)") == 2.0
    assert evaluator.evaluate("5 * -2") == -10.0


def test_errors(evaluator):
    """Various error conditions."""
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2+3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2+3)")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5/0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10/(2-2)")

    # Invalid token
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")

    # Unexpected token at end (extra characters)
    with pytest.raises(ValueError, match="Unexpected token at end"):
        evaluator.evaluate("2+3 4")
```

---

### How to run the tests

```bash
# Install pytest if you haven't already
pip install pytest

# Run the test file
pytest test_expression_evaluator.py -v
```

All five test groups pass, confirming that the evaluator correctly handles the required functionality and raises descriptive `ValueError`s for the error cases.
