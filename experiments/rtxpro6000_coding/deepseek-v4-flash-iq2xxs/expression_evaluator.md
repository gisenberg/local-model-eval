Below is the implementation of the `ExpressionEvaluator` class with a recursive descent parser, followed by five pytest test cases covering the required scenarios.

```python
from typing import List, Tuple, Any, Optional

# Token type constants
TOK_NUMBER = 'NUMBER'
TOK_PLUS = 'PLUS'
TOK_MINUS = 'MINUS'
TOK_TIMES = 'TIMES'
TOK_DIVIDE = 'DIVIDE'
TOK_LPAREN = 'LPAREN'
TOK_RPAREN = 'RPAREN'
TOK_EOF = 'EOF'


class ExpressionEvaluator:
    """
    A recursive descent parser for arithmetic expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating-point numbers.
    Raises ValueError for syntax errors, division by zero, invalid tokens, and empty expressions.
    """

    def __init__(self) -> None:
        self._tokens: List[Tuple[str, Any]] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate the given mathematical expression and return the result as a float.

        Args:
            expr: The expression string to evaluate.

        Returns:
            The computed floating-point result.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        self._tokens = self._tokenize(expr)
        self._pos = 0

        # Empty expression check
        if self._current_token()[0] == TOK_EOF:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        # Ensure all tokens were consumed
        if self._current_token()[0] != TOK_EOF:
            raise ValueError("Unexpected token after expression")

        return result

    # ---------- Tokenizer ----------

    def _tokenize(self, expr: str) -> List[Tuple[str, Any]]:
        """
        Convert the input string into a list of tokens.

        Each token is a tuple (type, value). For numbers, value is a float;
        for operators and parentheses, value is the character; for EOF, value is None.

        Raises ValueError for invalid characters or malformed numbers.
        """
        tokens: List[Tuple[str, Any]] = []
        i = 0
        n = len(expr)

        while i < n:
            # Skip whitespace
            if expr[i].isspace():
                i += 1
                continue

            # Number (including floating point)
            if expr[i].isdigit() or expr[i] == '.':
                start = i
                num_str = ''
                has_dot = False
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number: multiple decimal points")
                        has_dot = True
                    num_str += expr[i]
                    i += 1
                # Validate number string
                if num_str == '.' or num_str == '' or num_str.count('.') > 1:
                    raise ValueError(f"Invalid number: '{num_str}'")
                try:
                    value = float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number: '{num_str}'")
                tokens.append((TOK_NUMBER, value))
                continue

            # Operators and parentheses
            if expr[i] in '+-*/()':
                char = expr[i]
                token_type_map = {
                    '+': TOK_PLUS,
                    '-': TOK_MINUS,
                    '*': TOK_TIMES,
                    '/': TOK_DIVIDE,
                    '(': TOK_LPAREN,
                    ')': TOK_RPAREN,
                }
                tokens.append((token_type_map[char], char))
                i += 1
                continue

            # Invalid character
            raise ValueError(f"Invalid token: '{expr[i]}'")

        # Append EOF marker
        tokens.append((TOK_EOF, None))
        return tokens

    # ---------- Parser helpers ----------

    def _current_token(self) -> Tuple[str, Any]:
        """Return the token at the current position."""
        return self._tokens[self._pos]

    def _advance(self) -> None:
        """Move to the next token."""
        self._pos += 1

    # ---------- Recursive descent parsing ----------

    def _parse_expression(self) -> float:
        """
        Parse an expression: term ( (PLUS|MINUS) term )*
        """
        value = self._parse_term()
        while self._current_token()[0] in (TOK_PLUS, TOK_MINUS):
            op = self._current_token()[1]
            self._advance()
            right = self._parse_term()
            if op == '+':
                value += right
            else:
                value -= right
        return value

    def _parse_term(self) -> float:
        """
        Parse a term: factor ( (TIMES|DIVIDE) factor )*
        """
        value = self._parse_factor()
        while self._current_token()[0] in (TOK_TIMES, TOK_DIVIDE):
            op = self._current_token()[1]
            self._advance()
            right = self._parse_factor()
            if op == '*':
                value *= right
            else:  # division
                if right == 0:
                    raise ValueError("Division by zero")
                value /= right
        return value

    def _parse_factor(self) -> float:
        """
        Parse a factor: (MINUS factor) | primary
        Handles unary minus (e.g., '-3', '-(2+1)').
        """
        if self._current_token()[0] == TOK_MINUS:
            self._advance()
            value = self._parse_factor()
            return -value
        else:
            return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parse a primary: NUMBER | LPAREN expression RPAREN
        Raises ValueError for mismatched parentheses.
        """
        token = self._current_token()
        if token[0] == TOK_NUMBER:
            self._advance()
            return token[1]
        elif token[0] == TOK_LPAREN:
            self._advance()
            value = self._parse_expression()
            # Expect closing parenthesis
            if self._current_token()[0] != TOK_RPAREN:
                raise ValueError("Mismatched parentheses: missing closing parenthesis")
            self._advance()
            return value
        elif token[0] == TOK_RPAREN:
            raise ValueError("Mismatched parentheses: unexpected closing parenthesis")
        elif token[0] == TOK_EOF:
            raise ValueError("Empty expression or incomplete expression")
        else:
            raise ValueError(f"Unexpected token: {token[1]}")
```

## Pytest Tests

```python
import pytest
from expression_evaluator import ExpressionEvaluator  # adjust import as needed

def test_basic_arithmetic() -> None:
    """Test basic addition, subtraction, multiplication, division."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2+3") == 5.0
    assert ev.evaluate("10-4") == 6.0
    assert ev.evaluate("3*5") == 15.0
    assert ev.evaluate("8/2") == 4.0
    assert ev.evaluate("2+3*4") == 14.0  # multiplication before addition

def test_precedence() -> None:
    """Test operator precedence: * and / before + and -."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2+3*4") == 14.0
    assert ev.evaluate("10-6/2") == 7.0
    assert ev.evaluate("2*3+4/2") == 8.0
    assert ev.evaluate("2+3+4*5") == 25.0

def test_parentheses() -> None:
    """Test grouping with parentheses."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2+3)*4") == 20.0
    assert ev.evaluate("10/(2+3)") == 2.0
    assert ev.evaluate("((2+3)*4)") == 20.0
    assert ev.evaluate("(2+3)*(4+5)") == 45.0

def test_unary_minus() -> None:
    """Test unary minus on numbers and expressions."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("--3") == 3.0
    assert ev.evaluate("-(2+1)") == -3.0
    assert ev.evaluate("5+-3") == 2.0
    assert ev.evaluate("-2*3") == -6.0
    assert ev.evaluate("2*-3") == -6.0

def test_error_cases() -> None:
    """Test various error conditions."""
    ev = ExpressionEvaluator()
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")

    # Invalid token
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2+3a")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2?3")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2+3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("2+3)")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("((2+3)")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("5/0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("(2+3)/(1-1)")

    # Invalid number
    with pytest.raises(ValueError, match="Invalid number"):
        ev.evaluate("3.14.15")
    with pytest.raises(ValueError, match="Invalid number"):
        ev.evaluate(".")
```

## Explanation

- **Tokenization**: Converts the input string into a list of tokens. Handles whitespace, numbers (including floating point with optional leading/trailing dot), operators, and parentheses. Raises `ValueError` for invalid characters or malformed numbers.
- **Recursive descent parser**: Implements the grammar:
  - `expression` → `term` ( `+` `term` | `-` `term` )*
  - `term` → `factor` ( `*` `factor` | `/` `factor` )*
  - `factor` → `-` `factor` | `primary`
  - `primary` → `NUMBER` | `(` `expression` `)`
- **Error handling**: Each error case (empty expression, invalid token, mismatched parentheses, division by zero) raises a descriptive `ValueError`.
- **Type hints and docstrings**: Provided for all methods.
- **Tests**: Five pytest functions cover basic arithmetic, precedence, parentheses, unary minus, and error cases.