# Gemma 4 31B 4bit MLX — expression_evaluator

## Thinking (3584 chars)

```

The objective is to build a mathematical expression evaluator in Python using a recursive descent parser. It needs to handle basic operators, precedence, parentheses, unary minus, and floating-point numbers, while implementing specific error handling and following a class-based structure.

    *   *Grammar Definition:*
        *   `expression` $\rightarrow$ `term` { (`+` | `-`) `term` }
        *   `term` $\rightarrow$ `factor` { (`*` | `/`) `factor` }
        *   `factor` $\rightarrow$ `unary`
        *   `unary` $\rightarrow$ `-` `unary` | `primary`
        *   `primary` $\rightarrow$ `number` | `(` `expression` `)`

    *   *Lexer (Tokenizer):* I need a way to break the input string into tokens (numbers, operators, parentheses). Regular expressions are usually the cleanest way to do this.

    *   *Parser (Recursive Descent):*
        *   `evaluate()`: Entry point.
        *   `parse_expression()`: Handles addition and subtraction.
        *   `parse_term()`: Handles multiplication and division.
        *   `parse_unary()`: Handles the unary minus.
        *   `parse_primary()`: Handles numbers and nested expressions.

    *   *Error Handling:*
        *   Mismatched parentheses: Check if the current token is `)` when not expected, or if tokens remain after the main expression is parsed.
        *   Division by zero: Check divisor before dividing.
        *   Invalid tokens: The lexer should identify characters that don't fit the pattern.
        *   Empty expressions: Check if the token list is empty.

    *   *Class Structure:*
        ```python
        class ExpressionEvaluator:
            def evaluate(self, expr: str) -> float:
                # Tokenize
                # Parse
                # Return result
        ```

    *   *Lexer:* Use `re.finditer` or `re.findall`. A pattern like `r'\d*\.\d+|\d+|[+\-*/\(\)]'` should work. Wait, I need to handle whitespace. I'll filter out whitespace or use a regex that ignores it.

    *   *Parser State:* I'll use a pointer (index) to keep track of the current token.

    *   *Recursive Descent Logic:*
        *   `parse_expression`: calls `parse_term`, then loops while `+` or `-` is found.
        *   `parse_term`: calls `parse_unary`, then loops while `*` or `/` is found.
        *   `parse_unary`: if `-`, return `-1 * parse_unary()`. Else, call `parse_primary()`.
        *   `parse_primary`: if `(`, call `parse_expression()` and expect `)`. Else, parse a number.

    *   *Edge Case: Unary Minus:* The grammar `unary -> - unary | primary` allows for `--5` (which is 5). This is standard math behavior.

    *   *Edge Case: Division by Zero:* In `parse_term`, if the operator is `/` and the result of `parse_unary` is 0, raise `ValueError`.

    *   *Edge Case: Mismatched Parentheses:*
        *   If `parse_primary` sees `(`, it calls `parse_expression`. After that, it *must* see `)`. If not, raise `ValueError`.
        *   After the top-level `parse_expression` finishes, if there are tokens left (like an extra `)`), raise `ValueError`.

    *   *Edge Case: Invalid Tokens:* If the lexer finds something that doesn't match the regex, or if the parser encounters a token it doesn't expect, raise `ValueError`.

    *   *Type Hints:* Use `List[str]`, `float`, `str`.
    *   *Docstrings:* Brief and clear.
    *   *Pytest:* Create a separate test file or a block of code.

    *   *Lexer:* `r'\d+\.\d+|\d+|[+\-*/\(\)]'`
    *   *Token consumption:* A helper method `_consume()` to move the pointer.
    *   *Peek:* A helper method `_peek()` to look at the current token without moving.
```

## Output

```python
import re
from typing import List

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic,
    operator precedence, parentheses, and unary minus using a 
    recursive descent parser.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical string expression and returns the result as a float.
        
        Args:
            expr: The string expression to evaluate.
            
        Returns:
            The result of the evaluation.
            
        Raises:
            ValueError: For division by zero, mismatched parentheses, 
                        invalid tokens, or empty expressions.
        """
        # Tokenize the input string
        # Pattern matches: floating point numbers, integers, and single-character operators/parens
        token_pattern = r'\d*\.\d+|\d+|[+\-*/\(\)]'
        # Remove whitespace and find all matches
        self._tokens = re.findall(token_pattern, expr.replace(" ", ""))
        
        # Check for invalid characters by comparing reconstructed string with original (minus whitespace)
        cleaned_expr = expr.replace(" ", "")
        if "".join(self._tokens) != cleaned_expr:
            raise ValueError(f"Invalid tokens found in expression: {expr}")

        if not self._tokens:
            raise ValueError("Expression is empty")

        self._pos = 0
        result = self._parse_expression()

        # If we've finished parsing but there are tokens left, it's a syntax error (e.g., extra closing paren)
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")

        return float(result)

    def _peek(self) -> str:
        """Returns the current token without advancing the pointer."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else ""

    def _consume(self) -> str:
        """Returns the current token and advances the pointer."""
        token = self._peek()
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Handles addition and subtraction (lowest precedence)."""
        node = self._parse_term()

        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                node += right
            else:
                node -= right
        return node

    def _parse_term(self) -> float:
        """Handles multiplication and division."""
        node = self._parse_unary()

        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_unary()
            if op == '*':
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                node /= right
        return node

    def _parse_unary(self) -> float:
        """Handles unary minus."""
        if self._peek() == '-':
            self._consume()  # consume '-'
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and grouped expressions (highest precedence)."""
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            return result
        
        try:
            self._consume()
            return float(token)
        except (ValueError, TypeError):
            raise ValueError(f"Expected number or '(', found: {token}")

# ==========================================
# Pytest Tests
# ==========================================
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    assert evaluator.evaluate("10 - 4 - 2") == 4.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Division before addition
    assert evaluator.evaluate("10 + 20 / 5") == 14.0
    # Left to right for same precedence
    assert evaluator.evaluate("10 - 5 + 2") == 7.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    assert evaluator.evaluate("((1 + 1) * (2 + 2))") == 8.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("5 + -3") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0  # Double negative

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    # Mismatched parentheses (missing closing)
    with pytest.raises(ValueError, match="missing closing '\)'"):
        evaluator.evaluate("(1 + 2")
        
    # Mismatched parentheses (extra closing)
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("1 + 2)")
        
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid tokens"):
        evaluator.evaluate("3 @ 4")
        
    # Empty expression
    with pytest.raises(ValueError, match="Expression is empty"):
        evaluator.evaluate("   ")

if __name__ == "__main__":
    # To run tests without pytest installed, you can use:
    # pytest this_file.py
    print("Run with 'pytest <filename>.py' to execute tests.")
```
