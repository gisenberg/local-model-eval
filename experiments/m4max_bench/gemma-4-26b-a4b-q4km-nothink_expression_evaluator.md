# Gemma 4 26B-A4B Q4_K_M — expression_evaluator

## Output

To solve this problem, we will implement a **Recursive Descent Parser**. This approach involves two main stages:
1.  **Lexer (Tokenizer):** Converts the raw string into a stream of meaningful tokens (numbers, operators, parentheses).
2.  **Parser:** Consumes tokens according to a formal grammar to compute the result.

### Grammar Definition
To handle precedence and unary operators, we use the following grammar:
- `expression` $\to$ `term` { (`+` | `-`) `term` }
- `term` $\to$ `factor` { (`*` | `/`) `factor` }
- `factor` $\to$ `+` `factor` | `-` `factor` | `primary`
- `primary` $\to$ `number` | `(` `expression` `)`

```python
import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    Supports +, -, *, /, parentheses, and unary minus.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string.

        Args:
            expr: The string expression to evaluate.

토        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: For division by zero, mismatched parentheses, 
                        invalid tokens, or empty expressions.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression is empty")

        self._tokenize(expr)
        self._pos = 0
        
        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")
        
        return float(result)

    def _tokenize(self, expr: str) -> None:
        """Converts the input string into a list of tokens."""
        # Regex to match numbers (including floats), operators, and parentheses
        token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')
        self._tokens = token_pattern.findall(expr)
        
        # Check if the reconstructed string matches the original (ignoring whitespace)
        # This helps detect invalid characters like 'abc' or '@'
        reconstructed = "".join(self._tokens)
        original_cleaned = re.sub(r'\s+', '', expr)
        
        # We check if any non-whitespace characters were lost during tokenization
        if len(reconstructed) < len(original_cleaned):
            # Find the first character that wasn't captured
            # This is a simplified check for invalid tokens
            raise ValueError("Expression contains invalid tokens")
        
        if not self._tokens:
            raise ValueError("Expression contains no valid tokens")

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Consumes and returns the current token."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
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
        node = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                node /= right
        return node

    def _parse_factor(self) -> float:
        """Handles unary operators (+ and -)."""
        if self._peek() == '-':
            self._consume()
            return -self._parse_factor()
        if self._peek() == '+':
            self._consume()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and parentheses (highest precedence)."""
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()  # consume ')'
            return result
        
        if token is None:
            raise ValueError("Unexpected end of expression")

        # Check if token is a number
        try:
            self._consume()
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid token: {token}")

# --- Pytest Suite ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 - 2") == 13.0
    assert evaluator.evaluate("10 * 2 / 5") == 4.0
    assert evaluator.evaluate("3.5 * 2") == 7.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Parentheses override precedence
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5 + 3") == -2.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("5 * -2") == -10.0
    assert evaluator.evaluate("--5") == 5.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="invalid tokens|Invalid token"):
        evaluator.evaluate("10 + abc")
    
    # Empty expression
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate("   ")

def test_complex_expression():
    evaluator = ExpressionEvaluator()
    # Deeply nested and mixed operators
    expr = "((10 + 5) * (2 - 1)) / (0.5 + 0.5)"
    assert evaluator.evaluate(expr) == 15.0
```

### Key Implementation Details:

1.  **Recursive Descent Logic**: 
    *   `_parse_expression` handles `+` and `-`.
    *   `_parse_term` handles `*` and `/`.
    *   `_parse_factor` handles unary `-` and `+`. This allows for things like `--5` (which becomes `5`).
    *   `_parse_primary` handles the base cases: numbers and nested `(expression)`.
2.  **Error Handling**:
    *   **Division by Zero**: Explicitly checked during the division operation in `_parse_term`.
    *   **Mismatched Parentheses**: Checked in `_parse_primary` by verifying the presence of a closing `)`.
    *   **Invalid Tokens**: The tokenizer uses regex to find valid patterns. If the length of the reconstructed string is less than the original (excluding whitespace), it means characters were ignored, implying invalid input.
    *   **Empty Expression**: Checked at the start of the `evaluate` method.
3.  **Complexity**:
    *   **Time Complexity**: $O(N)$, where $N$ is the number of characters in the string, as we traverse the string once for tokenization and once for parsing.
    *   **Space Complexity**: $O(N)$ to store the list of tokens.
