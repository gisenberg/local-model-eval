# Gemma 4 26B-A4B Q6_K (f16 @ 32K, new base default ub) — expression_evaluator

**Hypothesis:** Refresh baseline: f16 at 32K works on new base without -ub 256

## Output

Here is a complete implementation of a mathematical expression evaluator using a recursive descent parser.

```python
import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser
    to handle arithmetic operations, precedence, and unary operators.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates a mathematical expression string.

        Args:
            expr: The string expression to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or results in division by zero.
        """
        if not expr.strip():
            raise ValueError("Expression cannot be empty.")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError("Expression contains no valid tokens.")

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at end of expression: '{self._tokens[self._pos]}'")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of meaningful tokens using regex.
        """
        # Pattern matches: floats/integers, operators (+, -, *, /), or parentheses
        token_pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        tokens = re.findall(token_pattern, expr)
        
        # Validation: Check if the reconstructed string matches the original (ignoring whitespace)
        # This is a simple way to detect invalid characters like 'a', '$', etc.
        cleaned_expr = re.sub(r'\s+', '', expr)
        reconstructed = "".join(tokens)
        
        # Note: This simple check might fail on complex regex, but for this scope, 
        # it ensures we didn't skip invalid characters.
        if len(reconstructed) < len(cleaned_expr.replace('.', '')): # Basic check for invalid chars
             # A more robust way is to check if any non-whitespace char was ignored
             pass 
        
        # Better validation: check if any character in the original string is not in our allowed set
        allowed_chars = set("0123456789.+-*/() \t\n\r")
        for char in expr:
            if char not in allowed_chars:
                raise ValueError(f"Invalid character in expression: '{char}'")

        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self, expected: Optional[str] = None) -> str:
        """Consumes the current token and returns it. Optionally validates the token."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression.")
        if expected and token != expected:
            raise ValueError(f"Expected '{expected}' but found '{token}'")
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Handles addition and subtraction (lowest precedence).
        Grammar: expression -> term { ('+' | '-') term }
        """
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
        """
        Handles multiplication and division.
        Grammar: term -> factor { ('*' | '/') factor }
        """
        node = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero.")
                node /= right
        return node

    def _parse_factor(self) -> float:
        """
        Handles unary minus and parentheses.
        Grammar: factor -> '-' factor | primary
        """
        token = self._peek()
        if token == '-':
            self._consume()
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Handles numbers and parenthesized expressions.
        Grammar: primary -> number | '(' expression ')'
        """
        token = self._peek()

        if token == '(':
            self._consume('(')
            result = self._parse_expression()
            self._consume(')')
            return result
        
        if token is None:
            raise ValueError("Unexpected end of expression.")

        # Try to parse as a number
        try:
            # We consume the token and convert to float
            val = float(self._consume())
            return val
        except ValueError:
            raise ValueError(f"Invalid token: '{token}'")
        except IndexError:
            raise ValueError("Unexpected end of expression.")

# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 - 2") == 13.0
    assert evaluator.evaluate("10 * 5 / 2") == 25.0
    assert evaluator.evaluate("3.5 * 2") == 7.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Division before subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0
    # Complex precedence
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 1) * (2 + 2))") == 8.0
    with pytest.raises(ValueError, match="Expected '\)'"):
        evaluator.evaluate("(2 + 3")

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 + --3") == 8.0  # Double unary
    assert evaluator.evaluate("-5 * -2") == 10.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    # Empty expression
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + a")
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Expected '\)'"):
        evaluator.evaluate("2 + (3 * 4")
```

### Implementation Details:

1.  **Grammar Structure**: I used a standard recursive descent hierarchy to enforce operator precedence:
    *   `expression` $\rightarrow$ `term` $\pm$ `term` (Lowest precedence)
    *   `term` $\rightarrow$ `factor` $\times / \div$ `factor`
    *   `factor` $\rightarrow$ `-` `factor` (Unary minus)
    *   `primary` $\rightarrow$ `number` or `(` `expression` `)` (Highest precedence)
2.  **Tokenization**: Uses `re.findall` to split the string into numbers and operators. I added a manual character check to ensure no illegal characters (like letters) are passed through.
3.  **Unary Minus**: Handled in the `_parse_factor` method. Because it calls itself recursively (`-self._parse_factor()`), it naturally supports multiple unary operators like `--5`.
4.  **Error Handling**: 
    *   **Division by zero**: Checked during the division operation in `_parse_term`.
    *   **Mismatched Parentheses**: Handled by the `_consume` method which expects a specific token (like `)`) after an expression is parsed.
    *   **Invalid Tokens**: Handled by checking characters during tokenization and attempting `float()` conversion in `_parse_primary`.
5.  **Complexity**: 
    *   **Time Complexity**: $O(n)$, where $n$ is the length of the string, as we visit each token exactly once.
    *   **Space Complexity**: $O(n)$ for the token list and the recursion stack.
