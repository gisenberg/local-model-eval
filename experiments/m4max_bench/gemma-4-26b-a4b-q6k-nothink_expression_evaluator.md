# Gemma 4 26B-A4B Q6_K — expression_evaluator

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
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError("No valid tokens found in expression.")

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
        
        # We check if the number of characters in tokens matches the cleaned input
        # Note: This is a heuristic; a more robust way is checking if any non-whitespace 
        # char in expr was skipped by the regex.
        if len(re.sub(r'\s+', '', "".join(re.findall(r'\S', expr)))) != len(reconstructed):
            # Find the first invalid character for a better error message
            for char in expr:
                if not re.match(r'\s|\d|\.|\+|\-|\*|\/|\(|\)', char):
                    raise ValueError(f"Invalid token detected: '{char}'")
            raise ValueError("Invalid expression format.")

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
        elif token == '+':
            self._consume()
            return self._parse_factor()
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
            val = float(self._consume())
            return val
        except ValueError:
            raise ValueError(f"Expected number or '(' but found '{token}'")

# --- Pytest Tests ---

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
    # Division before subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0
    # Complex mix
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 1) * (2 + 2))") == 8.0
    with pytest.raises(ValueError, match="Expected '\)'"):
        evaluator.evaluate("(1 + 2")

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 + -3") == 2.0
    assert evaluator.evaluate("--5") == 5.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("10 + abc")
    # Empty expression
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("   ")
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Expected '\)'"):
        evaluator.evaluate("1 + (2 * 3")
```

### Implementation Details:

1.  **Grammar Structure**: I used a standard recursive descent hierarchy to enforce operator precedence:
    *   `expression` $\rightarrow$ `term` $\pm$ `term` (Lowest precedence)
    *   `term` $\rightarrow$ `factor` $\times / \div$ `factor`
    *   `factor` $\rightarrow$ unary `-` or `+`
    *   `primary` $\rightarrow$ `number` or `( expression )` (Highest precedence)
2.  **Tokenization**: Uses `re.findall` to split the string into numbers and operators. It includes a validation step to ensure no illegal characters (like letters) are present in the input.
3.  **Unary Minus**: Handled in the `_parse_factor` method, which allows for recursive unary operators (e.g., `--5` becomes `5`).
4.  **Error Handling**: 
    *   `Division by zero` is caught during the division operation.
    *   `Mismatched parentheses` are caught when `_consume(')')` fails to find a closing bracket.
    *   `Invalid tokens` are caught during the regex validation phase.
    *   `Empty expressions` are caught at the start of the `evaluate` method.
5.  **Complexity**: 
    *   **Time Complexity**: $O(N)$, where $N$ is the length of the string, as we visit each token once.
    *   **Space Complexity**: $O(N)$ for the token list and the recursion stack.
