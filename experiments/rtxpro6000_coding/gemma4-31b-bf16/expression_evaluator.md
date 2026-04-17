Here is the complete implementation. I have used a recursive descent parser based on a formal grammar to ensure correct operator precedence and grouping.

### Implementation

```python
import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic,
    parentheses, unary minus, and floating point numbers.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates a mathematical expression string.
        
        Args:
            expr: The expression string to evaluate.
        Returns:
            The result as a float.
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or division by zero.
        """
        self._tokens = self._tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError("Expression cannot be empty")

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token '{self._tokens[self._pos]}' at end of expression")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the input string into a list of tokens."""
        # Regex matches: floating point numbers, or single characters (+, -, *, /, (, ))
        token_pattern = r"(\d*\.\d+|\d+|[+\-*/()])"
        tokens = re.findall(token_pattern, expr)
        
        # Validation: Check if the reconstructed string (ignoring whitespace) matches the original
        # This ensures no invalid characters (like 'abc' or '&') were skipped by findall
        cleaned_expr = "".join(expr.split())
        if "".join(tokens) != cleaned_expr:
            # Find the first character that didn't match a token
            # This is a simplified check to identify invalid tokens
            raise ValueError("Expression contains invalid characters")
            
        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Returns the current token and advances the position."""
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
            return -self._parse_unary()  # Recursive to handle --5
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and parentheses (highest precedence)."""
        token = self._consume()

        if token == '(':
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            return result
        
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid number token: {token}")
```

### Test Suite

To run these tests, you will need `pytest` installed (`pip install pytest`). Save the code above in a file named `evaluator.py` and the tests below in `test_evaluator.py`.

```python
import pytest
from evaluator import ExpressionEvaluator

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 2") == 5.0
    assert evaluator.evaluate("3.14 + 2.86") == 6.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Division before subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0
    # Complex precedence
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    # Force addition first
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # Nested parentheses
    assert evaluator.evaluate("2 * (3 + (4 / 2))") == 10.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5 + 2") == -3.0
    assert evaluator.evaluate("5 + (-2)") == 3.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0  # Double negative

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
        
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
        
    # Invalid tokens
    with pytest.raises(ValueError, match="invalid characters"):
        evaluator.evaluate("2 + abc")
        
    # Empty expression
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("")
        
    # Trailing operators/Malformed
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 2 3")
```

### Explanation of the Design

1.  **Grammar**: I implemented a standard expression grammar:
    *   `Expression` $\rightarrow$ `Term` $\{ (+|-) \text{ Term} \}$
    *   `Term` $\rightarrow$ `Unary` $\{ (*|/) \text{ Unary} \}$
    *   `Unary` $\rightarrow$ `-` `Unary` | `Primary`
    *   `Primary` $\rightarrow$ `Number` | `(` `Expression` `)`
2.  **Recursive Descent**: Each grammar rule is mapped to a method. This naturally handles operator precedence (addition is handled at the top level, meaning it's evaluated last) and recursion (parentheses call the top-level `_parse_expression` again).
3.  **Lexing**: I used a regular expression to split the string into a list of tokens. To prevent `re.findall` from silently ignoring invalid characters (like letters), I compare the joined tokens against the original string.
4.  **Complexity**: 
    *   **Time Complexity**: $O(N)$ where $N$ is the length of the string, as we traverse the tokens once.
    *   **Space Complexity**: $O(N)$ to store the tokens and the recursion stack.