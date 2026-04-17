Here is the complete implementation of the `ExpressionEvaluator` using a recursive descent parser.

```python
import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic,
    operator precedence, parentheses, and unary minus.
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
                        has mismatched parentheses, or division by zero occurs.
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
        """
        Converts the input string into a list of tokens.
        """
        # Regex matches floating point numbers or single characters (+, -, *, /, (, ))
        token_pattern = r"(\d+(\.\d+)?|[+\-*/()])"
        tokens = []
        
        # We use finditer to ensure we don't skip invalid characters
        last_end = 0
        for match in re.finditer(token_pattern, expr):
            # Check for invalid characters between tokens
            if expr[last_end:match.start()].strip():
                raise ValueError(f"Invalid token found: {expr[last_end:match.start()].strip()}")
            tokens.append(match.group(0))
            last_end = match.end()
        
        # Check for trailing invalid characters
        if expr[last_end:].strip():
            raise ValueError(f"Invalid token found: {expr[last_end:].strip()}")
            
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
        Grammar: term -> unary { ('*' | '/') unary }
        """
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
        """
        Handles unary minus.
        Grammar: unary -> '-' unary | primary
        """
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Handles numbers and grouped expressions.
        Grammar: primary -> number | '(' expression ')'
        """
        token = self._consume()

        if token == '(':
            result = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            return result
        
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Expected number or '(', found '{token}'")

# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 - 1") == 7.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0

def test_precedence_and_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-3 * -2") == 6.0

def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("10 / 4") == 2.5
    assert evaluator.evaluate("0.1 + 0.2") == pytest.approx(0.3)

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
        
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + abc")
        
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")
        
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3 )")
```

### Implementation Details:

1.  **Grammar Hierarchy**: I implemented a standard recursive descent hierarchy to ensure correct operator precedence:
    *   `Expression` $\rightarrow$ `Term` $\rightarrow$ `Unary` $\rightarrow$ `Primary`.
    *   This ensures that `*` and `/` are evaluated before `+` and `-`, and unary operators are handled before binary ones.
2.  **Lexer**: The `_tokenize` method uses a regular expression to split the string. It specifically checks for characters that don't match the pattern to ensure "invalid tokens" (like letters) trigger a `ValueError`.
3.  **Unary Minus**: The `_parse_unary` method is recursive. This allows for multiple unary operators (e.g., `--5` becomes `5`).
4.  **Error Handling**:
    *   **Division by Zero**: Explicitly checked in `_parse_term`.
    *   **Mismatched Parentheses**: Checked in `_parse_primary` (missing closing) and `evaluate` (extra closing).
    *   **Empty Expressions**: Checked at the start of the `evaluate` method.
5.  **Complexity**:
    *   **Time Complexity**: $O(n)$ where $n$ is the length of the expression string, as we traverse the tokens once.
    *   **Space Complexity**: $O(n)$ to store the tokens and the recursion stack.