# Gemma 4 31B-IT Q4_K_M (turbo4 KV) — expression_evaluator

## Output

```python
import re
from typing import List, Tuple

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic, 
    operator precedence, parentheses, and unary minus using a 
    recursive descent parser.
    """

    def __init__(self):
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: The string expression to evaluate.
            
        Returns:
            The result of the evaluation.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens, 
                        has mismatched parentheses, or division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        # Tokenize the input string
        # Regex matches: floating point numbers, operators, and parentheses
        self._tokens = re.findall(r'\d*\.\d+|\d+|[+\-*/()]', expr)
        
        # Validate that no invalid characters were skipped by the regex
        # We reconstruct the string without whitespace to compare lengths
        cleaned_expr = "".join(expr.split())
        if len("".join(self._tokens)) != len(cleaned_expr):
            # This is a simple check; a more robust one would check if 
            # the tokens actually cover the whole string.
            # Let's refine: check if any character in cleaned_expr is not in tokens.
            # Actually, a better way is to check if the regex missed anything.
            pass 

        # To properly detect invalid tokens, we can use a more strict regex or 
        # check the total length of tokens vs the original string.
        # Let's use a simpler validation: check if the tokens joined match the 
        # non-whitespace version of the input.
        if "".join(self._tokens) != cleaned_expr:
            raise ValueError("Expression contains invalid characters")

        self._pos = 0
        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at position {self._pos}: {self._tokens[self._pos]}")

        return float(result)

    def _peek(self) -> str:
        """Returns the current token without advancing the position."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else ""

    def _consume(self) -> str:
        """Returns the current token and advances the position."""
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
        """Handles unary minus, parentheses, and numbers (highest precedence)."""
        token = self._peek()

        if token == '-':
            self._consume()
            return -self._parse_factor()
        
        if token == '(':
            self._consume() # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume() # consume ')'
            return result

        if token and (token[0].isdigit() or (len(token) > 1 and token[1].isdigit())):
            return float(self._consume())

        raise ValueError(f"Unexpected token: {token if token else 'End of expression'}")

# --- Pytest Tests ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    assert evaluator.evaluate("10 - 2 - 3") == 5.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Division before subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (10 - (3 + 2))") == 10.0
    assert evaluator.evaluate("((1 + 1))") == 2.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0 # Double negative
    assert evaluator.evaluate("-3.14 * 2") == -6.28

def test_error_cases():
    evaluator = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
        
    with pytest.raises(ValueError, match="invalid characters"):
        evaluator.evaluate("2 + a")
        
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate("   ")
```

### Explanation of Implementation:
1.  **Tokenization**: I used `re.findall` to split the string into a list of numbers, operators, and parentheses. This simplifies the parsing logic by removing the need to handle whitespace manually.
2.  **Recursive Descent**:
    *   `_parse_expression`: Handles `+` and `-`. It calls `_parse_term` first to ensure multiplication/division happens first.
    *   `_parse_term`: Handles `*` and `/`. It calls `_parse_factor` to ensure unary operators and parentheses are handled first.
    *   `_parse_factor`: Handles the "atoms" of the expression: numbers, expressions wrapped in `()`, and the unary `-` operator.
3.  **Unary Minus**: By calling `self._parse_factor()` recursively inside the unary minus block, the evaluator supports nested unary operators (e.g., `--5`).
4.  **Error Handling**: 
    *   **Division by Zero**: Explicitly checked during the division operation.
    *   **Mismatched Parentheses**: Checked by verifying that a `(` is always followed by a `)`.
    *   **Invalid Tokens**: Verified by comparing the joined tokens against the original string (minus whitespace).
    *   **Empty Expressions**: Checked at the start of the `evaluate` method.
