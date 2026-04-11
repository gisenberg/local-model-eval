# Qwen3.5-122B-A10B Q4_K_M (bartowski) [rotorquant planar3K/f16V] — KV planar3K/f16V (K-only, zero-PPL claim) — Expression Evaluator

## Output

Here is a complete implementation of the `ExpressionEvaluator` using a recursive descent parser, followed by the required `pytest` tests.

### Implementation: `expression_evaluator.py`

```python
import re
from typing import List, Union, Optional

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        self.tokens: List[Union[str, float]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: The mathematical expression string.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: {self.tokens[self.pos]}")

        return result

    def _tokenize(self, expr: str) -> None:
        """
        Converts the input string into a list of tokens (numbers and operators).
        Handles floating point numbers and unary minus detection.
        """
        # Regex to match numbers (int or float), operators, and parentheses
        # Note: We handle unary minus during parsing, but we need to distinguish 
        # it from subtraction. We'll tokenize '-' as an operator and resolve 
        # unary context in the parser.
        token_pattern = r'\s*(?:(\d+\.?\d*)|([+\-*/()]))\s*'
        matches = re.findall(token_pattern, expr)
        
        if not matches:
            # Check if there are non-whitespace characters that didn't match
            if re.search(r'\S', expr):
                raise ValueError(f"Invalid token in expression: {expr}")
            return

        self.tokens = []
        for number, op in matches:
            if number:
                self.tokens.append(float(number))
            elif op:
                self.tokens.append(op)

    def _current_token(self) -> Optional[Union[str, float]]:
        """Returns the current token or None if end of input."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self) -> Union[str, float]:
        """Consumes and returns the current token, advancing the position."""
        token = self._current_token()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Grammar: expression -> term (('+' | '-') term)*
        """
        left = self._parse_term()

        while True:
            token = self._current_token()
            if token == '+':
                self._consume()
                right = self._parse_term()
                left += right
            elif token == '-':
                self._consume()
                right = self._parse_term()
                left -= right
            else:
                break
        
        return left

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Grammar: term -> factor (('*' | '/') factor)*
        """
        left = self._parse_factor()

        while True:
            token = self._current_token()
            if token == '*':
                self._consume()
                right = self._parse_factor()
                left *= right
            elif token == '/':
                self._consume()
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
            else:
                break
        
        return left

    def _parse_factor(self) -> float:
        """
        Parses unary operators and primary values (highest precedence).
        Grammar: factor -> ('-' | '+') factor | '(' expression ')' | number
        """
        token = self._current_token()

        # Handle unary minus or plus
        if token == '-':
            self._consume()
            return -self._parse_factor()
        elif token == '+':
            self._consume()
            return self._parse_factor()

        # Handle parentheses
        if token == '(':
            self._consume()
            result = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses: expected ')'")
            self._consume()
            return result

        # Handle numbers
        if isinstance(token, (int, float)):
            self._consume()
            return float(token)

        # If we reach here, it's an invalid token or unexpected end
        if token is None:
            raise ValueError("Unexpected end of expression")
        
        raise ValueError(f"Invalid token: {token}")
```

### Test Suite: `test_expression_evaluator.py`

```python
import pytest
from expression_evaluator import ExpressionEvaluator

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    """Tests basic addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("8 / 2") == 4.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # Precedence check
    assert evaluator.evaluate("10 / 2 + 3") == 8.0

def test_operator_precedence(evaluator):
    """Tests correct operator precedence (* and / before + and -)."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("1 + 2 * 3 - 4 / 2") == 1 + 6 - 2.0  # 5.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 6 + 20.0  # 26.0

def test_parentheses_grouping(evaluator):
    """Tests grouping with parentheses."""
    assert evaluator.evaluate("(1 + 2) * 3") == 9.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    assert evaluator.evaluate("((2 + 3) * 4) - 5") == 15.0
    assert evaluator.evaluate("(1 + (2 * 3))") == 7.0

def test_unary_minus(evaluator):
    """Tests unary minus support."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-2 + 5") == 3.0
    assert evaluator.evaluate("3 * -4") == -12.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-(3 * -2)") == 6.0

def test_error_cases(evaluator):
    """Tests various error conditions."""
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("1 + 2)")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (2 - 2)")

    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1 + a")
    
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1 @ 2")

    # Unexpected tokens at end
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("1 + 2 +")
```

### How to Run

1.  Save the first code block as `expression_evaluator.py`.
2.  Save the second code block as `test_expression_evaluator.py`.
3.  Install `pytest` if you haven't already:
    ```bash
    pip install pytest
    ```
4.  Run the tests:
    ```bash
    pytest test_expression_evaluator.py -v
    ```

### Key Design Decisions

1.  **Recursive Descent**: The parser is structured into three levels of precedence:
    *   `_parse_expression`: Handles `+` and `-`.
    *   `_parse_term`: Handles `*` and `/`.
    *   `_parse_factor`: Handles unary operators (`-`, `+`), parentheses, and numbers.
    This structure naturally enforces operator precedence without needing complex precedence tables.

2.  **Unary Minus Handling**: The `_parse_factor` method checks if the current token is `-`. If so, it consumes the token and recursively calls `_parse_factor` again. This allows for nested unary operators (e.g., `--5`) and unary operators applied to parenthesized expressions (e.g., `-(2+1)`).

3.  **Tokenization**: A regular expression is used to split the string into numbers and operators. This handles floating-point numbers (e.g., `3.14`) and ignores whitespace.

4.  **Error Handling**:
    *   **Empty Expression**: Checked at the start and after tokenization.
    *   **Mismatched Parentheses**: Checked when closing a `(` group; if `)` is missing, a `ValueError` is raised.
    *   **Division by Zero**: Checked explicitly before performing division.
    *   **Invalid Tokens**: Caught if the tokenizer finds non-matching characters or if the parser encounters an unexpected token type.
