# Qwen 3.5 27B Opus-Distilled Q4_K_M (planar3/planar3 @ 65K) — expression_evaluator

**Hypothesis:** H2 probe: can we reach 64K context on Qwen 27B with rotorquant where f16 OOMs?

## Output

Here is the complete solution containing the `ExpressionEvaluator` class and the corresponding pytest tests.

### Implementation Details

The solution uses a **Recursive Descent Parser**. This approach breaks down the grammar into functions that call each other based on operator precedence:
1.  **Expression**: Handles addition and subtraction (lowest precedence).
2.  **Term**: Handles multiplication and division.
3.  **Factor**: Handles unary operators (negation) and parentheses.
4.  **Number**: Parses integers and floating-point numbers.

### Python Code (`evaluator.py`)

```python
import re
from typing import Iterator, List, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        # Regex to match numbers (integers or floats)
        self.number_pattern = re.compile(r'^-?\d+(\.\d+)?')
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the mathematical expression string and returns the result as a float.
        
        Args:
            expr: A string containing the mathematical expression.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty or whitespace only.")

        # Tokenize: split by spaces, but keep operators and parentheses attached to numbers if needed
        # We use a regex to split while keeping delimiters, then filter out empty strings
        raw_tokens = re.split(r'(\s+|[\+\-\*\/\(\)])', expr)
        self.tokens = [t for t in raw_tokens if t.strip()]
        self.pos = 0

        if not self.tokens:
            raise ValueError("Expression cannot be empty.")

        # Parse the expression
        result = self._parse_expression()

        # Ensure all tokens were consumed
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at end of expression.")

        return result

    def _current_token(self) -> str:
        """Returns the current token or None if end of input."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _advance(self) -> str:
        """Advances the position and returns the current token."""
        token = self._current_token()
        if token is not None:
            self.pos += 1
            return token
        raise ValueError("Unexpected end of expression")

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Grammar: Expression -> Term (('+' | '-') Term)*
        """
        value = self._parse_term()

        while True:
            token = self._current_token()
            if token == '+':
                self._advance()
                value += self._parse_term()
            elif token == '-':
                self._advance()
                value -= self._parse_term()
            else:
                break
        return value

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Grammar: Term -> Factor (('*' | '/') Factor)*
        """
        value = self._parse_factor()

        while True:
            token = self._current_token()
            if token == '*':
                self._advance()
                right = self._parse_factor()
                value *= right
            elif token == '/':
                self._advance()
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero is not allowed.")
                value /= right
            else:
                break
        return value

    def _parse_factor(self) -> float:
        """
        Parses unary operators and parentheses.
        Grammar: Factor -> ('+' | '-') Factor | '(' Expression ')' | Number
        """
        token = self._current_token()

        # Handle Unary Plus/Minus
        if token == '-':
            self._advance()
            return -self._parse_factor()
        elif token == '+':
            self._advance()
            return self._parse_factor()

        # Handle Parentheses
        if token == '(':
            self._advance() # consume '('
            value = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses: expected closing ')'")
            self._advance() # consume ')'
            return value

        # Handle Numbers
        if token and self._is_number(token):
            self._advance()
            return float(token)

        # If we reach here, the token is invalid
        raise ValueError(f"Invalid token: '{token}'")

    def _is_number(self, token: str) -> bool:
        """Checks if a token represents a valid number."""
        try:
            float(token)
            return True
        except ValueError:
            return False

# --- Testing Section ---
if __name__ == "__main__":
    evaluator = ExpressionEvaluator()
    test_cases = [
        "2 + 2",
        "10 * 5 + 2",
        "(2 + 3) * 4",
        "-3 + 5",
        "-(2 + 1)",
        "3.14 * 2",
        "10 / 2",
    ]
    
    for case in test_cases:
        try:
            print(f"{case} = {evaluator.evaluate(case)}")
        except ValueError as e:
            print(f"{case} raised ValueError: {e}")
```

### Pytest Tests (`test_evaluator.py`)

```python
import pytest
from evaluator import ExpressionEvaluator

class TestExpressionEvaluator:
    def setup_method(self):
        """Initialize a fresh evaluator for each test."""
        self.evaluator = ExpressionEvaluator()

    def test_basic_arithmetic(self):
        """Test basic addition, subtraction, multiplication, and division."""
        assert self.evaluator.evaluate("2 + 2") == 4.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("3 * 4") == 12.0
        assert self.evaluator.evaluate("15 / 3") == 5.0
        assert self.evaluator.evaluate("2.5 + 1.5") == 4.0

    def test_operator_precedence(self):
        """Test that multiplication/division happens before addition/subtraction."""
        # 2 + 3 * 4 should be 2 + 12 = 14
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
        # 10 - 2 * 3 should be 10 - 6 = 4
        assert self.evaluator.evaluate("10 - 2 * 3") == 4.0
        # 2 * 3 + 4 * 5 should be 6 + 20 = 26
        assert self.evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

    def test_parentheses(self):
        """Test grouping with parentheses."""
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert self.evaluator.evaluate("10 / (2 + 3)") == 2.0
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20.0
        assert self.evaluator.evaluate("1 + (2 * (3 + 4))") == 15.0

    def test_unary_minus(self):
        """Test unary minus at start and inside parentheses."""
        assert self.evaluator.evaluate("-3") == -3.0
        assert self.evaluator.evaluate("-3 + 5") == 2.0
        assert self.evaluator.evaluate("5 + -3") == 2.0
        assert self.evaluator.evaluate("-(2 + 1)") == -3.0
        assert self.evaluator.evaluate("-(-5)") == 5.0
        assert self.evaluator.evaluate("2 * -3") == -6.0

    def test_error_cases(self):
        """Test that appropriate ValueErrors are raised."""
        # Empty expression
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate("")
        
        # Whitespace only
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate("   ")

        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched"):
            self.evaluator.evaluate("(2 + 3")
        
        with pytest.raises(ValueError, match="Mismatched"):
            self.evaluator.evaluate("2 + 3)")

        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("10 / 0")

        # Invalid token
        with pytest.raises(ValueError, match="Invalid token"):
            self.evaluator.evaluate("2 + a")
        
        with pytest.raises(ValueError, match="Invalid token"):
            self.evaluator.evaluate("2 @ 2")
```

### How to Run

1.  Save the first block as `evaluator.py`.
2.  Save the second block as `test_evaluator.py`.
3.  Run the tests using pytest:
    ```bash
    pytest test_evaluator.py -v
    ```
