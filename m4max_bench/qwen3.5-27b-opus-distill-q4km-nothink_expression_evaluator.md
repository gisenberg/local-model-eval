# Qwen 3.5 27B Opus-Distilled Q4_K_M — expression_evaluator

## Output

Here is the complete solution containing the `ExpressionEvaluator` class and the corresponding pytest tests.

### Implementation Details

The solution uses a **Recursive Descent Parser**. This approach breaks the grammar down into functions that call each other, naturally handling operator precedence.

**The Grammar:**
1.  **Expression**: Handles addition and subtraction (lowest precedence).
2.  **Term**: Handles multiplication and division (higher precedence).
3.  **Factor**: Handles unary operators (like `-3`) and parentheses.
4.  **Number**: Parses integers and floats.

### Python Code

```python
import re
from typing import Iterator, List, Tuple, Union

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: A string containing the mathematical expression.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")

        # 1. Tokenization
        # Regex breakdown:
        # \d+\.?\d* : Matches integers (123) and floats (123.45, .45)
        # [+\-*/()]  : Matches operators and parentheses
        # \s+        : Matches whitespace (to be ignored)
        token_pattern = r'(\d+\.?\d*|[+\-*/()])'
        tokens = re.findall(token_pattern, expr)
        
        # Check for invalid characters (anything not matched by the regex)
        cleaned_expr = re.sub(token_pattern, '', expr)
        if cleaned_expr.strip():
            raise ValueError(f"Invalid character found in expression: '{cleaned_expr.strip()}'")

        self.tokens = tokens
        self.pos = 0

        # 2. Parsing
        if not self.tokens:
            raise ValueError("Expression is empty.")

        result = self._parse_expression()

        # Ensure all tokens were consumed
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at end of expression.")

        return result

    def _current_token(self) -> Union[str, None]:
        """Returns the current token or None if end of list."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self) -> str:
        """Consumes the current token and advances the position."""
        token = self._current_token()
        if token is None:
            raise ValueError("Unexpected end of expression.")
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Grammar: Expression -> Term { (+|-) Term }
        """
        value = self._parse_term()
        
        while True:
            token = self._current_token()
            if token == '+':
                self._consume()
                value += self._parse_term()
            elif token == '-':
                self._consume()
                value -= self._parse_term()
            else:
                break
        return value

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Grammar: Term -> Factor { (*|/) Factor }
        """
        value = self._parse_factor()
        
        while True:
            token = self._current_token()
            if token == '*':
                self._consume()
                right = self._parse_factor()
                value *= right
            elif token == '/':
                self._consume()
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero.")
                value /= right
            else:
                break
        return value

    def _parse_factor(self) -> float:
        """
        Parses unary operators and parentheses.
        Grammar: Factor -> ( '+' | '-' )? ( Number | '(' Expression ')' )
        """
        token = self._current_token()
        
        # Handle Unary Plus/Minus
        if token == '+':
            self._consume()
            return self._parse_factor()
        elif token == '-':
            self._consume()
            return -self._parse_factor()
        
        # Handle Parentheses
        if token == '(':
            self._consume()
            value = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._consume()
            return value
        
        # Handle Numbers
        if token and re.match(r'^\d+\.?\d*$', token):
            self._consume()
            return float(token)
        
        # If we reach here, it's an invalid token for a factor
        raise ValueError(f"Invalid token '{token}' encountered.")

# --- Testing ---
if __name__ == "__main__":
    evaluator = ExpressionEvaluator()
    test_cases = [
        "1 + 2",
        "10 * 2 + 5",
        "(1 + 2) * 3",
        "-5 + 10",
        "-(2 + 3) * 4",
        "3.14 * 2",
        "10 / 2 / 2"
    ]
    
    for case in test_cases:
        try:
            print(f"{case} = {evaluator.evaluate(case)}")
        except ValueError as e:
            print(f"{case} raised ValueError: {e}")
```

### Pytest Tests

Save the following code in a file named `test_evaluator.py` to run the tests.

```python
import pytest
from expression_evaluator import ExpressionEvaluator

class TestExpressionEvaluator:
    
    def setup_method(self):
        """Initialize a fresh evaluator for each test."""
        self.evaluator = ExpressionEvaluator()

    def test_basic_arithmetic(self):
        """Test basic addition, subtraction, multiplication, and division."""
        assert self.evaluator.evaluate("1 + 1") == 2.0
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
        assert self.evaluator.evaluate("(1 + 2) * 3") == 9.0
        assert self.evaluator.evaluate("10 / (2 + 3)") == 2.0
        assert self.evaluator.evaluate("((2 + 3) * 4)") == 20.0

    def test_unary_minus(self):
        """Test unary minus at start and inside expressions."""
        assert self.evaluator.evaluate("-5") == -5.0
        assert self.evaluator.evaluate("-3 + 2") == -1.0
        assert self.evaluator.evaluate("5 - -2") == 7.0
        assert self.evaluator.evaluate("-(2 + 3)") == -5.0
        assert self.evaluator.evaluate("-(-5)") == 5.0

    def test_error_cases(self):
        """Test that appropriate ValueErrors are raised."""
        # Empty expression
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate("")
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched"):
            self.evaluator.evaluate("(1 + 2")
            
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("10 / 0")
            
        # Invalid token
        with pytest.raises(ValueError, match="Invalid"):
            self.evaluator.evaluate("10 + a")
```

### How to Run

1.  Save the first block of code as `expression_evaluator.py`.
2.  Save the second block of code as `test_evaluator.py`.
3.  Run the tests using pytest:
    ```bash
    pytest test_evaluator.py -v
    ```
