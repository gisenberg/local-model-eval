# Qwen3.5-9B (L40S vLLM) — Expression Evaluator — Run 3

## Output

Here is the complete implementation of the `ExpressionEvaluator` class using a recursive descent parser, followed by the required pytest tests.

### 1. Expression Evaluator Implementation

```python
import re
from typing import List, Tuple, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator supporting +, -, *, /, parentheses,
    and unary minus with correct operator precedence.
    """

    def __init__(self):
        # Tokenizer pattern: numbers (int or float), operators, parentheses
        self.token_pattern = re.compile(r'\s*(\d+\.?\d*|\.\d+|[+\-*/()])\s*')
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string.

        Args:
            expr: A string containing a valid mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, has mismatched parentheses,
                       contains invalid tokens, or results in division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")

        # Tokenize
        self.tokens = self.token_pattern.findall(expr)
        if not self.tokens:
            raise ValueError("Expression contains no valid tokens.")

        # Check for trailing/leading whitespace issues implicitly handled by regex,
        # but ensure we consume all tokens
        self.pos = 0
        result = self._parse_expression()

        # Ensure all tokens were consumed
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: {self.tokens[self.pos]}")

        return result

    def _parse_expression(self) -> float:
        """
        Parses an expression handling addition and subtraction (+, -).
        Lower precedence than multiplication/division.
        """
        left = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            
            if op == '+':
                left = left + right
            else:
                left = left - right
                
        return left

    def _parse_term(self) -> float:
        """
        Parses a term handling multiplication and division (*, /).
        Higher precedence than addition/subtraction.
        """
        left = self._parse_factor()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()
            
            if op == '*':
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero.")
                left = left / right
                
        return left

    def _parse_factor(self) -> float:
        """
        Parses a factor handling numbers, parentheses, and unary minus.
        Highest precedence.
        """
        # Handle Unary Minus
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1
            operand = self._parse_factor()
            return -operand

        # Handle Unary Plus (optional, but good for completeness)
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '+':
            self.pos += 1
            return self._parse_factor()

        # Handle Parentheses
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing parenthesis.")
            self.pos += 1
            return result

        # Handle Numbers
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            try:
                if '.' in token:
                    return float(token)
                else:
                    return float(token)
            except ValueError:
                raise ValueError(f"Invalid token: {token}")

        raise ValueError(f"Unexpected end of expression or invalid token at position {self.pos}")

```

### 2. Pytest Tests

```python
import pytest
from expression_evaluator import ExpressionEvaluator

class TestExpressionEvaluator:
    def setup_method(self):
        self.evaluator = ExpressionEvaluator()

    # Test 1: Basic Arithmetic
    def test_basic_arithmetic(self):
        """Tests simple addition, subtraction, multiplication, and division."""
        assert self.evaluator.evaluate("3 + 5") == 8.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("2 * 3") == 6.0
        assert self.evaluator.evaluate("10 / 2") == 5.0
        assert self.evaluator.evaluate("3.5 + 2.5") == 6.0

    # Test 2: Operator Precedence
    def test_operator_precedence(self):
        """Tests that multiplication/division happens before addition/subtraction."""
        # 2 + 3 * 4 should be 2 + 12 = 14
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
        # 10 - 2 * 3 should be 10 - 6 = 4
        assert self.evaluator.evaluate("10 - 2 * 3") == 4.0
        # 10 / 2 * 5 should be 5 * 5 = 25 (left associative)
        assert self.evaluator.evaluate("10 / 2 * 5") == 25.0

    # Test 3: Parentheses
    def test_parentheses(self):
        """Tests grouping with parentheses."""
        # (2 + 3) * 4 should be 5 * 4 = 20
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        # Nested: ((1 + 2) * 3) + 4
        assert self.evaluator.evaluate("((1 + 2) * 3) + 4") == 13.0
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("2 + 3)")

    # Test 4: Unary Minus
    def test_unary_minus(self):
        """Tests unary minus for negative numbers and negated expressions."""
        assert self.evaluator.evaluate("-5") == -5.0
        assert self.evaluator.evaluate("-3 + 2") == -1.0
        assert self.evaluator.evaluate("-(2 + 3)") == -5.0
        assert self.evaluator.evaluate("-(-5)") == 5.0
        assert self.evaluator.evaluate("3 * -2") == -6.0

    # Test 5: Error Cases
    def test_error_cases(self):
        """Tests handling of invalid inputs and edge cases."""
        # Empty expression
        with pytest.raises(ValueError, match="cannot be empty"):
            self.evaluator.evaluate("")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("10 / 0")
        
        # Invalid token (e.g., 'abc')
        with pytest.raises(ValueError, match="Invalid token"):
            self.evaluator.evaluate("3 + abc")
        
        # Unexpected token (trailing garbage)
        with pytest.raises(ValueError, match="Unexpected token"):
            self.evaluator.evaluate("3 + 5 *")
```

### How to Run

1.  Save the first code block as `expression_evaluator.py`.
2.  Save the second code block as `test_expression_evaluator.py`.
3.  Run the tests using pytest:
    ```bash
    pytest test_expression_evaluator.py -v
    ```

### Key Features of the Implementation

*   **Recursive Descent**: The parser is broken down into four levels of functions (`_parse_expression`, `_parse_term`, `_parse_factor`) corresponding to the grammar hierarchy (Addition < Multiplication < Factors).
*   **Precedence**: By calling `_parse_term` inside `_parse_expression` and `_parse_factor` inside `_parse_term`, we ensure higher precedence operators are evaluated first.
*   **Unary Minus**: Handled specifically in `_parse_factor` before attempting to parse a number or parenthesis, allowing constructs like `-5` and `-(2+3)`.
*   **Error Handling**: Explicit checks for empty strings, mismatched parentheses (via stack-like logic in the factor/parenthesis handling), division by zero, and unconsumed tokens ensure robust error reporting via `ValueError`.
*   **No `eval`**: The solution relies entirely on manual tokenization and parsing logic.
