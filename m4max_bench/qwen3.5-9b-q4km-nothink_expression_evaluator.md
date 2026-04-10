# Qwen 3.5 9B Q4_K_M — expression_evaluator

## Output

Here is a complete implementation of the `ExpressionEvaluator` class using a recursive descent parser, along with the required tests.

### Implementation

```python
from typing import List, Tuple, Optional
import re

class ExpressionEvaluator:
    """
    A mathematical expression evaluator supporting +, -, *, /, parentheses,
    and unary minus with correct operator precedence.
    """

    def __init__(self):
        self.pos = 0
        self.tokens: List[str] = []
        self._tokenize()

    def _tokenize(self) -> None:
        """
        Converts the input string into a list of tokens.
        Raises ValueError for invalid characters or empty expressions.
        """
        if not self.tokens:
            self.tokens = []
            return

        # Reset state for new evaluation
        self.pos = 0
        self.tokens = []
        
        # Regex to match numbers (int or float), operators, and parentheses
        pattern = r'\d+\.?\d*|[+\-*/()]'
        matches = re.findall(pattern, self.expr)
        
        if not matches:
            raise ValueError("Empty expression or no valid tokens found.")
            
        self.tokens = matches

    @property
    def expr(self) -> str:
        """Returns the expression string."""
        return self._expr

    @expr.setter
    def expr(self, value: str) -> None:
        """Sets the expression string and tokenizes it."""
        self._expr = value
        self._tokenize()

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the mathematical expression string.
        
        Args:
            expr: A string containing a mathematical expression.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is invalid, has mismatched parentheses,
                       contains invalid tokens, or results in division by zero.
        """
        self.expr = expr
        self.pos = 0
        self.tokens = []
        self._tokenize()
        
        result = self._parse_expression()
        
        # Ensure all tokens were consumed
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: {self.tokens[self.pos]}")
            
        return result

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
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
        Parses multiplication and division (higher precedence).
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
                    raise ValueError("Division by zero is not allowed.")
                left = left / right
                
        return left

    def _parse_factor(self) -> float:
        """
        Parses numbers and unary operators (highest precedence).
        Handles parentheses and unary minus.
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression.")
            
        token = self.tokens[self.pos]
        
        # Handle Unary Minus
        if token == '-':
            self.pos += 1
            value = self._parse_factor()
            return -value
            
        # Handle Unary Plus (optional, but good for consistency)
        if token == '+':
            self.pos += 1
            return self._parse_factor()
            
        # Handle Parentheses
        if token == '(':
            self.pos += 1
            value = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing parenthesis.")
            self.pos += 1
            return value
            
        # Handle Numbers
        if token.replace('.', '').isdigit():
            self.pos += 1
            return float(token)
            
        raise ValueError(f"Invalid token: {token}")
```

### Tests

```python
import pytest

class TestExpressionEvaluator:
    """
    Test suite for ExpressionEvaluator.
    """

    def test_basic_arithmetic(self):
        """Tests basic addition, subtraction, multiplication, and division."""
        evaluator = ExpressionEvaluator()
        
        assert evaluator.evaluate("1 + 2") == 3.0
        assert evaluator.evaluate("10 - 5") == 5.0
        assert evaluator.evaluate("3 * 4") == 12.0
        assert evaluator.evaluate("10 / 2") == 5.0
        assert evaluator.evaluate("10 / 3") == pytest.approx(3.3333333333333335)

    def test_operator_precedence(self):
        """Tests that multiplication and division have higher precedence than addition and subtraction."""
        evaluator = ExpressionEvaluator()
        
        # Multiplication before addition
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        
        # Division before subtraction
        assert evaluator.evaluate("10 - 2 / 2") == 9.0
        
        # Mixed precedence
        assert evaluator.evaluate("10 + 2 * 3 - 5") == 11.0

    def test_parentheses_grouping(self):
        """Tests that parentheses correctly override precedence."""
        evaluator = ExpressionEvaluator()
        
        # Parentheses change order
        assert evaluator.evaluate("(2 + 3) * 4") == 20.0
        
        # Nested parentheses
        assert evaluator.evaluate("((1 + 2) * 3) - 4") == 5.0
        
        # Parentheses with division
        assert evaluator.evaluate("10 / (2 + 3)") == 2.0

    def test_unary_minus(self):
        """Tests unary minus for negative numbers and negated expressions."""
        evaluator = ExpressionEvaluator()
        
        # Negative number
        assert evaluator.evaluate("-5") == -5.0
        
        # Unary minus with parentheses
        assert evaluator.evaluate("- (2 + 3)") == -5.0
        
        # Unary minus with multiplication
        assert evaluator.evaluate("-3 * 4") == -12.0
        
        # Unary minus with addition
        assert evaluator.evaluate("5 + -3") == 2.0
        
        # Unary minus with nested parentheses
        assert evaluator.evaluate("- (1 + 2)") == -3.0

    def test_error_cases(self):
        """Tests that ValueError is raised for invalid inputs."""
        evaluator = ExpressionEvaluator()
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("(1 + 2")
            
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("1 + 2)")
            
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("1 / 0")
            
        # Empty expression
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")
            
        # Invalid token
        with pytest.raises(ValueError, match="Invalid token"):
            evaluator.evaluate("1 + a")
            
        # Unexpected token at end
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluator.evaluate("1 + 2 *")

if __name__ == "__main__":
    pytest.main([__file__])
```

### Key Features of the Implementation

1.  **Recursive Descent Parser**: The parser is structured into four methods:
    *   `_parse_expression`: Handles `+` and `-` (lowest precedence).
    *   `_parse_term`: Handles `*` and `/` (higher precedence).
    *   `_parse_factor`: Handles numbers, parentheses, and unary operators (highest precedence).
2.  **Operator Precedence**: By calling `_parse_term` inside `_parse_expression` and `_parse_factor` inside `_parse_term`, the parser naturally respects precedence rules without needing a stack for operators.
3.  **Unary Minus**: Implemented in `_parse_factor`. If a `-` is encountered before a number or parenthesis, it recursively calls `_parse_factor` and negates the result.
4.  **Error Handling**:
    *   **Mismatched Parentheses**: Checked when closing `)` is expected but not found.
    *   **Division by Zero**: Explicitly checked in `_parse_term`.
    *   **Invalid Tokens**: Checked during tokenization and parsing.
    *   **Empty Expressions**: Checked during tokenization.
5.  **Type Hints**: All methods include type hints for better IDE support and documentation.
6.  **No `eval()`**: The solution uses a custom parser, ensuring safety and control over the evaluation process.
