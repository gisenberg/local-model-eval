# Qwen3.5-122B-A10B Q4_K_M (bartowski) — KV f16K/q8V (asymmetric) — Expression Evaluator

## Output

Here is a complete implementation of the `ExpressionEvaluator` using a recursive descent parser, followed by the required test suite.

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
            expr: A string containing the mathematical expression.
            
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
        # context in the parser.
        token_pattern = r'\s*(?:(\d+\.?\d*)|([+\-*/()]))\s*'
        matches = re.findall(token_pattern, expr)
        
        if not matches:
            # Check if there are non-whitespace characters that didn't match
            if re.search(r'\S', expr):
                raise ValueError(f"Invalid token in expression: {expr}")
            return

        for number, operator in matches:
            if number:
                self.tokens.append(float(number))
            elif operator:
                self.tokens.append(operator)
            else:
                # This case should theoretically not happen with the regex above
                # unless there's a logic gap, but good for safety.
                pass

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
        
        if token == '-':
            self._consume()
            return -self._parse_factor()
        elif token == '+':
            self._consume()
            return self._parse_factor()
        elif token == '(':
            self._consume()
            result = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses: expected ')'")
            self._consume()
            return result
        elif isinstance(token, float):
            self._consume()
            return token
        else:
            if token is None:
                raise ValueError("Unexpected end of expression")
            raise ValueError(f"Invalid token: {token}")

```

### Test Suite: `test_expression_evaluator.py`

```python
import pytest
from expression_evaluator import ExpressionEvaluator

evaluator = ExpressionEvaluator()

def test_basic_arithmetic():
    """Tests basic addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("8 / 2") == 4.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # Precedence check
    assert evaluator.evaluate("10 / 2 + 3") == 8.0

def test_operator_precedence():
    """Tests that multiplication/division happens before addition/subtraction."""
    assert evaluator.evaluate("2 + 3 * 4 - 1") == 13.0  # 2 + 12 - 1
    assert evaluator.evaluate("10 - 2 * 3 + 4") == 8.0  # 10 - 6 + 4
    assert evaluator.evaluate("1 + 2 * 3 * 4") == 25.0  # 1 + 24
    assert evaluator.evaluate("100 / 2 / 5") == 10.0    # Left-to-right associativity

def test_parentheses():
    """Tests grouping with parentheses."""
    assert evaluator.evaluate("(1 + 2) * 3") == 9.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    assert evaluator.evaluate("(1 + 2) * (3 + 4)") == 21.0

def test_unary_minus():
    """Tests unary minus operator support."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-2 + 5") == 3.0
    assert evaluator.evaluate("3 * -4") == -12.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-(3 * -2)") == 6.0
    assert evaluator.evaluate("1 + -2") == -1.0

def test_error_cases():
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
        evaluator.evaluate("1 + 2 *")

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
    *   `_parse_factor`: Handles numbers, parentheses, and unary operators (`+`, `-`).
    This structure naturally enforces operator precedence without needing an explicit precedence table.

2.  **Unary Minus Handling**: The `_parse_factor` method checks if the current token is `-`. If so, it consumes it and recursively calls `_parse_factor`. This allows for chains like `--5` or `-(2+1)` and correctly distinguishes unary minus from binary subtraction (which is handled in `_parse_expression`).

3.  **Tokenization**: A regular expression is used to split the string into numbers and operators. This handles floating-point numbers (e.g., `3.14`) and ignores whitespace.

4.  **Error Handling**:
    *   **Empty Input**: Checked at the start.
    *   **Invalid Tokens**: Caught if the regex doesn't match a character or if an unexpected token appears during parsing.
    *   **Mismatched Parentheses**: Checked after parsing a sub-expression inside `(` to ensure a matching `)` exists.
    *   **Division by Zero**: Checked immediately before performing the division operation.
