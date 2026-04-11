# Qwen3.5-122B-A10B Q4_K_M (bartowski) [rotorquant iso3] — KV iso3 (3-bit sym, 4D quaternion) — Expression Evaluator

## Output

Here is a complete implementation of the `ExpressionEvaluator` using a recursive descent parser, followed by the required `pytest` tests.

### Implementation

The parser follows the standard grammar for arithmetic expressions:
1.  **Expression**: Handles addition and subtraction (lowest precedence).
2.  **Term**: Handles multiplication and division.
3.  **Factor**: Handles numbers, parentheses, and unary operators (highest precedence).

```python
import re
from typing import List, Optional, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers
    
    Raises ValueError for invalid syntax, mismatched parentheses, or division by zero.
    """

    def __init__(self):
        self.tokens: List[Tuple[str, str]] = []
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
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at end of expression: {self.tokens[self.pos][1]}")
            
        return result

    def _tokenize(self, expr: str) -> None:
        """
        Converts the input string into a list of tokens.
        Tokens are tuples of (type, value).
        Types: 'NUMBER', 'PLUS', 'MINUS', 'MUL', 'DIV', 'LPAREN', 'RPAREN'
        """
        token_pattern = r'\s*(?:(\d+\.?\d*)|([+\-*/()]))'
        matches = re.findall(token_pattern, expr)
        
        self.tokens = []
        for num, op in matches:
            if num:
                self.tokens.append(('NUMBER', num))
            elif op:
                if op == '+':
                    self.tokens.append(('PLUS', op))
                elif op == '-':
                    self.tokens.append(('MINUS', op))
                elif op == '*':
                    self.tokens.append(('MUL', op))
                elif op == '/':
                    self.tokens.append(('DIV', op))
                elif op == '(':
                    self.tokens.append(('LPAREN', op))
                elif op == ')':
                    self.tokens.append(('RPAREN', op))
        
        # Check for invalid characters (those not matched by regex)
        # We reconstruct the string from tokens to see if anything was skipped
        reconstructed = "".join(t[1] for t in self.tokens)
        if reconstructed != "".join(expr.split()):
            # Find the first character that doesn't match
            clean_expr = "".join(expr.split())
            for i, char in enumerate(clean_expr):
                if char not in "0123456789.+-*/()":
                    raise ValueError(f"Invalid token: '{char}'")

    def _current_token(self) -> Optional[Tuple[str, str]]:
        """Returns the current token or None if at the end."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self, expected_type: Optional[str] = None) -> Tuple[str, str]:
        """
        Consumes the current token. If expected_type is provided, validates it.
        Raises ValueError if the token doesn't match or if end of input is reached.
        """
        token = self._current_token()
        if token is None:
            raise ValueError("Unexpected end of expression")
        
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[0]}")
            
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses an expression: Term { ('+' | '-') Term }
        Handles addition and subtraction (lowest precedence).
        """
        left = self._parse_term()
        
        while True:
            token = self._current_token()
            if token and token[0] in ('PLUS', 'MINUS'):
                op = token[0]
                self._consume()
                right = self._parse_term()
                if op == 'PLUS':
                    left += right
                else:
                    left -= right
            else:
                break
                
        return left

    def _parse_term(self) -> float:
        """
        Parses a term: Factor { ('*' | '/') Factor }
        Handles multiplication and division.
        """
        left = self._parse_factor()
        
        while True:
            token = self._current_token()
            if token and token[0] in ('MUL', 'DIV'):
                op = token[0]
                self._consume()
                right = self._parse_factor()
                if op == 'MUL':
                    left *= right
                else:
                    if right == 0:
                        raise ValueError("Division by zero")
                    left /= right
            else:
                break
                
        return left

    def _parse_factor(self) -> float:
        """
        Parses a factor: NUMBER | '(' Expression ')' | ('+' | '-') Factor
        Handles numbers, parentheses, and unary operators.
        """
        token = self._current_token()
        
        if token is None:
            raise ValueError("Unexpected end of expression")

        # Handle unary plus or minus
        if token[0] in ('PLUS', 'MINUS'):
            self._consume()
            val = self._parse_factor()
            if token[0] == 'MINUS':
                return -val
            return val

        # Handle numbers
        if token[0] == 'NUMBER':
            self._consume()
            return float(token[1])

        # Handle parentheses
        if token[0] == 'LPAREN':
            self._consume()
            val = self._parse_expression()
            if self._current_token() is None or self._current_token()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._consume() # Consume ')'
            return val

        raise ValueError(f"Invalid token: {token[1]}")
```

### Pytest Tests

Save the following code in a file named `test_expression_evaluator.py`. Ensure `pytest` is installed (`pip install pytest`).

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
    assert evaluator.evaluate("2 + 3 * 4") == 14.0 # Precedence check

def test_operator_precedence():
    """Tests that multiplication/division happens before addition/subtraction."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("1 + 2 * 3 - 4 / 2") == 5.0 # 1 + 6 - 2 = 5
    assert evaluator.evaluate("10 / 2 + 3 * 2") == 11.0 # 5 + 6 = 11

def test_parentheses():
    """Tests grouping with parentheses."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == 15.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    """Tests unary minus support."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-2 + 5") == 3.0
    assert evaluator.evaluate("3 * -4") == -12.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-(3 * 4)") == -12.0
    assert evaluator.evaluate("--5") == 5.0 # Double negative
    assert evaluator.evaluate("1 - -2") == 3.0

def test_error_cases():
    """Tests various error conditions."""
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / (2 - 2)")

    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")
    
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")

    # Floating point support
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("10 / 3") == pytest.approx(3.3333333333333335)
```

### How to Run

1.  Save the parser code as `expression_evaluator.py`.
2.  Save the test code as `test_expression_evaluator.py`.
3.  Run the tests using:
    ```bash
    pytest test_expression_evaluator.py -v
    ```

### Key Features of the Implementation

1.  **Recursive Descent**: The logic is split into `_parse_expression`, `_parse_term`, and `_parse_factor`, strictly following operator precedence rules without using `eval()`.
2.  **Tokenization**: Uses regex to split the string into numbers and operators, handling whitespace automatically.
3.  **Unary Handling**: The `_parse_factor` method checks for `+` or `-` at the start of a factor, allowing for expressions like `-3` or `-(2+1)`.
4.  **Error Handling**:
    *   **Empty**: Checks for empty strings before processing.
    *   **Syntax**: Validates token types and ensures parentheses are balanced.
    *   **Runtime**: Checks for division by zero immediately before performing the operation.
    *   **Invalid**: Detects characters not in the allowed set.
5.  **Type Hints**: All methods and return values are annotated.
