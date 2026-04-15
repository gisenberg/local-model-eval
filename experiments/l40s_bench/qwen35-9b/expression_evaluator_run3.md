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
    unary minus, and floating-point numbers with correct operator precedence.
    """

    def __init__(self):
        # Tokenizer pattern: numbers (int or float), operators, parentheses
        self.token_pattern = re.compile(
            r'\s*(?P<NUMBER>-?\d+(?:\.\d+)?)'
            r'|(?P<PLUS>\+)'
            r'|(?P<MINUS>-)'
            r'|(?P<MULT>\*)'
            r'|(?P<DIV>/)'
            r'|(?P<LPAREN>\()'
            r'|(?P<RPAREN>\))'
        )

    def tokenize(self, expr: str) -> List[Tuple[str, str]]:
        """
        Convert the input string into a list of tokens.
        
        Args:
            expr: The mathematical expression string.
            
        Returns:
            A list of tuples containing (type, value) for each token.
            
        Raises:
            ValueError: If the expression contains invalid characters.
        """
        tokens = []
        last_end = 0
        for match in self.token_pattern.finditer(expr):
            start, end = match.span()
            token_type = match.lastgroup
            token_value = match.group()

            if token_type is None:
                # This shouldn't happen due to the regex, but safety check
                continue
                
            if token_value != ' ':
                if token_type == 'NUMBER':
                    tokens.append(('NUMBER', token_value))
                else:
                    tokens.append((token_type, token_value))
            
            # Check for invalid characters between matches
            if start != last_end:
                invalid_char = expr[last_end:start]
                if invalid_char.strip():
                    raise ValueError(f"Invalid character found: '{invalid_char}'")
            last_end = end

        # Check for trailing whitespace or incomplete tokens
        if last_end != len(expr):
            raise ValueError(f"Invalid character at end of expression: '{expr[last_end:]}'")

        return tokens

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: The expression string to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: For mismatched parentheses, division by zero, 
                       invalid tokens, or empty expressions.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        tokens = self.tokenize(expr)
        if not tokens:
            raise ValueError("Expression cannot be empty")

        # Check for balanced parentheses first (optional optimization, 
        # but the parser logic handles it via stack/expectation)
        
        pos = [0]  # Use list to allow modification in nested functions
        val_stack = []
        op_stack = []

        # We will use a standard recursive descent approach via helper methods
        # to handle precedence naturally.
        
        try:
            result = self._parse_expression(tokens, pos)
            if pos[0] != len(tokens):
                raise ValueError(f"Unexpected token after expression: {tokens[pos[0]]}")
            return result
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except IndexError:
            # This can happen if parentheses are mismatched during parsing
            raise ValueError("Mismatched parentheses")

    def _parse_expression(self, tokens: List[Tuple[str, str]], pos: List[int]) -> float:
        """
        Parse an addition/subtraction expression (lowest precedence).
        Handles: term (+ term | - term)
        """
        left = self._parse_term(tokens, pos)
        
        while pos[0] < len(tokens) and tokens[pos[0]][0] in ('PLUS', 'MINUS'):
            op = tokens[pos[0]][0]
            pos[0] += 1
            right = self._parse_term(tokens, pos)
            
            if op == 'PLUS':
                left += right
            else:
                left -= right
                
        return left

    def _parse_term(self, tokens: List[Tuple[str, str]], pos: List[int]) -> float:
        """
        Parse a multiplication/division expression (higher precedence).
        Handles: factor (* factor | / factor)
        """
        left = self._parse_factor(tokens, pos)
        
        while pos[0] < len(tokens) and tokens[pos[0]][0] in ('MULT', 'DIV'):
            op = tokens[pos[0]][0]
            pos[0] += 1
            right = self._parse_factor(tokens, pos)
            
            if op == 'MULT':
                left *= right
            else:
                if right == 0:
                    raise ZeroDivisionError()
                left /= right
                
        return left

    def _parse_factor(self, tokens: List[Tuple[str, str]], pos: List[int]) -> float:
        """
        Parse a primary factor (numbers, parenthesized expressions, or unary minus).
        Handles: NUMBER | LPAREN ... RPAREN | - factor
        """
        if pos[0] >= len(tokens):
            raise ValueError("Unexpected end of expression")

        token_type, token_value = tokens[pos[0]]

        if token_type == 'NUMBER':
            pos[0] += 1
            return float(token_value)
        
        elif token_type == 'MINUS':
            # Unary minus
            pos[0] += 1
            value = self._parse_factor(tokens, pos)
            return -value
            
        elif token_type == 'PLUS':
            # Unary plus (optional, but good for consistency)
            pos[0] += 1
            return self._parse_factor(tokens, pos)
            
        elif token_type == 'LPAREN':
            pos[0] += 1
            value = self._parse_expression(tokens, pos)
            
            if pos[0] >= len(tokens) or tokens[pos[0]][0] != 'RPAREN':
                raise ValueError("Mismatched parentheses: missing closing parenthesis")
            pos[0] += 1
            return value
            
        else:
            raise ValueError(f"Invalid token: {token_value}")
```

### 2. Pytest Tests

```python
import pytest
from expression_evaluator import ExpressionEvaluator

class TestExpressionEvaluator:
    def setup_method(self):
        self.evaluator = ExpressionEvaluator()

    def test_basic_arithmetic(self):
        """Test basic addition, subtraction, multiplication, and division."""
        assert self.evaluator.evaluate("1 + 2") == 3.0
        assert self.evaluator.evaluate("10 - 4") == 6.0
        assert self.evaluator.evaluate("3 * 4") == 12.0
        assert self.evaluator.evaluate("10 / 2") == 5.0
        assert self.evaluator.evaluate("5 + 5 * 2") == 15.0  # Precedence check
        assert self.evaluator.evaluate("10 - 2 * 3") == 4.0

    def test_parentheses_grouping(self):
        """Test correct handling of parentheses for grouping."""
        assert self.evaluator.evaluate("(1 + 2) * 3") == 9.0
        assert self.evaluator.evaluate("10 / (2 + 3)") == 2.0
        assert self.evaluator.evaluate("((2 + 3) * 4) - 5") == 15.0
        assert self.evaluator.evaluate("2 * (3 + 4) * (5 - 1)") == 40.0

    def test_unary_minus(self):
        """Test support for unary minus on numbers and expressions."""
        assert self.evaluator.evaluate("-5 + 3") == -2.0
        assert self.evaluator.evaluate("3 - -5") == 8.0
        assert self.evaluator.evaluate("-(-5)") == 5.0
        assert self.evaluator.evaluate("- (2 + 3)") == -5.0
        assert self.evaluator.evaluate("-(10 / 2)") == -5.0

    def test_floating_point_numbers(self):
        """Test support for floating point literals."""
        assert self.evaluator.evaluate("3.14 + 2.86") == 6.0
        assert self.evaluator.evaluate("1.5 * 2") == 3.0
        assert self.evaluator.evaluate("10 / 3.0") == 3.3333333333333335
        assert self.evaluator.evaluate("-3.5") == -3.5

    def test_error_cases(self):
        """Test ValueError and ZeroDivisionError for invalid inputs."""
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("(1 + 2")
        
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            self.evaluator.evaluate("1 + 2)")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("1 / 0")
        
        with pytest.raises(ValueError, match="Division by zero"):
            self.evaluator.evaluate("5 / (2 + 0)")
            
        # Invalid tokens
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("1 + a")
            
        with pytest.raises(ValueError, match="Invalid character"):
            self.evaluator.evaluate("1 @ 2")
            
        # Empty expression
        with pytest.raises(ValueError, match="cannot be empty"):
            self.evaluator.evaluate("")
            
        with pytest.raises(ValueError, match="cannot be empty"):
            self.evaluator.evaluate("   ")
```

### How to Run

1.  Save the first code block as `expression_evaluator.py`.
2.  Save the second code block as `test_expression_evaluator.py`.
3.  Run the tests using pytest:
    ```bash
    pytest test_expression_evaluator.py -v
    ```

### Key Features of the Implementation

*   **Recursive Descent**: The parser is structured into four levels (`_parse_expression`, `_parse_term`, `_parse_factor`) corresponding to the precedence hierarchy (Add/Sub < Mul/Div < Primary).
*   **Tokenization**: Uses a regular expression to robustly separate numbers, operators, and parentheses while ignoring whitespace.
*   **Unary Minus**: Handled specifically in `_parse_factor` by consuming a `MINUS` token and negating the result of the subsequent factor.
*   **Error Handling**:
    *   Checks for empty strings immediately.
    *   Validates characters during tokenization.
    *   Checks for remaining tokens after parsing to ensure no trailing garbage.
    *   Catches `ZeroDivisionError` inside the parser and re-raises it as a descriptive `ValueError`.
    *   Checks parenthesis balance implicitly by expecting `RPAREN` after evaluating a parenthesized group.
*   **Type Safety**: Uses Python 3.9+ style type hints (`-> float`, `List[Tuple[str, str]]`) and docstrings for all public and private methods.
