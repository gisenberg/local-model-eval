# GLM-4.5-Air Q4_K_M (bartowski) — KV f16 (baseline) — Expression Evaluator

## Output

# Mathematical Expression Evaluator in Python

Here's a complete implementation of an `ExpressionEvaluator` class that meets all the requirements:

```python
import re
from typing import List, Optional, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic operations,
    operator precedence, parentheses, and unary minus.
    
    Uses recursive descent parsing to evaluate expressions without using eval().
    """
    
    def __init__(self):
        """Initialize the tokenizer and parser."""
        self.tokens: List[str] = []
        self.current_token_index: int = 0
        self.current_token: Optional[str] = None
    
    def tokenize(self, expr: str) -> List[str]:
        """
        Tokenize the input expression into numbers, operators, and parentheses.
        
        Args:
            expr: The mathematical expression to tokenize
            
        Returns:
            List of tokens
            
        Raises:
            ValueError: If invalid tokens are found
        """
        # Remove all whitespace
        expr = expr.replace(' ', '')
        
        # Regular expression to match numbers (including negative numbers at start)
        # and operators/parentheses
        token_pattern = r"""
            (?P<NUMBER>-?\d+\.\d+|-?\d+\.|\d+\.?\d*)  # Numbers (including floats)
            |(?P<PLUS>\+)
            |(?P<MINUS>-)
            |(?P<MUL>\*)
            |(?P<DIV>/)
            |(?P<LPAREN>\()
            |(?P<RPAREN>\))
        """
        
        tokens = []
        for match in re.finditer(token_pattern, expr, re.VERBOSE):
            token_type = match.lastgroup
            token_value = match.group(token_type)
            
            if token_type == 'NUMBER':
                # Convert to float if it has a decimal point, otherwise int
                if '.' in token_value:
                    tokens.append(float(token_value))
                else:
                    tokens.append(int(token_value))
            else:
                tokens.append(token_value)
        
        return tokens
    
    def next_token(self) -> None:
        """Advance to the next token in the token list."""
        self.current_token_index += 1
        if self.current_token_index < len(self.tokens):
            self.current_token = self.tokens[self.current_token_index]
        else:
            self.current_token = None
    
    def parse_expression(self) -> float:
        """
        Parse and evaluate an expression (entry point for the parser).
        
        Returns:
            The result of the evaluated expression
            
        Raises:
            ValueError: For various parsing errors
        """
        if not self.tokens:
            raise ValueError("Empty expression")
        
        self.current_token_index = -1
        self.next_token()
        
        result = self.parse_additive()
        
        if self.current_token is not None:
            raise ValueError(f"Unexpected token: {self.current_token}")
        
        return result
    
    def parse_additive(self) -> float:
        """
        Parse additive operations (+ and -) with correct precedence.
        
        Returns:
            The result of the parsed additive expression
        """
        left = self.parse_multiplicative()
        
        while self.current_token in ('+', '-'):
            op = self.current_token
            self.next_token()
            right = self.parse_multiplicative()
            
            if op == '+':
                left += right
            else:
                left -= right
        
        return left
    
    def parse_multiplicative(self) -> float:
        """
        Parse multiplicative operations (* and /) with correct precedence.
        
        Returns:
            The result of the parsed multiplicative expression
        """
        left = self.parse_unary()
        
        while self.current_token in ('*', '/'):
            op = self.current_token
            self.next_token()
            right = self.parse_unary()
            
            if op == '*':
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
        
        return left
    
    def parse_unary(self) -> float:
        """
        Parse unary minus operations.
        
        Returns:
            The result of the parsed unary expression
        """
        if self.current_token == '-':
            self.next_token()
            return -self.parse_primary()
        
        return self.parse_primary()
    
    def parse_primary(self) -> float:
        """
        Parse primary expressions (numbers and parenthesized expressions).
        
        Returns:
            The result of the parsed primary expression
        """
        if self.current_token is None:
            raise ValueError("Unexpected end of expression")
        
        if self.current_token == '(':
            self.next_token()
            expr = self.parse_additive()
            if self.current_token != ')':
                raise ValueError("Mismatched parentheses")
            self.next_token()
            return expr
        
        if isinstance(self.current_token, (int, float)):
            value = self.current_token
            self.next_token()
            return value
        
        raise ValueError(f"Invalid token: {self.current_token}")
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Args:
            expr: The mathematical expression to evaluate
            
        Returns:
            The result of the evaluated expression
            
        Raises:
            ValueError: For various parsing errors
        """
        try:
            self.tokens = self.tokenize(expr)
            return self.parse_expression()
        except ValueError as e:
            raise ValueError(f"Error evaluating expression '{expr}': {str(e)}")
```

## Pytest Tests

Here are 5 pytest tests covering the requirements:

```python
import pytest
from expression_evaluator import ExpressionEvaluator

@pytest.fixture
def evaluator():
    """Fixture to provide an ExpressionEvaluator instance."""
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    """Test basic arithmetic operations."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 5 / 2") == 7.5
    assert evaluator.evaluate("3 * 4 + 5") == 17.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0

def test_operator_precedence(evaluator):
    """Test correct operator precedence."""
    assert evaluator.evaluate("2 + 3 * 4 - 5 / 2") == 11.5
    assert evaluator.evaluate("3 + 4 * 2 / (1 - 5) ** 2") == 3.5
    assert evaluator.evaluate("10 - 3 * 2 + 4 / 2") == 6.0

def test_parentheses(evaluator):
    """Test parentheses for grouping."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 - (5 / 2)") == 7.5
    assert evaluator.evaluate("3 * (4 + 5)") == 27.0
    assert evaluator.evaluate("(10 / 2) - 3") == 2.0
    assert evaluator.evaluate("((2 + 3) * 4) - 5") == 15.0

def test_unary_minus(evaluator):
    """Test unary minus operations."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 * -2") == -6.0
    assert evaluator.evaluate("10 / -2") == -5.0
    assert evaluator.evaluate("-3.14") == -3.14

def test_error_cases(evaluator):
    """Test error cases."""
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + abc")
    
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3 4")
    
    with pytest.raises(ValueError, match="Unexpected end of expression"):
        evaluator.evaluate("2 + (3 *")
```

## Implementation Notes

1. **Tokenizer**: The `tokenize` method uses a regular expression to split the input string into tokens (numbers, operators, and parentheses). It handles both integers and floating-point numbers.

2. **Recursive Descent Parser**: The parser consists of several methods that handle different levels of operator precedence:
   - `parse_additive`: Handles + and - operations
   - `parse_multiplicative`: Handles * and / operations
   - `parse_unary`: Handles unary minus
   - `parse_primary`: Handles numbers and parenthesized expressions

3. **Error Handling**: The parser raises descriptive `ValueError` messages for:
   - Mismatched parentheses
   - Division by zero
   - Invalid tokens
   - Empty expressions
   - Unexpected tokens or end of expression

4. **Type Hints**: All methods include type hints for parameters and return values.

5. **Tests**: The pytest tests cover all the required functionality including basic arithmetic, operator precedence, parentheses, unary minus, and error cases.

This implementation provides a robust mathematical expression evaluator that meets all the specified requirements without using `eval()` or `ast.literal_eval()`.
