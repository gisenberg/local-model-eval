# MiniMax-M2.5 UD-IQ2_XXS (empty-think, 160K ctx) — Expression Evaluator

## Output (11638 chars)

The user wants a mathematical expression evaluator in Python with specific requirements. Let's break down the requirements:

1. Support +, -, *, / with correct operator precedence
2. Support parentheses for grouping
3. Support unary minus (e.g., "-3", "-(2+1)")
4. Support floating point numbers (e.g., "3.14")
5. Raise ValueError with descriptive messages for: mismatched parentheses, division by zero, invalid tokens, empty expressions
6. Implement as a class called ExpressionEvaluator with an evaluate(expr: str) -> float method
7. Use a recursive descent parser — do NOT use eval() or ast.literal_eval()
8. Include type hints throughout and a brief docstring on each method
9. Write 5 pytest tests covering: basic arithmetic, precedence, parentheses, unary minus, and error cases

This is a classic recursive descent parser problem. The typical approach is:
- Tokenize the input (lexer)
- Parse using recursive descent (parser)
- Evaluate the parsed expression

For operator precedence, we can use either:
- Multiple levels of parsing functions (term, factor, etc.)
- Or use precedence climbing method

I'll implement a recursive descent parser with proper operator precedence:
- Addition/Subtraction (lowest precedence)
- Multiplication/Division (higher precedence)
- Unary minus (highest precedence)
- Parentheses (highest, but handled specially)

Let me write the complete solution:

```python
"""
Mathematical Expression Evaluator

A recursive descent parser for evaluating mathematical expressions
supporting +, -, *, /, parentheses, unary minus, and floating point numbers.
"""

from __future__ import annotations

import re
from typing import List


class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    
    Supports: +, -, *, /, parentheses, unary minus, floating point numbers.
    
    Raises ValueError for: mismatched parentheses, division by zero,
    invalid tokens, empty expressions.
    """
    
    def __init__(self) -> None:
        """Initialize the evaluator."""
        self.tokens: List[str] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Args:
            expr: The expression string to evaluate.
            
        Returns:
            The result of the expression as a float.
            
        Raises:
            ValueError: For invalid expressions, mismatched parentheses,
                       division by zero, or invalid tokens.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        # Tokenize the input
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Empty expression")
        
        # Parse and evaluate
        result = self._parse_expression()
        
        # Check for unexpected tokens (leftover)
        if self.pos < len(self.tokens):
            raise ValueError(f"Invalid expression: unexpected token '{self.tokens[self.pos]}'")
        
        return result
    
    def _tokenize(self, expr: str) -> List[str]:
        """
        Tokenize the expression string into a list of tokens.
        
        Args:
            expr: The expression string.
            
        Returns:
            List of tokens.
            
        Raises:
            ValueError: For invalid tokens.
        """
        # Remove whitespace
        expr = expr.strip()
        
        tokens = []
        i = 0
        while i < len(expr):
            char = expr[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
            
            # Operators and parentheses
            if char in '+-*/()':
                tokens.append(char)
                i += 1
                continue
            
            # Number (integer or float)
            if char.isdigit() or (char == '.' and i + 1 < len(expr) and expr[i + 1].isdigit()):
                # Parse the number
                start = i
                has_dot = False
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number: multiple decimal points at position {i}")
                        has_dot = True
                    i += 1
                tokens.append(expr[start:i])
                continue
            
            # Invalid character
            raise ValueError(f"Invalid token: '{char}' at position {i}")
        
        return tokens
    
    def _parse_expression(self) -> float:
        """Parse and evaluate an expression (addition/subtraction)."""
        return self._parse_add_sub()
    
    def _parse_add_sub(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_mul_div()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_mul_div()
            
            if op == '+':
                left += right
            else:
                left -= right
        
        return left
    
    def _parse_mul_div(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        left = self._parse_unary()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_unary()
            
            if op == '*':
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
        
        return left
    
    def _parse_unary(self) -> float:
        """Parse unary operators (unary minus/plus)."""
        if self.pos < len(self.tokens):
            if self.tokens[self.pos] == '-':
                self.pos += 1
                operand = self._parse_unary()
                return -operand
            elif self.tokens[self.pos] == '+':
                self.pos += 1
                return self._parse_unary()
        
        return self._parse_primary()
    
    def _parse_primary(self) -> float:
        """Parse primary values (numbers and parenthesized expressions)."""
        if self.pos >= len(self.tokens):
            raise ValueError("Incomplete expression")
        
        token = self.tokens[self.pos]
        
        # Number
        if self._is_number(token):
            self.pos += 1
            return float(token)
        
        # Parentheses
        if token == '(':
            self.pos += 1
            result = self._parse_add_sub()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            
            self.pos += 1
            return result
        
        # Unexpected token
        raise ValueError(f"Invalid token: '{token}'")
    
    def _is_number(self, token: str) -> bool:
        """Check if a token is a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False


# Pytest tests
import pytest


def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    
    # Addition
    assert evaluator.evaluate("2 + 3") == 5.0
    
    # Subtraction
    assert evaluator.evaluate("5 - 3") == 2.0
    
    # Multiplication
    assert evaluator.evaluate("2 * 3") == 6.0
    
    # Division
    assert evaluator.evaluate("6 / 3") == 2.0
    
    # Multiple operations
    assert evaluator.evaluate("2 + 3 * 4") == 14.0


def test_precedence():
    """Test operator precedence."""
    evaluator = ExpressionEvaluator()
    
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    
    # Division before subtraction
    assert evaluator.evaluate("10 - 4 / 2") == 8.0
    
    # Parentheses override precedence
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    
    # Complex precedence
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0


def test_parentheses():
    """Test parentheses for grouping."""
    evaluator = ExpressionEvaluator()
    
    # Simple parentheses
    assert evaluator.evaluate("(2 + 3)") == 5.0
    
    # Nested parentheses
    assert evaluator.evaluate("((2 + 3) * (4 + 1))") == 25.0
    
    # Parentheses with precedence
    assert evaluator.evaluate("(2 + 3) * (4 - 2)") == 10.0
    
    # Multiple nested levels
    assert evaluator.evaluate("(((1 + 2)))") == 3.0


def test_unary_minus():
    """Test unary minus support."""
    evaluator = ExpressionEvaluator()
    
    # Simple unary minus
    assert evaluator.evaluate("-3") == -3.0
    
    # Unary minus with addition
    assert evaluator.evaluate("-3 + 5") == 2.0
    
    # Unary minus with multiplication
    assert evaluator.evaluate("-3 * 4") == -12.0
    
    # Unary minus with parentheses
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    
    # Unary minus in complex expression
    assert evaluator.evaluate("5 + -3 * 2") == -1.0


def test_error_cases():
    """Test error handling."""
    evaluator = ExpressionEvaluator()
    
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    
    # Whitespace only
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
    
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 ^ 3")
    
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("a + b")


if __name__ == "__main__":
    # Quick demo
    evaluator = ExpressionEvaluator()
    
    print("Expression Evaluator Demo")
    print("-" * 30)
    
    test_cases = [
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "-3 + 5",
        "-(2 + 1)",
        "3.14 * 2",
        "((1 + 2) * 3) - 4 / 2",
    ]
    
    for expr in test_cases:
        result = evaluator.evaluate(expr)
        print(f"{expr:30} = {result}")
```

Now let me verify this works correctly by running through the key test cases:

1. Basic arithmetic: `2 + 3 * 4` should be 14 (multiplication first)
2. Precedence: `(2 + 3) * 4` should be 20
3. Parentheses: `((2 + 3) * (4 + 1))` should be 25
4. Unary minus: `-3 + 5` should be 2, `-(2 + 1)` should be -3
5. Error cases: empty, mismatched parentheses, division by zero, invalid tokens

The solution implements a proper recursive descent parser with:
- Tokenizer that handles numbers, operators, and parentheses
- Parser with proper precedence:
  - `_parse_add_sub` handles + and -
  - `_parse_mul_div` handles * and /
  - `_parse_unary` handles unary minus/plus
  - `_parse_primary` handles numbers and parenthesized expressions

This follows the standard precedence climbing approach where higher precedence operations are parsed first, and the parser naturally handles operator precedence correctly.
