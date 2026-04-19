# Mathematical Expression Evaluator

## Architecture

I'll implement a **recursive descent parser** following this grammar:

```
expression → term (('+' | '-') term)*
term       → factor (('*' | '/') factor)*
factor     → ('+' | '-') factor | '(' expression ')' | number
```

This naturally gives us correct operator precedence without needing shunting-yard or other algorithms.

## Implementation

```python
"""
Mathematical expression evaluator using recursive descent parsing.

Supports +, -, *, / operators with correct precedence, parentheses,
unary minus, and floating point numbers.
"""

from typing import List, Tuple, Optional


class Token:
    """Represents a lexical token in the expression."""
    
    def __init__(self, type: str, value: Optional[float] = None):
        self.type = type  # 'NUMBER', 'PLUS', 'MINUS', 'MULT', 'DIV', 'LPAREN', 'RPAREN'
        self.value = value
    
    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value})"


class ExpressionEvaluator:
    """
    Evaluates mathematical expressions using recursive descent parsing.
    
    Supports:
    - Binary operators: +, -, *, /
    - Unary minus: -3, -(2+1)
    - Parentheses for grouping
    - Floating point numbers
    
    Raises ValueError for invalid expressions.
    """
    
    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: A string containing a mathematical expression.
            
        Returns:
            The result as a float.
            
        Raises:
            ValueError: If the expression is empty, has mismatched parentheses,
                       contains invalid tokens, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        self._tokens = self._tokenize(expr)
        self._pos = 0
        
        # Check for mismatched parentheses
        self._check_parentheses_balance()
        
        result = self._parse_expression()
        
        # Ensure all tokens were consumed
        if self._pos < len(self._tokens):
            raise ValueError(
                f"Unexpected token '{self._tokens[self._pos]}' at position {self._pos}"
            )
        
        return result
    
    def _tokenize(self, expr: str) -> List[Token]:
        """
        Convert expression string into a list of tokens.
        
        Args:
            expr: The expression string to tokenize.
            
        Returns:
            A list of Token objects.
            
        Raises:
            ValueError: If an invalid character is encountered.
        """
        tokens: List[Token] = []
        i = 0
        expr = expr.strip()
        
        while i < len(expr):
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
            elif char == '+':
                tokens.append(Token('PLUS'))
                i += 1
            elif char == '-':
                tokens.append(Token('MINUS'))
                i += 1
            elif char == '*':
                tokens.append(Token('MULT'))
                i += 1
            elif char == '/':
                tokens.append(Token('DIV'))
                i += 1
            elif char == '(':
                tokens.append(Token('LPAREN'))
                i += 1
            elif char == ')':
                tokens.append(Token('RPAREN'))
                i += 1
            elif char.isdigit() or char == '.':
                # Parse a number (integer or float)
                j = i
                has_dot = False
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format at position {j}")
                        has_dot = True
                    j += 1
                num_str = expr[i:j]
                try:
                    value = float(num_str)
                    tokens.append(Token('NUMBER', value))
                except ValueError:
                    raise ValueError(f"Invalid number '{num_str}' at position {i}")
                i = j
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")
        
        if not tokens:
            raise ValueError("Empty expression")
        
        return tokens
    
    def _check_parentheses_balance(self) -> None:
        """
        Verify that parentheses are properly balanced.
        
        Raises:
            ValueError: If parentheses are mismatched.
        """
        depth = 0
        for i, token in enumerate(self._tokens):
            if token.type == 'LPAREN':
                depth += 1
            elif token.type == 'RPAREN':
                depth -= 1
                if depth < 0:
                    raise ValueError(
                        f"Unmatched closing parenthesis at position {i}"
                    )
        if depth > 0:
            raise ValueError(
                f"Unmatched opening parenthesis: {depth} unclosed"
            )
    
    def _current_token(self) -> Optional[Token]:
        """Get the current token without advancing."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None
    
    def _consume(self) -> Token:
        """
        Consume and return the current token, advancing the position.
        
        Returns:
            The current token.
        """
        token = self._current_token()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
        return token
    
    def _expect(self, token_type: str) -> Token:
        """
        Consume and return the current token if it matches the expected type.
        
        Args:
            token_type: The expected token type.
            
        Returns:
            The consumed token.
            
        Raises:
            ValueError: If the current token doesn't match.
        """
        token = self._current_token()
        if token is None or token.type != token_type:
            expected = f"'{token_type}'"
            actual = f"'{token.type}'" if token else "end of expression"
            raise ValueError(f"Expected {expected} but got {actual}")
        return self._consume()
    
    # Grammar rules implemented as methods
    
    def _parse_expression(self) -> float:
        """
        Parse: expression → term (('+' | '-') term)*
        
        Returns:
            The evaluated result.
        """
        result = self._parse_term()
        
        while self._current_token() and self._current_token().type in ('PLUS', 'MINUS'):
            op = self._consume()
            right = self._parse_term()
            
            if op.type == 'PLUS':
                result += right
            else:  # MINUS
                result -= right
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse: term → factor (('*' | '/') factor)*
        
        Returns:
            The evaluated result.
        """
        result = self._parse_factor()
        
        while self._current_token() and self._current_token().type in ('MULT', 'DIV'):
            op = self._consume()
            right = self._parse_factor()
            
            if op.type == 'MULT':
                result *= right
            else:  # DIV
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parse: factor → ('+' | '-') factor | '(' expression ')' | number
        
        Returns:
            The evaluated result.
        """
        token = self._current_token()
        
        if token is None:
            raise ValueError("Unexpected end of expression")
        
        # Unary plus or minus
        if token.type == 'MINUS':
            self._consume()
            return -self._parse_factor()
        elif token.type == 'PLUS':
            self._consume()
            return self._parse_factor()
        
        # Parenthesized expression
        if token.type == 'LPAREN':
            self._consume()
            result = self._parse_expression()
            self._expect('RPAREN')
            return result
        
        # Number
        if token.type == 'NUMBER':
            self._consume()
            return token.value
        
        raise ValueError(f"Unexpected token '{token.type}'")


# ============================================================================
# Pytest Tests
# ============================================================================

if __name__ == "__main__":
    import pytest
    import sys
    
    # Run tests
    sys.exit(pytest.main([__file__, "-v"]))


def test_basic_arithmetic():
    """Test basic addition, subtraction, multiplication, and division."""
    evaluator = ExpressionEvaluator()
    
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 2") == 5.0
    assert evaluator.evaluate("1 + 2 + 3") == 6.0
    assert evaluator.evaluate("10 - 3 - 2") == 5.0


def test_operator_precedence():
    """Test that multiplication/division have higher precedence than addition/subtraction."""
    evaluator = ExpressionEvaluator()
    
    # 2 + 3 * 4 should be 2 + 12 = 14, not 20
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    
    # 10 - 2 * 3 should be 10 - 6 = 4, not 24
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    
    # 2 * 3 + 4 * 5 should be 6 + 20 = 26
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    
    # 100 / 10 / 2 should be 10 / 2 = 5 (left-to-right)
    assert evaluator.evaluate("100 / 10 / 2") == 5.0
    
    # 2 + 3 * 4 - 5 * 6 should be 2 + 12 - 30 = -16
    assert evaluator.evaluate("2 + 3 * 4 - 5 * 6") == -16.0


def test_parentheses():
    """Test parentheses for grouping and overriding precedence."""
    evaluator = ExpressionEvaluator()
    
    # (2 + 3) * 4 should be 5 * 4 = 20
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    
    # 2 * (3 + 4) should be 2 * 7 = 14
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    
    # Nested parentheses: ((2 + 3) * 4) / 2
    assert evaluator.evaluate("((2 + 3) * 4) / 2") == 10.0
    
    # (10 - 5) * (3 + 2) = 5 * 5 = 25
    assert evaluator.evaluate("(10 - 5) * (3 + 2)") == 25.0
    
    # Complex: (2 + 3) * (4 - 1) / (5 - 2) = 5 * 3 / 3 = 5
    assert evaluator.evaluate("(2 + 3) * (4 - 1) / (5 - 2)") == 5.0


def test_unary_minus():
    """Test unary minus operations."""
    evaluator = ExpressionEvaluator()
    
    # Simple unary minus
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-3.14") == -3.14
    
    # Unary minus with expression
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    
    # Unary minus in arithmetic
    assert evaluator.evaluate("10 + -3") == 7.0
    assert evaluator.evaluate("10 - -3") == 13.0
    
    # Multiple unary minuses
    assert evaluator.evaluate("-(-5)") == 5.0
    assert evaluator.evaluate("-(-(-3))") == -3.0
    
    # Unary minus with parentheses
    assert evaluator.evaluate("- (2 * 3)") == -6.0


def test_error_cases():
    """Test various error conditions."""
    evaluator = ExpressionEvaluator()
    
    # Empty expression
    try:
        evaluator.evaluate("")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()
    
    # Mismatched parentheses - unclosed
    try:
        evaluator.evaluate("(2 + 3")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unmatched" in str(e).lower() or "parenthesis" in str(e).lower()
    
    # Mismatched parentheses - extra closing
    try:
        evaluator.evaluate("2 + 3)")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unmatched" in str(e).lower() or "parenthesis" in str(e).lower()
    
    # Division by zero
    try:
        evaluator.evaluate("10 / 0")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "division by zero" in str(e).lower()
    
    # Invalid character
    try:
        evaluator.evaluate("2 + @ 3")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid" in str(e).lower()
    
    # Whitespace only
    try:
        evaluator.evaluate("   ")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()
```

## Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Parser Type** | Recursive Descent | Natural mapping to grammar, easy to extend |
| **Tokenization** | Separate phase | Cleaner separation of concerns |
| **Error Handling** | ValueError with messages | Pythonic, descriptive |
| **Precedence** | Grammar hierarchy | `term` > `expression` naturally gives `*` > `+` |
| **Unary Minus** | In `factor` rule | Standard approach in expression grammars |

## How It Works

1. **Tokenization**: Converts `"2 + 3 * 4"` → `[NUMBER(2), PLUS, NUMBER(3), MULT, NUMBER(4)]`

2. **Parsing**:
   - `_parse_expression()` handles `+` and `-` (lowest precedence)
   - `_parse_term()` handles `*` and `/` (higher precedence)
   - `_parse_factor()` handles numbers, parentheses, and unary operators

3. **Example trace for `2 + 3 * 4`**:
   ```
   _parse_expression()
     → _parse_term() returns 2
     → sees PLUS, consumes it
     → _parse_term()
         → _parse_factor() returns 3
         → sees MULT, consumes it
         → _parse_factor() returns 4
         → returns 3 * 4 = 12
     → returns 2 + 12 = 14
   ```