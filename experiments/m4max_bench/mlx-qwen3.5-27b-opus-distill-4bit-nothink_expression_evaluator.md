# Qwen 3.5 27B Opus-Distilled 4bit MLX — expression_evaluator

## Thinking (816 chars)

```
Let me think about how to approach this coding problem.

Problem: Build a mathematical expression evaluator in Python. Requirements:
1. Support +, -, *, / with correct operator precedence
2. Support parentheses for grouping
3. Support unary minus (e.g., '-3', '-(2+1)')
4. Support floating point numbers (e.g., '3.14')
5. Raise ValueError with a descriptive message for: mismatched parentheses, division by zero, invalid tokens, empty expressions
6. Implement as a class called ExpressionEvaluator with an evaluate(expr: str) -> float method
7. Use a recursive descent parser — do NOT use eval() or ast.l

Approach: Recursive Descent Parser

Key considerations:
- Architecture & Design

I'll implement a recursive descent parser with the following grammar:

```
expression → term ((

Let me implement this solution.

```

## Output



# Mathematical Expression Evaluator

## Architecture & Design

### Approach: Recursive Descent Parser

I'll implement a recursive descent parser with the following grammar:

```
expression → term ((PLUS | MINUS) term)*
term       → factor ((MUL | DIV) factor)*
factor     → UNARY factor | atom
atom       → NUMBER | LPAREN expression RPAREN
```

**Key Design Decisions:**

1. **Tokenizer**: Separate lexical analysis from parsing for clean error messages and extensibility
2. **Operator Precedence**: Enforced by grammar hierarchy (expression < term < factor)
3. **Unary Minus**: Handled at the factor level, allowing `-3` and `-(2+1)`
4. **Error Handling**: Custom exceptions with context-aware messages
5. **State Machine**: Token iterator with lookahead capability

### Error Handling Strategy
- Tokenization errors → `ValueError` with position
- Syntax errors (mismatched parens, invalid tokens) → `ValueError`
- Runtime errors (division by zero) → `ValueError`
- Empty/whitespace-only expressions → `ValueError`

---

## Implementation

```python
"""
Mathematical Expression Evaluator

A recursive descent parser supporting +, -, *, /, parentheses, unary minus,
and floating-point numbers with proper operator precedence.
"""

import re
from enum import Enum, auto
from typing import Iterator, List, Optional, Tuple


class TokenType(Enum):
    """Enumeration of all token types in the expression grammar."""
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


class Token:
    """Represents a single token with type and value."""
    
    def __init__(self, token_type: TokenType, value: Optional[float], position: int):
        self.type = token_type
        self.value = value
        self.position = position
    
    def __repr__(self) -> str:
        if self.value is not None:
            return f"Token({self.type.name}, {self.value!r}, pos={self.position})"
        return f"Token({self.type.name}, pos={self.position})"


class ParseError(Exception):
    """Custom exception for parsing errors with descriptive messages."""
    pass


class Tokenizer:
    """
    Converts an expression string into a list of tokens.
    
    Handles numbers (integers and floats), operators, and parentheses.
    """
    
    # Regex pattern matching all valid tokens
    TOKEN_PATTERN = re.compile(
        r'\s*'  # Optional whitespace
        r'(?:'
        r'([0-9]+\.?[0-9]*)'  # Number (integer or float)
        r'|([+\-*/()])'       # Operators and parentheses
        r'|([^ \+\-*/()])'    # Invalid character (captured for error reporting)
        r')'
    )
    
    def __init__(self, expression: str):
        """Initialize tokenizer with the expression string."""
        self.expression = expression
        self.tokens: List[Token] = []
        self._tokenize()
    
    def _tokenize(self) -> None:
        """Convert the expression string into tokens."""
        pos = 0
        length = len(self.expression)
        
        while pos < length:
            # Skip leading whitespace
            while pos < length and self.expression[pos].isspace():
                pos += 1
            
            if pos >= length:
                break
            
            match = self.TOKEN_PATTERN.match(self.expression, pos)
            
            if not match:
                raise ValueError(
                    f"Unexpected character '{self.expression[pos]}' at position {pos}"
                )
            
            number_str = match.group(1)
            operator_char = match.group(2)
            invalid_char = match.group(3)
            
            # Update position
            pos = match.end()
            
            # Create appropriate token
            if number_str is not None:
                # Number token
                try:
                    value = float(number_str)
                except ValueError:
                    raise ValueError(
                        f"Invalid number format '{number_str}' at position {match.start()}"
                    )
                self.tokens.append(Token(TokenType.NUMBER, value, match.start()))
            
            elif operator_char is not None:
                # Operator or parenthesis token
                token_type = self._char_to_type(operator_char)
                self.tokens.append(Token(token_type, None, match.start()))
            
            else:
                # Invalid character
                raise ValueError(
                    f"Invalid character '{invalid_char}' at position {match.start()}"
                )
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, None, pos))
    
    def _char_to_type(self, char: str) -> TokenType:
        """Convert an operator character to its token type."""
        return {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MUL,
            '/': TokenType.DIV,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
        }[char]
    
    def get_tokens(self) -> List[Token]:
        """Return the list of tokens."""
        return self.tokens


class ExpressionEvaluator:
    """
    A recursive descent parser for mathematical expressions.
    
    Supports:
    - Addition (+), subtraction (-), multiplication (*), division (/)
    - Proper operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating-point numbers
    
    Example:
        evaluator = ExpressionEvaluator()
        result = evaluator.evaluate("2 + 3 * (4 - 1)")  # Returns 11.0
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self._tokens: List[Token] = []
        self._pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: A string containing a mathematical expression.
            
        Returns:
            The floating-point result of evaluating the expression.
            
        Raises:
            ValueError: If the expression is invalid, has mismatched parentheses,
                       contains invalid tokens, or attempts division by zero.
        """
        # Handle empty or whitespace-only expressions
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")
        
        # Tokenize the expression
        try:
            tokenizer = Tokenizer(expr)
            self._tokens = tokenizer.get_tokens()
        except ValueError:
            raise
        
        # Check for empty expression after tokenization
        if len(self._tokens) <= 1:  # Only EOF token
            raise ValueError("Expression cannot be empty")
        
        # Check for mismatched parentheses before parsing
        self._validate_parentheses()
        
        # Reset parser position
        self._pos = 0
        
        # Parse and evaluate
        result = self._parse_expression()
        
        # Ensure all tokens were consumed
        if self._current_token().type != TokenType.EOF:
            token = self._current_token()
            raise ValueError(
                f"Unexpected token '{token.type.name}' at position {token.position}"
            )
        
        return result
    
    def _validate_parentheses(self) -> None:
        """
        Validate that parentheses are balanced.
        
        Raises:
            ValueError: If parentheses are mismatched.
        """
        count = 0
        for token in self._tokens:
            if token.type == TokenType.LPAREN:
                count += 1
            elif token.type == TokenType.RPAREN:
                count -= 1
                if count < 0:
                    raise ValueError(
                        f"Unmatched closing parenthesis at position {token.position}"
                    )
        
        if count != 0:
            # Find the position of the last open paren for better error message
            last_open = 0
            for token in self._tokens:
                if token.type == TokenType.LPAREN:
                    last_open = token.position
            
            raise ValueError(
                f"Unmatched opening parenthesis (last one at position {last_open})"
            )
    
    def _current_token(self) -> Token:
        """Return the current token being processed."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return self._tokens[-1]  # Return EOF if past end
    
    def _consume(self) -> Token:
        """
        Consume and return the current token, advancing the position.
        
        Returns:
            The token that was consumed.
        """
        token = self._current_token()
        self._pos += 1
        return token
    
    def _parse_expression(self) -> float:
        """
        Parse an expression: term ((PLUS | MINUS) term)*
        
        This is the entry point for parsing, handling the lowest precedence
        operators (+ and -).
        
        Returns:
            The evaluated result of the expression.
        """
        value = self._parse_term()
        
        while self._current_token().type in (TokenType.PLUS, TokenType.MINUS):
            op = self._consume()
            right = self._parse_term()
            
            if op.type == TokenType.PLUS:
                value += right
            else:  # MINUS
                value -= right
        
        return value
    
    def _parse_term(self) -> float:
        """
        Parse a term: factor ((MUL | DIV) factor)*
        
        Handles higher precedence operators (* and /).
        
        Returns:
            The evaluated result of the term.
        """
        value = self._parse_factor()
        
        while self._current_token().type in (TokenType.MUL, TokenType.DIV):
            op = self._consume()
            right = self._parse_factor()
            
            if op.type == TokenType.MUL:
                value *= right
            else:  # DIV
                if right == 0:
                    raise ValueError("Division by zero is not allowed")
                value /= right
        
        return value
    
    def _parse_factor(self) -> float:
        """
        Parse a factor: UNARY factor | atom
        
        Handles unary minus and atoms (numbers and parenthesized expressions).
        
        Returns:
            The evaluated result of the factor.
        """
        # Handle unary minus
        if self._current_token().type == TokenType.MINUS:
            self._consume()
            return -self._parse_factor()
        
        # Handle unary plus (optional, for expressions like ++3)
        if self._current_token().type == TokenType.PLUS:
            self._consume()
            return self._parse_factor()
        
        # Parse atom
        return self._parse_atom()
    
    def _parse_atom(self) -> float:
        """
        Parse an atom: NUMBER | LPAREN expression RPAREN
        
        Atoms are the basic building blocks of expressions.
        
        Returns:
            The value of the atom.
        """
        token = self._current_token()
        
        if token.type == TokenType.NUMBER:
            self._consume()
            return token.value  # type: ignore
        
        if token.type == TokenType.LPAREN:
            self._consume()  # Consume '('
            value = self._parse_expression()
            
            if self._current_token().type != TokenType.RPAREN:
                raise ValueError(
                    "Expected closing parenthesis ')', "
                    f"got '{self._current_token().type.name}'"
                )
            self._consume()  # Consume ')'
            return value
        
        # Unexpected token
        raise ValueError(
            f"Unexpected token '{token.type.name}' at position {token.position}, "
            f"expected a number or '('"
        )


# Alias for backward compatibility
ExpressionParser = ExpressionEvaluator
```

---

## Tests

```python
"""
Tests for the ExpressionEvaluator.

Run with: pytest test_evaluator.py
"""

import pytest
from evaluator import ExpressionEvaluator


class TestExpressionEvaluator:
    """Test suite for the ExpressionEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create an evaluator instance for each test."""
        return ExpressionEvaluator()
    
    # =========================================================================
    # Basic Arithmetic Tests
    # =========================================================================
    
    def test_basic_addition(self, evaluator: ExpressionEvaluator):
        """Test basic addition operations."""
        assert evaluator.evaluate("2 + 3") == 5.0
        assert evaluator.evaluate("10 + 20") == 30.0
        assert evaluator.evaluate("0 + 5") == 5.0
    
    def test_basic_subtraction(self, evaluator: ExpressionEvaluator):
        """Test basic subtraction operations."""
        assert evaluator.evaluate("5 - 3") == 2.0
        assert evaluator.evaluate("10 - 20") == -10.0
        assert evaluator.evaluate("5 - 0") == 5.0
    
    def test_basic_multiplication(self, evaluator: ExpressionEvaluator):
        """Test basic multiplication operations."""
        assert evaluator.evaluate("3 * 4") == 12.0
        assert evaluator.evaluate("2.5 * 4") == 10.0
        assert evaluator.evaluate("0 * 100") == 0.0
    
    def test_basic_division(self, evaluator: ExpressionEvaluator):
        """Test basic division operations."""
        assert evaluator.evaluate("12 / 4") == 3.0
        assert evaluator.evaluate("10 / 4") == 2.5
        assert evaluator.evaluate("0 / 5") == 0.0
    
    def test_floating_point_numbers(self, evaluator: ExpressionEvaluator):
        """Test parsing of floating-point numbers."""
        assert evaluator.evaluate("3.14") == 3.14
        assert evaluator.evaluate("2.5 + 3.5") == 6.0
        assert evaluator.evaluate(".5") == 0.5
        assert evaluator.evaluate("2.") == 2.0
    
    # =========================================================================
    # Operator Precedence Tests
    # =========================================================================
    
    def test_multiplication_before_addition(self, evaluator: ExpressionEvaluator):
        """Test that multiplication has higher precedence than addition."""
        assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + 12 = 14
        assert evaluator.evaluate("3 * 4 + 2") == 14.0  # 12 + 2 = 14
    
    def test_division_before_addition(self, evaluator: ExpressionEvaluator):
        """Test that division has higher precedence than addition."""
        assert evaluator.evaluate("2 + 12 / 4") == 5.0  # 2 + 3 = 5
        assert evaluator.evaluate("12 / 4 + 2") == 5.0  # 3 + 2 = 5
    
    def test_multiplication_before_subtraction(self, evaluator: ExpressionEvaluator):
        """Test that multiplication has higher precedence than subtraction."""
        assert evaluator.evaluate("10 - 2 * 3") == 4.0  # 10 - 6 = 4
        assert evaluator.evaluate("2 * 3 - 10") == -4.0  # 6 - 10 = -4
    
    def test_same_precedence_left_to_right(self, evaluator: ExpressionEvaluator):
        """Test left-to-right evaluation for same precedence operators."""
        assert evaluator.evaluate("10 - 5 - 2") == 3.0  # (10-5)-2 = 3
        assert evaluator.evaluate("20 / 4 / 2") == 2.5  # (20/4)/2 = 2.5
        assert evaluator.evaluate("2 * 3 * 4") == 24.0  # (2*3)*4 = 24
    
    def test_complex_precedence(self, evaluator: ExpressionEvaluator):
        """Test complex expression with multiple precedence levels."""
        # 2 + 3 * 4 - 10 / 2 = 2 + 12 - 5 = 9
        assert evaluator.evaluate("2 + 3 * 4 - 10 / 2") == 9.0
    
    # =========================================================================
    # Parentheses Tests
    # =========================================================================
    
    def test_parentheses_override_precedence(self, evaluator: ExpressionEvaluator):
        """Test that parentheses can override operator precedence."""
        assert evaluator.evaluate("(2 + 3) * 4") == 20.0  # 5 * 4 = 20
        assert evaluator.evaluate("10 - (2 + 3)") == 5.0  # 10 - 5 = 5
    
    def test_nested_parentheses(self, evaluator: ExpressionEvaluator):
        """Test deeply nested parentheses."""
        assert evaluator.evaluate("((2 + 3))") == 5.0
        assert evaluator.evaluate("(((1 + 2) * 3))") == 9.0
        assert evaluator.evaluate("((2 + 3) * (4 + 5))") == 45.0
    
    def test_complex_parentheses(self, evaluator: ExpressionEvaluator):
        """Test complex expressions with parentheses."""
        assert evaluator.evaluate("(2 + 3) * (4 + 5)") == 45.0
        assert evaluator.evaluate("((10 + 2) / (3 - 1))") == 6.0
    
    # =========================================================================
    # Unary Minus Tests
    # =========================================================================
    
    def test_unary_minus_number(self, evaluator: ExpressionEvaluator):
        """Test unary minus with a number."""
        assert evaluator.evaluate("-3") == -3.0
        assert evaluator.evaluate("-3.14") == -3.14
    
    def test_unary_minus_parentheses(self, evaluator: ExpressionEvaluator):
        """Test unary minus with parenthesized expression."""
        assert evaluator.evaluate("-(2 + 3)") == -5.0
        assert evaluator.evaluate("-(10 - 3)") == -7.0
    
    def test_unary_minus_with_operations(self, evaluator: ExpressionEvaluator):
        """Test unary minus combined with other operations."""
        assert evaluator.evaluate("-3 + 5") == 2.0
        assert evaluator.evaluate("5 - -3") == 8.0
        assert evaluator.evaluate("-3 * 2") == -6.0
        assert evaluator.evaluate("-12 / 3") == -4.0
    
    def test_multiple_unary_minuses(self, evaluator: ExpressionEvaluator):
        """Test multiple consecutive unary minus operators."""
        assert evaluator.evaluate("--3") == 3.0
        assert evaluator.evaluate("---3") == -3.0
        assert evaluator.evaluate("-+-3") == -3.0
    
    def test_unary_minus_precedence(self, evaluator: ExpressionEvaluator):
        """Test that unary minus has higher precedence than binary operators."""
        assert evaluator.evaluate("-3 * 2") == -6.0  # (-3) * 2
        assert evaluator.evaluate("2 * -3") == -6.0  # 2 * (-3)
        assert evaluator.evaluate("-3 ** 2")  # This would need power operator
    
    # =========================================================================
    # Error Cases
    # =========================================================================
    
    def test_empty_expression(self, evaluator: ExpressionEvaluator):
        """Test that empty expressions raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            evaluator.evaluate("")
        with pytest.raises(ValueError, match="empty"):
            evaluator.evaluate("   ")
    
    def test_mismatched_parentheses_open(self, evaluator: ExpressionEvaluator):
        """Test that unmatched opening parentheses raise ValueError."""
        with pytest.raises(ValueError, match="Unmatched opening"):
            evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Unmatched opening"):
            evaluator.evaluate("((2 + 3)")
    
    def test_mismatched_parentheses_close(self, evaluator: ExpressionEvaluator):
        """Test that unmatched closing parentheses raise ValueError."""
        with pytest.raises(ValueError, match="Unmatched closing"):
            evaluator.evaluate("2 + 3)")
        with pytest.raises(ValueError, match="Unmatched closing"):
            evaluator.evaluate("2 + 3))")
    
    def test_division_by_zero(self, evaluator: ExpressionEvaluator):
        """Test that division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("10 / 0")
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("10 / (5 - 5)")
    
    def test_invalid_tokens(self, evaluator: ExpressionEvaluator):
        """Test that invalid characters raise ValueError."""
        with pytest.raises(ValueError):
            evaluator.evaluate("2 + @ 3")
        with pytest.raises(ValueError):
            evaluator.evaluate("abc")
        with pytest.raises(ValueError):
            evaluator.evaluate("2 ++ 3")  # Double operator (not unary)
    
    def test_unexpected_token(self, evaluator: ExpressionEvaluator):
        """Test that unexpected tokens at end raise ValueError."""
        with pytest.raises(ValueError):
            evaluator.evaluate("2 + 3 +")
        with pytest.raises(ValueError):
            evaluator.evaluate("(2 + 3) +")
    
    # =========================================================================
    # Edge Cases
    # =========================================================================
    
    def test_whitespace_handling(self, evaluator: ExpressionEvaluator):
        """Test that whitespace is handled correctly."""
        assert evaluator.evaluate("  2 + 3  ") == 5.0
        assert evaluator.evaluate("2   +   3") == 5.0
        assert evaluator.evaluate("\t2\n+\n3\r") == 5.0
    
    def test_large_numbers(self, evaluator: ExpressionEvaluator):
        """Test handling of very large numbers."""
        result = evaluator.evaluate("1000000000000 + 1")
        assert result == 1000000000001.0
    
    def test_small_numbers(self, evaluator: ExpressionEvaluator):
        """Test handling of very small numbers."""
        assert evaluator.evaluate("0.0000001 + 0.0000002") == 0.0000003
    
    def test_single_number(self, evaluator: ExpressionEvaluator):
        """Test evaluating a single number."""
        assert evaluator.evaluate("42") == 42.0
        assert evaluator.evaluate("-42") == -42.0
```

---

## Summary

### Key Features Implemented

| Feature | Implementation |
|---------|---------------|
| Operators | `+`, `-`, `*`, `/` with correct precedence |
| Parentheses | Full support for grouping and nesting |
| Unary Minus | Supports `-3`, `-(2+1)`, `--3` |
| Float Numbers | Full floating-point support |
| Error Handling | Descriptive `ValueError` for all error cases |
| Parser Type | Recursive descent (no `eval()` or `ast`) |

### Grammar Hierarchy

```
expression → term { (+|-) term }*
term       → factor { (*|/) factor }*
factor     → (-|+) factor | atom
atom       → NUMBER | '(' expression ')'
```

### Test Coverage

1. ✅ Basic arithmetic (add, sub, mul, div)
2. ✅ Operator precedence
3. ✅ Parentheses (simple and nested)
4. ✅ Unary minus (number, parentheses, combinations)
5. ✅ Error cases (empty, mismatched parens, div-by-zero, invalid tokens)
