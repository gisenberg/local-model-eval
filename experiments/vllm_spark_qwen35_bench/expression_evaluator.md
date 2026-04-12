# Qwen3.5-122B vLLM+MTP (Spark) — Expression Evaluator

## Output (10817 chars)



```python
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

class TokenType(Enum):
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()

@dataclass
class Token:
    type: TokenType
    value: float | str

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        parser = _Parser(tokens)
        result = parser.parse()
        
        # Ensure all tokens were consumed
        if parser.current_token.type != TokenType.EOF:
            raise ValueError("Invalid expression: unexpected tokens at end")
            
        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """
        Converts the input string into a list of tokens.
        
        Args:
            expr: The raw expression string.
            
        Returns:
            A list of Token objects.
            
        Raises:
            ValueError: If an invalid character is encountered.
        """
        tokens: List[Token] = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
            
            # Handle numbers (including floats)
            if char.isdigit() or char == '.':
                start = i
                has_dot = False
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid token: multiple decimal points at position {i}")
                        has_dot = True
                    i += 1
                num_str = expr[start:i]
                try:
                    value = float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number format: {num_str}")
                tokens.append(Token(TokenType.NUMBER, value))
                continue
            
            # Handle operators and parentheses
            if char == '+':
                tokens.append(Token(TokenType.PLUS, '+'))
            elif char == '-':
                tokens.append(Token(TokenType.MINUS, '-'))
            elif char == '*':
                tokens.append(Token(TokenType.MULTIPLY, '*'))
            elif char == '/':
                tokens.append(Token(TokenType.DIVIDE, '/'))
            elif char == '(':
                tokens.append(Token(TokenType.LPAREN, '('))
            elif char == ')':
                tokens.append(Token(TokenType.RPAREN, ')'))
            else:
                raise ValueError(f"Invalid token: '{char}' at position {i}")
            
            i += 1
            
        tokens.append(Token(TokenType.EOF, None))
        return tokens


class _Parser:
    """
    Internal recursive descent parser class.
    Grammar:
        Expression -> Term { ('+' | '-') Term }
        Term       -> Factor { ('*' | '/') Factor }
        Factor     -> Number | '(' Expression ')' | UnaryOp Factor
        UnaryOp    -> '+' | '-'
    """
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    @property
    def current_token(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        token = self.current_token
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def parse(self) -> float:
        """
        Entry point for parsing.
        """
        result = self.parse_expression()
        return result

    def parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        """
        left = self.parse_term()
        
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.advance()
            right = self.parse_term()
            if op.type == TokenType.PLUS:
                left += right
            else:
                left -= right
                
        return left

    def parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        """
        left = self.parse_factor()
        
        while self.current_token.type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            op = self.advance()
            right = self.parse_factor()
            if op.type == TokenType.MULTIPLY:
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
                
        return left

    def parse_factor(self) -> float:
        """
        Parses numbers, parentheses, and unary operators.
        """
        token = self.current_token
        
        # Handle Unary Minus (and Unary Plus for completeness)
        if token.type == TokenType.MINUS:
            self.advance()
            return -self.parse_factor()
        elif token.type == TokenType.PLUS:
            self.advance()
            return self.parse_factor()
            
        # Handle Parentheses
        if token.type == TokenType.LPAREN:
            self.advance()
            value = self.parse_expression()
            if self.current_token.type != TokenType.RPAREN:
                raise ValueError("Mismatched parentheses: expected ')'")
            self.advance()
            return value
            
        # Handle Numbers
        if token.type == TokenType.NUMBER:
            self.advance()
            return token.value
            
        # If none of the above, it's an error
        raise ValueError(f"Invalid token: '{token.value}' at position {self.pos}")

```

```python
# test_expression_evaluator.py
import pytest
from expression_evaluator import ExpressionEvaluator

evaluator = ExpressionEvaluator()

def test_basic_arithmetic():
    """Test basic addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("2 + 2") == 4.0
    assert evaluator.evaluate("10 - 5") == 5.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 2") == 5.0

def test_operator_precedence():
    """Test that * and / bind tighter than + and -."""
    # 2 + 3 * 4 should be 2 + 12 = 14
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # 10 - 2 * 3 should be 10 - 6 = 4
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    # 10 / 2 + 1 should be 5 + 1 = 6
    assert evaluator.evaluate("10 / 2 + 1") == 6.0

def test_parentheses():
    """Test grouping with parentheses overrides precedence."""
    # (2 + 3) * 4 should be 5 * 4 = 20
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # 10 / (2 + 3) should be 10 / 5 = 2
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    # Nested parentheses
    assert evaluator.evaluate("((1 + 2) * 3)") == 9.0

def test_unary_minus():
    """Test unary minus support at start and after operators."""
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-3.14") == -3.14
    # -(2 + 1)
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    # 3 - -5 (binary minus followed by unary minus)
    assert evaluator.evaluate("3 - -5") == 8.0
    # - ( - 5 )
    assert evaluator.evaluate("-(-5)") == 5.0

def test_error_cases():
    """Test that appropriate ValueErrors are raised."""
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
        
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
        
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")
```

### Explanation of Implementation

1.  **Tokenizer**: The `_tokenize` method scans the input string character by character. It groups digits and decimal points into `NUMBER` tokens and maps symbols to their respective `TokenType`. It raises `ValueError` immediately for invalid characters or malformed numbers (e.g., multiple dots).
2.  **Recursive Descent Parser**: The `_Parser` class implements the grammar logic.
    *   `parse_expression`: Handles `+` and `-`. It calls `parse_term` to get the left operand, then loops while it sees `+` or `-`, consuming the operator and the next term.
    *   `parse_term`: Handles `*` and `/`. It calls `parse_factor` and loops while it sees `*` or `/`. It includes the **division by zero check** here.
    *   `parse_factor`: Handles the highest precedence items. It checks for **unary minus** (if the current token is `-`, it consumes it and negates the result of the next factor). It also handles **parentheses** by recursively calling `parse_expression`. Finally, it consumes `NUMBER` tokens.
3.  **Error Handling**:
    *   **Empty Expression**: Checked at the start of `evaluate`.
    *   **Mismatched Parentheses**: Checked in `parse_factor` (expecting `)` after `(`) and implicitly by ensuring the parser finishes at `EOF`.
    *   **Invalid Tokens**: Caught during tokenization or if `parse_factor` encounters an unexpected token type.
    *   **Division by Zero**: Explicitly checked before performing division in `parse_term`.
4.  **Type Hints & Docstrings**: All methods and classes include type hints (e.g., `-> float`, `List[Token]`) and docstrings explaining their purpose and arguments.
5.  **Tests**: The `pytest` file covers the five required scenarios: basic arithmetic, precedence, parentheses, unary minus, and specific error conditions.

### How to Run

1.  Save the first code block as `expression_evaluator.py`.
2.  Save the second code block as `test_expression_evaluator.py`.
3.  Install pytest: `pip install pytest`
4.  Run tests: `pytest test_expression_evaluator.py -v`
