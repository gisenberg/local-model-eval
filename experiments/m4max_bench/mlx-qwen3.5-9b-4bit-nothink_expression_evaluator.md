# Qwen 3.5 9B 4bit MLX — expression_evaluator

## Output

Here is a complete implementation of the `ExpressionEvaluator` class using a recursive descent parser, along with the requested pytest tests.

### Implementation: `expression_evaluator.py`

```python
import re
from typing import List, Tuple, Union

class Token:
    def __init__(self, type_: str, value: Union[int, float, str]):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"

class ExpressionEvaluator:
    """
    A mathematical expression evaluator supporting +, -, *, /, parentheses,
    and unary minus using a recursive descent parser.
    """

    def __init__(self):
        # Token types
        self.TOKEN_NUM = 'NUMBER'
        self.TOKEN_PLUS = 'PLUS'
        self.TOKEN_MINUS = 'MINUS'
        self.TOKEN_MUL = 'MUL'
        self.TOKEN_DIV = 'DIV'
        self.TOKEN_LPAREN = 'LPAREN'
        self.TOKEN_RPAREN = 'RPAREN'
        self.TOKEN_EOF = 'EOF'

        # Token patterns
        self.token_pattern = re.compile(
            r'^'
            r'(?:'
            r'([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)'  # Numbers (int/float/scientific)
            r'|([+-])'                                    # Operators
            r'|(\()')                                     # Parentheses
            r')'
        )

        self.tokens: List[Token] = []
        self.pos = 0

    def tokenize(self, expr: str) -> List[Token]:
        """
        Converts a string expression into a list of Token objects.
        Raises ValueError for invalid characters or mismatched parentheses.
        """
        tokens = []
        current = ""
        i = 0
        length = len(expr)

        while i < length:
            char = expr[i]
            
            # Handle whitespace
            if char.isspace():
                i += 1
                continue
            
            # Handle numbers (including floats and scientific notation)
            if char.isdigit() or char == '.':
                current += char
                i += 1
                # Continue reading while it's a digit or decimal point
                while i < length and (expr[i].isdigit() or expr[i] == '.'):
                    current += expr[i]
                    i += 1
                # Check for scientific notation (e.g., 1e10, 2.5E-3)
                if i < length and expr[i] in 'eE':
                    current += expr[i]
                    i += 1
                    # Exponent part
                    if i < length and expr[i] in '+-':
                        current += expr[i]
                        i += 1
                    while i < length and expr[i].isdigit():
                        current += expr[i]
                        i += 1
                tokens.append(Token(self.TOKEN_NUM, float(current)))
                current = ""
            
            # Handle operators and parentheses
            elif char in '+-*/()':
                if current:
                    raise ValueError(f"Invalid token: '{current}' at position {i - len(current)}")
                tokens.append(Token(char, char))
                i += 1
            
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")

        # Check for balanced parentheses
        open_count = sum(1 for t in tokens if t.type == self.TOKEN_LPAREN)
        close_count = sum(1 for t in tokens if t.type == self.TOKEN_RPAREN)
        
        if open_count != close_count:
            raise ValueError("Mismatched parentheses in expression")

        return tokens

    def parse(self, expr: str) -> float:
        """
        Parses the expression string and evaluates it.
        Raises ValueError for division by zero or empty expressions.
        """
        self.tokens = self.tokenize(expr)
        
        if not self.tokens:
            raise ValueError("Empty expression")

        # Start parsing
        result = self._parse_additive()
        
        if self.pos < len(self.tokens):
            raise ValueError("Unexpected token at end of expression")

        return result

    def _current_token(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(self.TOKEN_EOF, None)

    def _peek_token(self) -> Token:
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        return Token(self.TOKEN_EOF, None)

    def _advance(self) -> Token:
        token = self._current_token()
        self.pos += 1
        return token

    def _parse_additive(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Handles unary minus as a prefix operator here.
        """
        token = self._current_token()
        
        # Handle Unary Minus
        if token.type == self.TOKEN_MINUS:
            self._advance()
            # Recursively parse the operand
            value = -self._parse_multiplicative()
            return value
        
        # Handle Unary Plus (optional, but good for consistency)
        if token.type == self.TOKEN_PLUS:
            self._advance()
            return self._parse_multiplicative()

        # Handle Number or Parenthesis
        if token.type == self.TOKEN_NUM:
            self._advance()
            return token.value
        
        if token.type == self.TOKEN_LPAREN:
            self._advance()
            value = self._parse_additive()
            if self._current_token().type != self.TOKEN_RPAREN:
                raise ValueError("Mismatched parentheses: expected ')'")
            self._advance()
            return value
        
        raise ValueError(f"Unexpected token: {token.type}")

    def _parse_multiplicative(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        """
        left = self._parse_additive()
        token = self._current_token()

        while token.type in (self.TOKEN_MUL, self.TOKEN_DIV):
            op = token.type
            self._advance()
            right = self._parse_additive()

            if op == self.TOKEN_DIV:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
            else:
                left = left * right
            
            token = self._current_token()

        return left
```

### Tests: `test_expression_evaluator.py`

```python
import pytest
from expression_evaluator import ExpressionEvaluator

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

class TestBasicArithmetic:
    def test_addition(self, evaluator):
        assert evaluator.evaluate("2 + 3") == 5.0

    def test_subtraction(self, evaluator):
        assert evaluator.evaluate("10 - 4") == 6.0

    def test_multiplication(self, evaluator):
        assert evaluator.evaluate("4 * 5") == 20.0

    def test_division(self, evaluator):
        assert evaluator.evaluate("10 / 2") == 5.0

    def test_mixed_basic(self, evaluator):
        assert evaluator.evaluate("2 + 3 * 4") == 14.0

class TestPrecedence:
    def test_mul_before_add(self, evaluator):
        # 2 + 3 * 4 should be 2 + 12 = 14
        assert evaluator.evaluate("2 + 3 * 4") == 14.0

    def test_div_before_mul(self, evaluator):
        # 10 / 2 * 5 should be 5 * 5 = 25
        assert evaluator.evaluate("10 / 2 * 5") == 25.0

    def test_left_associativity(self, evaluator):
        # 10 - 5 - 2 should be (10-5)-2 = 3
        assert evaluator.evaluate("10 - 5 - 2") == 3.0

    def test_complex_precedence(self, evaluator):
        # 2 + 3 * 4 - 5 / 1
        # 2 + 12 - 5 = 9
        assert evaluator.evaluate("2 + 3 * 4 - 5 / 1") == 9.0

class TestParentheses:
    def test_simple_parentheses(self, evaluator):
        assert evaluator.evaluate("(2 + 3) * 4") == 20.0

    def test_nested_parentheses(self, evaluator):
        assert evaluator.evaluate("((2 + 3) * 4) - 1") == 19.0

    def test_mismatched_parentheses_open(self, evaluator):
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("(2 + 3")

    def test_mismatched_parentheses_close(self, evaluator):
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("2 + 3)")

    def test_empty_expression(self, evaluator):
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")

class TestUnaryMinus:
    def test_unary_minus_number(self, evaluator):
        assert evaluator.evaluate("-5") == -5.0

    def test_unary_minus_expression(self, evaluator):
        assert evaluator.evaluate("- (2 + 3)") == -5.0

    def test_unary_minus_complex(self, evaluator):
        assert evaluator.evaluate("-2 * 3 + 4") == -2.0

    def test_double_unary_minus(self, evaluator):
        assert evaluator.evaluate("--5") == 5.0

    def test_unary_minus_with_parentheses(self, evaluator):
        assert evaluator.evaluate("-(2 + 3) * 2") == -10.0

class TestErrorCases:
    def test_division_by_zero(self, evaluator):
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("10 / 0")

    def test_invalid_char(self, evaluator):
        with pytest.raises(ValueError, match="Invalid character"):
            evaluator.evaluate("2 + a * 3")

    def test_float_division(self, evaluator):
        result = evaluator.evaluate("10 / 3")
        assert abs(result - 3.3333333333333335) < 1e-9

    def test_scientific_notation(self, evaluator):
        result = evaluator.evaluate("1e2 + 2e-1")
        # 100 + 0.2 = 100.2
        assert abs(result - 100.2) < 1e-9

    def test_empty_expression(self, evaluator):
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")

    def test_unexpected_token(self, evaluator):
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluator.evaluate("2 + + 3")
```

### How to Run

1.  Save the first code block as `expression_evaluator.py`.
2.  Save the second code block as `test_expression_evaluator.py`.
3.  Run the tests:
    ```bash
    pytest test_expression_evaluator.py -v
    ```

### Key Design Decisions

1.  **Recursive Descent**: The parser is structured into methods representing precedence levels:
    *   `_parse_additive`: Handles `+` and `-`.
    *   `_parse_multiplicative`: Handles `*` and `/`.
    *   This naturally enforces operator precedence without needing a stack-based shunting-yard algorithm.
2.  **Unary Minus**: Implemented inside `_parse_additive`. When the parser sees a `-` at the start of an additive expression, it consumes it and negates the result of the subsequent multiplicative expression.
3.  **Tokenization**: A custom tokenizer handles floats, scientific notation (`1e10`), and parentheses. It validates the input stream before parsing begins.
4.  **Error Handling**:
    *   **Mismatched Parentheses**: Checked during the tokenization phase by counting open vs. close parens.
    *   **Division by Zero**: Checked explicitly during the division operation.
    *   **Invalid Tokens**: Caught during tokenization if a character doesn't match a number or operator.
    *   **Empty Expression**: Checked after tokenization if the list is empty.
    *   **Unexpected Tokens**: Checked at the end of the parse tree to ensure the entire string was consumed.
5.  **Type Hints**: Used throughout (`List[Token]`, `-> float`, etc.) for clarity and IDE support.
