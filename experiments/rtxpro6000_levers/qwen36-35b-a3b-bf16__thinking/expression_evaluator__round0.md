Here's a complete, production-ready implementation that satisfies all your requirements. It uses a recursive descent parser with a custom tokenizer, strict type hints, descriptive error handling, and comprehensive pytest tests.

### `expression_evaluator.py`
```python
from typing import List, Tuple, Union

Token = Tuple[str, Union[float, str]]

class ExpressionEvaluator:
    """Evaluates mathematical expressions using a recursive descent parser."""

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result as a float."""
        if not expr or expr.strip() == '':
            raise ValueError("Empty expression")

        tokens: List[Token] = self._tokenize(expr)
        pos: int = 0

        def current() -> Token:
            return tokens[pos]

        def advance() -> Token:
            nonlocal pos
            token = tokens[pos]
            if pos < len(tokens) - 1:
                pos += 1
            return token

        def parse_expr() -> float:
            """Parse addition and subtraction (lowest precedence)."""
            result = parse_term()
            while current()[0] in ('+', '-'):
                op = advance()[0]
                right = parse_term()
                result = result + right if op == '+' else result - right
            return result

        def parse_term() -> float:
            """Parse multiplication and division (higher precedence)."""
            result = parse_factor()
            while current()[0] in ('*', '/'):
                op = advance()[0]
                right = parse_factor()
                if op == '*':
                    result *= right
                else:
                    if right == 0.0:
                        raise ValueError("Division by zero")
                    result /= right
            return result

        def parse_factor() -> float:
            """Parse numbers, unary operators, and parenthesized expressions."""
            token = current()
            if token[0] == '-':
                advance()
                return -parse_factor()
            if token[0] == '+':
                advance()
                return parse_factor()
            if token[0] == 'NUMBER':
                advance()
                return token[1]
            if token[0] == '(':
                advance()
                result = parse_expr()
                if current()[0] != ')':
                    raise ValueError("Mismatched parentheses: missing closing ')'")
                advance()
                return result
            if token[0] == ')':
                raise ValueError("Mismatched parentheses: unexpected ')'")
            raise ValueError(f"Invalid token: '{token[0]}'")

        result = parse_expr()
        if current()[0] != 'EOF':
            raise ValueError("Unexpected token after expression")
        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Convert expression string into a list of tokens."""
        tokens: List[Token] = []
        i = 0
        n = len(expr)
        while i < n:
            if expr[i].isspace():
                i += 1
                continue
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                num_str = expr[i:j]
                try:
                    tokens.append(('NUMBER', float(num_str)))
                except ValueError:
                    raise ValueError(f"Invalid number: '{num_str}'")
                i = j
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")
        tokens.append(('EOF', None))
        return tokens
```

### `test_expression_evaluator.py`
```python
import pytest
from expression_evaluator import ExpressionEvaluator

evaluator = ExpressionEvaluator()

def test_basic_arithmetic() -> None:
    """Test basic +, -, *, / operations."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0

def test_operator_precedence() -> None:
    """Test that * and / bind tighter than + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3 + 4 / 2") == 6.0

def test_parentheses() -> None:
    """Test grouping with parentheses."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0

def test_unary_minus() -> None:
    """Test unary minus and nested unary operators."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0

def test_error_cases() -> None:
    """Test ValueError raising for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
        
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
        
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")
```

### Key Design Choices:
1. **Recursive Descent Grammar**: Follows the standard arithmetic grammar:
   `expr → term (('+' | '-') term)*`
   `term → factor (('*' | '/') factor)*`
   `factor → ('+' | '-') factor | NUMBER | '(' expr ')'`
   This naturally enforces operator precedence without explicit precedence tables.
2. **Tokenizer**: Separates lexical analysis from parsing. Handles floats, ignores whitespace, and validates numeric formats.
3. **Error Handling**: Each failure case raises a `ValueError` with a clear, descriptive message as requested.
4. **Type Hints**: Applied to all methods, parameters, return types, and internal variables.
5. **No `eval()`/`ast`**: Pure algorithmic parsing ensures security and full control over evaluation logic.