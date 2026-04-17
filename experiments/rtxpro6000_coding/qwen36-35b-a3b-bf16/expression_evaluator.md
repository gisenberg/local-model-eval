Here's a complete, production-ready implementation following all your requirements. It uses a classic recursive descent parser with a clear grammar structure, proper error handling, and full type hints.

### `expression_evaluator.py`
```python
class ExpressionEvaluator:
    """A mathematical expression evaluator using a recursive descent parser."""

    def __init__(self) -> None:
        self._tokens: list[tuple[str, str]] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result as a float.
        
        Args:
            expr: The mathematical expression to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        if self._current_token()[0] != 'EOF':
            raise ValueError(f"Unexpected token after expression: '{self._current_token()[1]}'")

        return result

    def _tokenize(self, expr: str) -> list[tuple[str, str]]:
        """Convert the input string into a list of tokens.
        
        Args:
            expr: The raw expression string.
            
        Returns:
            A list of tuples representing tokens (type, value).
            
        Raises:
            ValueError: If an invalid character is encountered.
        """
        tokens: list[tuple[str, str]] = []
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
                tokens.append(('NUMBER', expr[i:j]))
                i = j
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")

        tokens.append(('EOF', ''))
        return tokens

    def _current_token(self) -> tuple[str, str]:
        """Return the current token without advancing the position."""
        return self._tokens[self._pos]

    def _advance(self) -> None:
        """Advance the parser position to the next token."""
        self._pos += 1

    def _parse_expression(self) -> float:
        """Parse addition and subtraction operations (lowest precedence).
        
        Returns:
            The result of the expression evaluation.
        """
        result = self._parse_term()

        while self._current_token()[0] in ('+', '-'):
            op = self._current_token()[0]
            self._advance()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right

        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division operations (higher precedence).
        
        Returns:
            The result of the term evaluation.
            
        Raises:
            ValueError: If division by zero is encountered.
        """
        result = self._parse_factor()

        while self._current_token()[0] in ('*', '/'):
            op = self._current_token()[0]
            self._advance()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right

        return result

    def _parse_factor(self) -> float:
        """Parse numbers, parenthesized expressions, and unary operators (highest precedence).
        
        Returns:
            The result of the factor evaluation.
            
        Raises:
            ValueError: If parentheses are mismatched or an unexpected token is found.
        """
        token = self._current_token()

        if token[0] == 'NUMBER':
            self._advance()
            return float(token[1])
        elif token[0] == '(':
            self._advance()
            result = self._parse_expression()
            if self._current_token()[0] != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._advance()
            return result
        elif token[0] in ('+', '-'):
            self._advance()
            val = self._parse_factor()
            return val if token[0] == '+' else -val
        else:
            raise ValueError(f"Unexpected token: '{token[1]}'")
```

### `test_expression_evaluator.py`
```python
import pytest
from expression_evaluator import ExpressionEvaluator

evaluator = ExpressionEvaluator()

def test_basic_arithmetic():
    """Test basic +, -, *, / operations."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0

def test_precedence():
    """Test that * and / bind tighter than + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3 + 1") == 5.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses():
    """Test grouping with parentheses."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == 15.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    """Test unary minus at start, after operators, and nested."""
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("10 + -3") == 7.0
    assert evaluator.evaluate("-(-5)") == 5.0

def test_error_cases():
    """Test ValueError raising for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
        
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
        
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")
```

### How It Works
1. **Grammar Mapping**: The parser directly implements a standard arithmetic grammar:
   - `expression` → handles `+` and `-`
   - `term` → handles `*` and `/`
   - `factor` → handles numbers, `()`, and unary `+`/`-`
   This structure naturally enforces operator precedence without needing explicit precedence tables.
2. **Tokenization**: A lightweight lexer scans the string, grouping digits/dots into `NUMBER` tokens and isolating operators/parentheses. Whitespace is skipped.
3. **Recursive Descent**: Each grammar rule maps to a method. Methods call each other recursively, consuming tokens and building the result bottom-up.
4. **Error Handling**: Invalid characters, division by zero, mismatched parentheses, and trailing tokens are caught with descriptive `ValueError` messages.
5. **No `eval()`/`ast`**: Pure algorithmic parsing ensures security and full control over behavior.

### Running the Tests
```bash
pip install pytest
pytest test_expression_evaluator.py -v
```
All 5 test suites will pass, covering your exact requirements.