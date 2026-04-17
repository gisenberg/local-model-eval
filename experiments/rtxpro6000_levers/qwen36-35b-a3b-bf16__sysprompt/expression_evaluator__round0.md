Here is a complete, production-ready implementation following all your requirements. It uses a classic recursive descent parser with explicit grammar rules, comprehensive type hints, docstrings, and targeted error handling.

### Implementation

```python
from typing import List, Tuple

class ExpressionEvaluator:
    """Evaluates mathematical expressions using a recursive descent parser."""

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string.

        Args:
            expr: The mathematical expression to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or expr.strip() == '':
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        self._tokens = tokens
        self._pos = 0

        result = self._parse_expression()

        if self._current_token()[0] != 'EOF':
            raise ValueError(f"Unexpected token after expression: {self._current_token()}")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, str]]:
        """Converts the expression string into a list of (type, value) tokens."""
        tokens: List[Tuple[str, str]] = []
        i = 0
        while i < len(expr):
            if expr[i].isspace():
                i += 1
                continue
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                has_dot = False
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format: multiple decimal points")
                        has_dot = True
                    j += 1
                tokens.append(('NUMBER', expr[i:j]))
                i = j
            elif expr[i] in '+-*/':
                tokens.append(('OP', expr[i]))
                i += 1
            elif expr[i] == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif expr[i] == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")
        tokens.append(('EOF', ''))
        return tokens

    def _current_token(self) -> Tuple[str, str]:
        """Returns the current token without advancing the parser position."""
        return self._tokens[self._pos]

    def _advance(self) -> Tuple[str, str]:
        """Advances the parser position and returns the current token."""
        token = self._current_token()
        if self._pos < len(self._tokens) - 1:
            self._pos += 1
        return token

    def _expect(self, token_type: str, token_value: str = None) -> Tuple[str, str]:
        """
        Checks if the current token matches the expected type and value.
        Raises ValueError if it does not match.
        """
        tok = self._current_token()
        if tok[0] != token_type or (token_value is not None and tok[1] != token_value):
            raise ValueError(f"Expected {token_type} '{token_value}', got {tok}")
        return self._advance()

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Grammar: Expression -> Term (('+' | '-') Term)*
        """
        result = self._parse_term()
        while self._current_token()[0] == 'OP' and self._current_token()[1] in '+-':
            op = self._advance()[1]
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Grammar: Term -> Factor (('*' | '/') Factor)*
        """
        result = self._parse_factor()
        while self._current_token()[0] == 'OP' and self._current_token()[1] in '*/':
            op = self._advance()[1]
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """
        Parses numbers, parentheses, and unary minus.
        Grammar: Factor -> '-' Factor | '(' Expression ')' | Number
        """
        tok = self._current_token()
        if tok[0] == 'OP' and tok[1] == '-':
            self._advance()
            return -self._parse_factor()
        if tok[0] == 'LPAREN':
            self._advance()
            result = self._parse_expression()
            self._expect('RPAREN', ')')
            return result
        if tok[0] == 'NUMBER':
            self._advance()
            return float(tok[1])
        raise ValueError(f"Unexpected token: {tok}")
```

### Pytest Tests

```python
import pytest

class TestExpressionEvaluator:
    def test_basic_arithmetic(self):
        """Tests fundamental operations with single operators."""
        evaluator = ExpressionEvaluator()
        assert evaluator.evaluate("2 + 3") == 5.0
        assert evaluator.evaluate("10 - 4") == 6.0
        assert evaluator.evaluate("3 * 4") == 12.0
        assert evaluator.evaluate("15 / 3") == 5.0

    def test_operator_precedence(self):
        """Tests that * and / bind tighter than + and -, and left-associativity."""
        evaluator = ExpressionEvaluator()
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        assert evaluator.evaluate("10 - 2 * 3") == 4.0
        assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
        assert evaluator.evaluate("10 / 2 / 5") == 1.0

    def test_parentheses(self):
        """Tests grouping and nested parentheses."""
        evaluator = ExpressionEvaluator()
        assert evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
        assert evaluator.evaluate("10 - (2 + 3)") == 5.0

    def test_unary_minus(self):
        """Tests unary minus at start, inside parens, and chained."""
        evaluator = ExpressionEvaluator()
        assert evaluator.evaluate("-5") == -5.0
        assert evaluator.evaluate("-(2 + 3)") == -5.0
        assert evaluator.evaluate("10 + -3") == 7.0
        assert evaluator.evaluate("-(-5)") == 5.0

    def test_error_cases(self):
        """Tests ValueError raising for invalid inputs."""
        evaluator = ExpressionEvaluator()
        
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")
            
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("10 / 0")
            
        with pytest.raises(ValueError):
            evaluator.evaluate("(2 + 3")  # Mismatched parentheses
            
        with pytest.raises(ValueError):
            evaluator.evaluate("2 @ 3")   # Invalid token
            
        with pytest.raises(ValueError):
            evaluator.evaluate("2 3")     # Missing operator / unexpected token
```

### Key Design Decisions & Invariants Preserved
1. **Recursive Descent Structure**: The parser directly mirrors the standard arithmetic grammar (`Expression → Term → Factor`). This guarantees correct operator precedence and left-associativity without needing shunting-yard or AST conversion.
2. **Stateless Public API**: `evaluate()` resets internal parser state (`_tokens`, `_pos`) on each call, making the class safe for repeated invocations.
3. **Tokenization Boundary**: The tokenizer explicitly validates number formats (rejects `..` or `1.2.3`) and rejects unknown characters early, providing clear error messages.
4. **Division by Zero**: Checked at parse time against `0.0`, which safely catches both `0` and `-0.0` in Python.
5. **Parentheses Validation**: Mismatched opening/closing parentheses are caught either by `_expect('RPAREN')` failing or by `_parse_factor` encountering an unexpected token type, both raising descriptive `ValueError`s.
6. **Type Safety**: All public and private methods include precise `typing` annotations. No dynamic typing or `eval()` is used.