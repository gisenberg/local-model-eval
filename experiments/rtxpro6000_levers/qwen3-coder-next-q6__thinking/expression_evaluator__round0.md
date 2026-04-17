Here's a complete implementation of a mathematical expression evaluator using a recursive descent parser, meeting all your requirements.

---

### ✅ `expression_evaluator.py`

```python
from __future__ import annotations
from typing import List, Union, Tuple
import re


class ExpressionEvaluator:
    """
    A recursive descent parser-based mathematical expression evaluator.
    Supports +, -, *, /, parentheses, unary minus, and floating-point numbers.
    Raises ValueError for invalid input or runtime errors.
    """

    # Token types
    NUMBER = 'NUMBER'
    PLUS = 'PLUS'
    MINUS = 'MINUS'
    MULTIPLY = 'MULTIPLY'
    DIVIDE = 'DIVIDE'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    EOF = 'EOF'

    def __init__(self) -> None:
        self.tokens: List[Tuple[str, Union[str, float]]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr (str): The expression to evaluate.

        Returns:
            float: The result of the evaluation.

        Raises:
            ValueError: If the expression is invalid (e.g., mismatched parens, division by zero, invalid tokens).
        """
        if not expr or expr.strip() == "":
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        result = self._parse_expression()

        if self.pos < len(self.tokens):
            token_type, token_value = self.tokens[self.pos]
            raise ValueError(f"Unexpected token '{token_value}' at position {self.pos}")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, Union[str, float]]]:
        """
        Convert the input expression string into a list of tokens.

        Returns:
            List[Tuple[str, Union[str, float]]]: List of (token_type, value) tuples.
        """
        tokens = []
        i = 0
        while i < len(expr):
            ch = expr[i]

            if ch.isspace():
                i += 1
                continue

            # Number (including floats)
            if ch.isdigit() or (ch == '.' and i + 1 < len(expr) and expr[i + 1].isdigit()):
                j = i
                has_dot = False
                while j < len(expr) and (expr[j].isdigit() or (expr[j] == '.' and not has_dot)):
                    if expr[j] == '.':
                        has_dot = True
                    j += 1
                num_str = expr[i:j]
                try:
                    num = float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number: '{num_str}'")
                tokens.append((self.NUMBER, num))
                i = j
                continue

            # Operators and parentheses
            if ch == '+':
                tokens.append((self.PLUS, '+'))
            elif ch == '-':
                tokens.append((self.MINUS, '-'))
            elif ch == '*':
                tokens.append((self.MULTIPLY, '*'))
            elif ch == '/':
                tokens.append((self.DIVIDE, '/'))
            elif ch == '(':
                tokens.append((self.LPAREN, '('))
            elif ch == ')':
                tokens.append((self.RPAREN, ')'))
            else:
                raise ValueError(f"Invalid character: '{ch}'")

            i += 1

        if not tokens:
            raise ValueError("Empty expression")

        # Check for mismatched parentheses
        paren_count = 0
        for token_type, _ in tokens:
            if token_type == self.LPAREN:
                paren_count += 1
            elif token_type == self.RPAREN:
                paren_count -= 1
                if paren_count < 0:
                    raise ValueError("Mismatched parentheses: extra closing parenthesis")
        if paren_count != 0:
            raise ValueError("Mismatched parentheses: unclosed parenthesis")

        return tokens

    def _current_token(self) -> Tuple[str, Union[str, float]]:
        """Return the current token or (EOF, None) if at end."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (self.EOF, None)

    def _advance(self) -> Tuple[str, Union[str, float]]:
        """Advance to next token and return the previous one."""
        token = self._current_token()
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parse expression: term (('+' | '-') term)*
        Handles addition and subtraction (lowest precedence).
        """
        result = self._parse_term()

        while self._current_token()[0] in (self.PLUS, self.MINUS):
            op, _ = self._advance()
            right = self._parse_term()
            if op == self.PLUS:
                result += right
            else:
                result -= right

        return result

    def _parse_term(self) -> float:
        """
        Parse term: factor (('*' | '/') factor)*
        Handles multiplication and division (higher precedence).
        """
        result = self._parse_factor()

        while self._current_token()[0] in (self.MULTIPLY, self.DIVIDE):
            op, _ = self._advance()
            right = self._parse_factor()
            if op == self.MULTIPLY:
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right

        return result

    def _parse_factor(self) -> float:
        """
        Parse factor: ('+' | '-')? factor | primary
        Handles unary plus/minus and primary expressions (numbers, parentheses).
        """
        token_type, token_value = self._current_token()

        # Unary plus/minus
        if token_type == self.PLUS:
            self._advance()
            return self._parse_factor()
        elif token_type == self.MINUS:
            self._advance()
            return -self._parse_factor()

        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parse primary: NUMBER | '(' expression ')'
        Handles numbers and parenthesized expressions.
        """
        token_type, token_value = self._current_token()

        if token_type == self.NUMBER:
            self._advance()
            return token_value

        if token_type == self.LPAREN:
            self._advance()  # consume '('
            result = self._parse_expression()
            if self._current_token()[0] != self.RPAREN:
                raise ValueError("Mismatched parentheses: expected closing parenthesis")
            self._advance()  # consume ')'
            return result

        # Unexpected token
        if token_type == self.EOF:
            raise ValueError("Unexpected end of expression")
        else:
            raise ValueError(f"Unexpected token '{token_value}'")
```

---

### ✅ `test_expression_evaluator.py`

```python
import pytest
from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator():
    return ExpressionEvaluator()


class TestExpressionEvaluator:
    def test_basic_arithmetic(self, evaluator):
        assert evaluator.evaluate("2 + 3") == 5.0
        assert evaluator.evaluate("10 - 4") == 6.0
        assert evaluator.evaluate("6 * 7") == 42.0
        assert evaluator.evaluate("15 / 3") == 5.0

    def test_operator_precedence(self, evaluator):
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        assert evaluator.evaluate("10 - 2 * 3") == 4.0
        assert evaluator.evaluate("8 / 2 + 3") == 7.0

    def test_parentheses(self, evaluator):
        assert evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert evaluator.evaluate("((1 + 2) * 3) / 2") == 4.5
        assert evaluator.evaluate("2 * (3 + 4)") == 14.0

    def test_unary_minus(self, evaluator):
        assert evaluator.evaluate("-3") == -3.0
        assert evaluator.evaluate("-(2 + 3)") == -5.0
        assert evaluator.evaluate("5 + -2") == 3.0
        assert evaluator.evaluate("-( -3 )") == 3.0
        assert evaluator.evaluate("3 * -2") == -6.0

    def test_error_cases(self, evaluator):
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
            evaluator.evaluate("5 / 0")

        # Invalid token
        with pytest.raises(ValueError, match="Invalid character"):
            evaluator.evaluate("2 + a")

        # Unexpected token
        with pytest.raises(ValueError, match="Unexpected token"):
            evaluator.evaluate("2 + * 3")

        # Unclosed parenthesis
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("((1 + 2)")

        # Extra closing parenthesis
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("1 + 2))")
```

---

### ✅ How to Run

1. Save the first file as `expression_evaluator.py`
2. Save the second as `test_expression_evaluator.py`
3. Install pytest if needed: `pip install pytest`
4. Run tests: `pytest test_expression_evaluator.py -v`

Example output:
```
test_expression_evaluator.py::TestExpressionEvaluator::test_basic_arithmetic PASSED
test_expression_evaluator.py::TestExpressionEvaluator::test_operator_precedence PASSED
test_expression_evaluator.py::TestExpressionEvaluator::test_parentheses PASSED
test_expression_evaluator.py::TestExpressionEvaluator::test_unary_minus PASSED
test_expression_evaluator.py::TestExpressionEvaluator::test_error_cases PASSED
```

Let me know if you'd like to extend it with exponentiation (`^`), functions (`sin`, `cos`, etc.), or variables!