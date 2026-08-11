```python
import pytest
from typing import List, Tuple


class ExpressionEvaluator:
    """
    A recursive descent parser and evaluator for arithmetic expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def tokenize(self, expr: str) -> List[str]:
        """
        Convert an expression string into a list of tokens.
        Raises ValueError for invalid characters or empty input.
        """
        tokens: List[str] = []
        i = 0
        n = len(expr)

        while i < n:
            c = expr[i]

            # Skip whitespace
            if c.isspace():
                i += 1
                continue

            # Parse numbers (digits and at most one dot)
            if c.isdigit() or c == '.':
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                num_str = expr[start:i]
                # Validate that it is a proper float (e.g., "1.2.3" will fail later)
                tokens.append(num_str)
                continue

            # Operators and parentheses
            if c in '+-*/()':
                tokens.append(c)
                i += 1
                continue

            # Invalid character
            raise ValueError(f"Invalid token '{c}' at position {i}")

        return tokens

    def parse_factor(self, tokens: List[str], index: int) -> Tuple[float, int]:
        """
        Parse a factor: a number, a parenthesized expression, or a unary signed factor.
        Returns (value, new_index).
        """
        if index >= len(tokens):
            raise ValueError("Unexpected end of expression")

        token = tokens[index]

        # Unary plus / minus
        if token == '+':
            value, index = self.parse_factor(tokens, index + 1)
            return value, index
        elif token == '-':
            value, index = self.parse_factor(tokens, index + 1)
            return -value, index

        # Parenthesized expression
        elif token == '(':
            value, index = self.parse_expression(tokens, index + 1)
            if index >= len(tokens) or tokens[index] != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            return value, index + 1

        # Unexpected closing parenthesis
        elif token == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")

        # Number token
        else:
            try:
                value = float(token)
            except ValueError:
                raise ValueError(f"Invalid number '{token}'")
            return value, index + 1

    def parse_term(self, tokens: List[str], index: int) -> Tuple[float, int]:
        """
        Parse a term: factors combined by '*' or '/'.
        Returns (value, new_index).
        """
        left, index = self.parse_factor(tokens, index)

        while index < len(tokens) and tokens[index] in ('*', '/'):
            op = tokens[index]
            right, index = self.parse_factor(tokens, index + 1)

            if op == '*':
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right

        return left, index

    def parse_expression(self, tokens: List[str], index: int) -> Tuple[float, int]:
        """
        Parse an expression: terms combined by '+' or '-'.
        Returns (value, new_index).
        """
        left, index = self.parse_term(tokens, index)

        while index < len(tokens) and tokens[index] in ('+', '-'):
            op = tokens[index]
            right, index = self.parse_term(tokens, index + 1)

            if op == '+':
                left = left + right
            else:
                left = left - right

        return left, index

    def evaluate(self, expr: str) -> float:
        """
        Evaluate an arithmetic expression and return its numeric value.
        Raises ValueError for empty expressions, mismatched parentheses,
        division by zero, invalid tokens, or malformed input.
        """
        tokens = self.tokenize(expr)

        if not tokens:
            raise ValueError("Empty expression")

        value, index = self.parse_expression(tokens, 0)

        # Ensure all tokens were consumed
        if index != len(tokens):
            raise ValueError(f"Unexpected token '{tokens[index]}' after expression")

        return value


# ---------- Pytest tests ----------

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2+3") == 5
    assert evaluator.evaluate("10-4") == 6
    assert evaluator.evaluate("6*7") == 42
    assert evaluator.evaluate("8/2") == 4.0


def test_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2+3*4") == 14
    assert evaluator.evaluate("10-2*3") == 4
    assert evaluator.evaluate("20/4+2") == 7
    assert evaluator.evaluate("2*3+4*5") == 26


def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2+3)*4") == 20
    assert evaluator.evaluate("2*(3+4)") == 14
    assert evaluator.evaluate("((2+3)*4)") == 20


def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3
    assert evaluator.evaluate("-(2+1)") == -3
    assert evaluator.evaluate("-2+3") == 1
    assert evaluator.evaluate("--5") == 5  # double negation


def test_error_cases():
    evaluator = ExpressionEvaluator()

    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")

    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2+3")

    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("2/0")

    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2+@3")

    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2+3)")
```