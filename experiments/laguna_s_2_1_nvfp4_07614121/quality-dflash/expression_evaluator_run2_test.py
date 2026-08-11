import re

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic operations
    with correct operator precedence, parentheses, unary minus, and floating-point numbers.

    Uses a recursive descent parser. Does not use eval() or ast.literal_eval().

    Supported Operators:
        + (addition)
        - (subtraction / unary minus)
        * (multiplication)
        / (division)

    Supported Features:
        - Parentheses for grouping
        - Floating point numbers (e.g., 3.14)
        - Unary minus (e.g., -3, -(2+1))

    Raises:
        ValueError: For mismatched parentheses, division by zero,
                    invalid tokens, or empty expressions.
    """

    def __init__(self):
        """Initializes the evaluator."""
        self.pos = 0
        self.tokens = []

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression.

        Args:
            expr: A string containing the mathematical expression to evaluate.

        Returns:
            The result of the evaluated expression as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or results in division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")

        # Tokenize the expression
        self._tokenize(expr)
        self.pos = 0

        try:
            result = self._parse_expression()
            if self.pos != len(self.tokens):
                raise ValueError("Unexpected token(s) after end of expression.")
            return result
        except IndexError:
            raise ValueError("Unexpected end of expression.")

    def _tokenize(self, expr: str) -> None:
        """
        Converts the input string into a list of tokens.

        Args:
            expr: The input string to tokenize.

        Raises:
            ValueError: If an invalid token is encountered.
        """
        self.tokens = []
        i = 0
        while i < len(expr):
            char = expr[i]

            # Skip whitespace
            if char.isspace():
                i += 1
                continue

            # Number (including floats)
            if char.isdigit() or char == '.':
                match = re.match(r'\d+\.?\d*', expr[i:])
                if match:
                    self.tokens.append(float(match.group()))
                    i += match.end()
                    continue
                else:
                    raise ValueError(f"Invalid number format at position {i}.")

            # Single-character tokens
            if char in '+-*/()':
                self.tokens.append(char)
                i += 1
                continue

            # Invalid character
            raise ValueError(f"Invalid token '{char}' at position {i}.")

    def _peek(self) -> object:
        """Returns the current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self, expected: object) -> None:
        """
        Consumes the current token if it matches the expected value.

        Args:
            expected: The expected token.

        Raises:
            ValueError: If the current token does not match the expected value.
        """
        if self._peek() == expected:
            self.pos += 1
        else:
            raise ValueError(f"Expected '{expected}', but found '{self._peek()}'.")

    def _parse_expression(self) -> float:
        """Parses an expression: term (('+' | '-') term)*"""
        result = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._peek()
            self.pos += 1
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parses a term: factor (('*' | '/') factor)*"""
        result = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._peek()
            self.pos += 1
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero.")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parses a factor: ('+' | '-') factor | power"""
        if self._peek() in ('+', '-'):
            op = self._peek()
            self.pos += 1
            operand = self._parse_factor()
            return operand if op == '+' else -operand
        return self._parse_power()

    def _parse_power(self) -> float:
        """Parses a power: primary. Currently a pass-through for extensibility."""
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parses a primary: NUMBER | '(' expression ')'"""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression.")

        if isinstance(token, float):
            self.pos += 1
            return token

        if token == '(':
            self.pos += 1
            result = self._parse_expression()
            self._consume(')')
            return result

        raise ValueError(f"Unexpected token '{token}'.")

import pytest

def test_simple_operations():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 2") == 5.0

def test_operator_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -4") == -8.0

def test_floating_point():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 + 1.0") == 4.14
    assert evaluator.evaluate("2.5 * 2") == 5.0
    assert evaluator.evaluate("1.5 / 0.5") == 3.0

def test_errors():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 +")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3") # Invalid token