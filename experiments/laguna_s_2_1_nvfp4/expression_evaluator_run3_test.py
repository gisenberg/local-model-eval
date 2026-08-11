import re
from typing import List, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic operations,
    parentheses, unary minus, and floating-point numbers.

    Supported operators: +, -, *, /
    Supported features: parentheses for grouping, unary minus, floating-point numbers

    Raises:
        ValueError: For mismatched parentheses, division by zero, invalid tokens, or empty expressions
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.

        Args:
            expr: A string containing a mathematical expression

        Returns:
            The result of evaluating the expression as a float

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                       has mismatched parentheses, or involves division by zero
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        # Tokenize the expression
        self.tokens = self._tokenize(expr)
        self.pos = 0

        # Parse and evaluate
        result = self._parse_expression()

        # Check if all tokens were consumed
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert an expression string into a list of tokens.

        Args:
            expr: The expression string to tokenize

        Returns:
            A list of tokens (numbers, operators, parentheses)

        Raises:
            ValueError: If the expression contains invalid tokens
        """
        # Regular expression to match valid tokens
        token_pattern = r'(\d+\.?\d*|\.\d+|[+\-*/()])'
        tokens = re.findall(token_pattern, expr)

        # Check for invalid characters
        valid_chars = set('0123456789+-*/(). ')
        for char in expr:
            if char not in valid_chars:
                raise ValueError(f"Invalid character: {char}")

        # Check if we got any tokens
        if not tokens:
            raise ValueError("No valid tokens found")

        return tokens

    def _parse_expression(self) -> float:
        """
        Parse and evaluate an expression (handles + and -).
        Expression -> Term (('+' | '-') Term)*

        Returns:
            The result of the expression as a float
        """
        result = self._parse_term()

        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()

            if op == '+':
                result += right
            else:  # op == '-'
                result -= right

        return result

    def _parse_term(self) -> float:
        """
        Parse and evaluate a term (handles * and /).
        Term -> Factor (('*' | '/') Factor)*

        Returns:
            The result of the term as a float
        """
        result = self._parse_factor()

        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()

            if op == '*':
                result *= right
            else:  # op == '/'
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right

        return result

    def _parse_factor(self) -> float:
        """
        Parse and evaluate a factor (handles unary minus and parentheses).
        Factor -> ('-' | '+') Factor | Number | '(' Expression ')'

        Returns:
            The result of the factor as a float
        """
        # Handle unary minus and plus
        if self.pos < len(self.tokens) and self.tokens[self.pos] in ('-', '+'):
            op = self.tokens[self.pos]
            self.pos += 1
            operand = self._parse_factor()
            return -operand if op == '-' else operand

        # Handle parentheses
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '(':
            self.pos += 1  # Skip '('
            result = self._parse_expression()

            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")

            self.pos += 1  # Skip ')'
            return result

        # Handle number
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            try:
                self.pos += 1
                return float(token)
            except ValueError:
                raise ValueError(f"Invalid token: {token}")

        raise ValueError("Unexpected end of expression")

# Pytest tests
def test_basic_operations():
    """Test basic arithmetic operations with correct precedence."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("15 / 3") == 5.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # Precedence test
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0  # Parentheses test

def test_unary_minus():
    """Test unary minus operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-3.14") == -3.14
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0
    assert evaluator.evaluate("-(-5)") == 5.0

def test_floating_point():
    """Test floating point number support."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14") == 3.14
    assert evaluator.evaluate("0.5 + 0.25") == 0.75
    assert evaluator.evaluate("2.5 * 4") == 10.0
    assert abs(evaluator.evaluate("1.1 + 2.2") - 3.3) < 1e-10  # Floating point precision

def test_error_handling():
    """Test error handling for invalid inputs."""
    evaluator = ExpressionEvaluator()

    # Empty expression
    try:
        evaluator.evaluate("")
        assert False, "Should raise ValueError for empty expression"
    except ValueError:
        pass

    # Division by zero
    try:
        evaluator.evaluate("5 / 0")
        assert False, "Should raise ValueError for division by zero"
    except ValueError:
        pass

    # Mismatched parentheses
    try:
        evaluator.evaluate("(2 + 3")
        assert False, "Should raise ValueError for mismatched parentheses"
    except ValueError:
        pass

    # Invalid token
    try:
        evaluator.evaluate("2 + abc")
        assert False, "Should raise ValueError for invalid tokens"
    except ValueError:
        pass

def test_complex_expressions():
    """Test complex nested expressions."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("((2 + 3) * 4 - 1) / 3") == 6.333333333333333
    assert evaluator.evaluate("2 * (3 + 4) * (5 - 2)") == 42.0
    assert evaluator.evaluate("-((2 + 3) * -2)") == 10.0
    assert abs(evaluator.evaluate("3.14 * 2.5 + 1.7") - 9.52) < 1e-10