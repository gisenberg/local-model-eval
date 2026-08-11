import re
from typing import List, Union


class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic operations,
    parentheses, and floating-point numbers.

    This implementation uses a recursive descent parser and does not rely on
    `eval()` or `ast.literal_eval()`.

    Supported Operators:
        + : Addition
        - : Subtraction / Unary Minus
        * : Multiplication
        / : Division

    Examples:
        >>> evaluator = ExpressionEvaluator()
        >>> evaluator.evaluate("2 + 3 * 4")
        14.0
        >>> evaluator.evaluate("(2 + 3) * 4")
        20.0
        >>> evaluator.evaluate("-3.5")
        -3.5
        >>> evaluator.evaluate("10 / 0")
        Traceback (most recent call last):
            ...
        ValueError: Division by zero.
    """

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.pos: int = 0

    def tokenize(self, expr: str) -> List[str]:
        """
        Converts an expression string into a list of tokens.

        Args:
            expr: The mathematical expression string.

        Returns:
            A list of tokens (numbers, operators, parentheses).

        Raises:
            ValueError: If the expression contains invalid characters or is empty.
        """
        if not expr.strip():
            raise ValueError("Empty expression.")

        # Regular expression to match numbers, operators, and parentheses.
        # Numbers can be integers or floats.
        token_pattern = r'[-+*/()/*]|\d+\.?\d*|\.\d+'
        tokens = re.findall(token_pattern, expr)

        # Check for invalid tokens
        valid_token_pattern = r'^[-+*/()]|\d+\.?\d*|\.\d+$'
        for token in tokens:
            if not re.match(valid_token_pattern, token):
                raise ValueError(f"Invalid token: '{token}'")

        # Handle unary minus by distinguishing it from subtraction
        processed_tokens = []
        for i, token in enumerate(tokens):
            if token == '-':
                # It's a unary minus if it's the first token,
                # or the previous token is an operator or open parenthesis.
                if i == 0 or tokens[i-1] in ('+', '-', '*', '/', '('):
                    processed_tokens.append('u-')
                else:
                    processed_tokens.append('-')
            elif token == '+':
                # Similarly handle unary plus (though it has no effect)
                if i == 0 or tokens[i-1] in ('+', '-', '*', '/', '('):
                    processed_tokens.append('u+')
                else:
                    processed_tokens.append('+')
            else:
                processed_tokens.append(token)

        return processed_tokens

    def parse_number(self) -> float:
        """Parses a number token and advances the position."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression.")
        token = self.tokens[self.pos]
        try:
            num = float(token)
        except ValueError:
            raise ValueError(f"Expected a number but found '{token}'.")
        self.pos += 1
        return num

    def parse_factor(self) -> float:
        """
        Parses a factor, which can be a number, a parenthesized expression,
        or a unary operation.
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression.")

        token = self.tokens[self.pos]

        if token == '(':
            self.pos += 1
            result = self.parse_expression()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses.")
            self.pos += 1
            return result
        elif token == 'u-':
            self.pos += 1
            return -self.parse_factor()
        elif token == 'u+':
            self.pos += 1
            return self.parse_factor()
        else:
            return self.parse_number()

    def parse_term(self) -> float:
        """Parses a term, handling multiplication and division."""
        result = self.parse_factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.parse_factor()
            if op == '*':
                result *= right
            elif op == '/':
                if right == 0:
                    raise ValueError("Division by zero.")
                result /= right
        return result

    def parse_expression(self) -> float:
        """Parses an expression, handling addition and subtraction."""
        result = self.parse_term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.parse_term()
            if op == '+':
                result += right
            elif op == '-':
                result -= right
        return result

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression.

        Args:
            expr: The mathematical expression string.

        Returns:
            The result of the evaluated expression as a float.

        Raises:
            ValueError: For various parsing errors including mismatched parentheses,
                        division by zero, invalid tokens, and empty expressions.
        """
        self.tokens = self.tokenize(expr)
        self.pos = 0
        result = self.parse_expression()
        if self.pos != len(self.tokens):
            raise ValueError("Unexpected tokens at the end of expression.")
        return result


# --- Pytest Tests ---

def test_simple_operations():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 2") == 5.0

def test_operator_precedence():
    """Test that operator precedence is respected."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 6 / 2") == 7.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0

def test_unary_minus():
    """Test unary minus functionality."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-3.5") == -3.5
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -4") == -8.0

def test_floating_point_numbers():
    """Test support for floating-point numbers."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14") == 3.14
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("2.5 * 2.0") == 5.0

def test_error_handling():
    """Test that appropriate ValueErrors are raised for invalid input."""
    evaluator = ExpressionEvaluator()

    # Mismatched parentheses
    try:
        evaluator.evaluate("(2 + 3")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Mismatched parentheses" in str(e)

    # Division by zero
    try:
        evaluator.evaluate("10 / 0")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Division by zero" in str(e)

    # Invalid token
    try:
        evaluator.evaluate("2 + abc")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid token" in str(e)

    # Empty expression
    try:
        evaluator.evaluate("")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Empty expression" in str(e)

    # Unexpected end of expression
    try:
        evaluator.evaluate("2 +")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unexpected end of expression" in str(e)