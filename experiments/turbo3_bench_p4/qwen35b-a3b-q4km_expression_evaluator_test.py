from typing import List, Tuple, Optional
import pytest


class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        self.tokens: List[Tuple[str, Optional[float]]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.

        Args:
            expr: The mathematical expression string to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        if not self.tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError("Mismatched parentheses or extra tokens")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, Optional[float]]]:
        """
        Converts the input string into a list of tokens.

        Args:
            expr: The input expression string.

        Returns:
            A list of tuples (token_type, value).

        Raises:
            ValueError: If an invalid character is encountered.
        """
        tokens = []
        i = 0
        length = len(expr)

        while i < length:
            char = expr[i]

            if char.isspace():
                i += 1
                continue

            if char in "+-*/()":
                tokens.append((char, None))
                i += 1
            elif char.isdigit() or char == '.':
                # Parse number
                num_str = ""
                dot_count = 0
                while i < length and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        dot_count += 1
                        if dot_count > 1:
                            raise ValueError("Invalid number format: multiple decimal points")
                    num_str += expr[i]
                    i += 1
                if not num_str:
                    raise ValueError("Invalid number format")
                if num_str == '.' or num_str.endswith('.'):
                    raise ValueError("Invalid number format")
                tokens.append(('NUM', float(num_str)))
            else:
                raise ValueError(f"Invalid token: '{char}'")

        return tokens

    def _current_token(self) -> Tuple[str, Optional[float]]:
        """
        Returns the current token without advancing the position.

        Returns:
            The current token tuple.

        Raises:
            ValueError: If there are no more tokens.
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        return self.tokens[self.pos]

    def _advance(self) -> Tuple[str, Optional[float]]:
        """
        Returns the current token and advances the position.

        Returns:
            The current token tuple.
        """
        token = self._current_token()
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).

        Returns:
            The result of the expression.
        """
        left = self._parse_term()

        while True:
            token = self._current_token()
            if token[0] == '+':
                self._advance()
                right = self._parse_term()
                left = left + right
            elif token[0] == '-':
                self._advance()
                right = self._parse_term()
                left = left - right
            else:
                break

        return left

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).

        Returns:
            The result of the term.
        """
        left = self._parse_factor()

        while True:
            token = self._current_token()
            if token[0] == '*':
                self._advance()
                right = self._parse_factor()
                left = left * right
            elif token[0] == '/':
                self._advance()
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
            else:
                break

        return left

    def _parse_factor(self) -> float:
        """
        Parses numbers, parentheses, and unary minus (highest precedence).

        Returns:
            The result of the factor.
        """
        token = self._current_token()

        # Handle unary minus
        if token[0] == '-':
            self._advance()
            return -self._parse_factor()

        # Handle parentheses
        if token[0] == '(':
            self._advance()
            result = self._parse_expression()
            token = self._current_token()
            if token[0] != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result

        # Handle numbers
        if token[0] == 'NUM':
            self._advance()
            return token[1]

        # Unexpected token
        raise ValueError(f"Unexpected token: {token[0]}")


# --- Pytest Tests ---

def test_basic_arithmetic():
    """Test basic addition, subtraction, multiplication, and division."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 1") == 2.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 2") == 5.0
    assert evaluator.evaluate("1.5 + 2.5") == 4.0


def test_operator_precedence():
    """Test that multiplication and division happen before addition and subtraction."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    assert evaluator.evaluate("10 / 2 / 5") == 1.0


def test_parentheses_grouping():
    """Test that parentheses override standard precedence."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(1 + 2) * 3") == 9.0
    assert evaluator.evaluate("10 - (2 * 3)") == 4.0
    assert evaluator.evaluate("((1 + 2) * 3)") == 9.0


def test_unary_minus():
    """Test unary minus support."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("1 + -5") == -4.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-3 * 2") == -6.0


def test_error_cases():
    """Test error handling for invalid inputs."""
    evaluator = ExpressionEvaluator()

    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("1 + 2)")

    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1 @ 2")

    # Unexpected end of expression
    with pytest.raises(ValueError, match="Unexpected end of expression"):
        evaluator.evaluate("1 +")