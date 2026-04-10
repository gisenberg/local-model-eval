import re
from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    Supports basic arithmetic, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self._tokens: List[Tuple[str, Union[str, float]]] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: The mathematical expression string to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expressions")

        self._tokens = self._tokenize(expr)
        if not self._tokens:
            raise ValueError("Empty expressions")

        self._pos = 0
        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError("Mismatched parentheses")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, Union[str, float]]]:
        """
        Convert the input string into a list of tokens.

        Args:
            expr: The expression string.

        Returns:
            A list of tuples representing tokens (type, value).

        Raises:
            ValueError: If the expression contains invalid characters.
        """
        # Regex to match numbers (integers and floats) and operators/parens
        # Allows for formats like 3, 3.14, .5, 3.
        number_pattern = r'\d+\.\d*|\d*\.\d+|\d+'
        operator_pattern = r'[+\-*/()]'
        valid_pattern = rf'({number_pattern}|{operator_pattern})'

        # Check for invalid characters first
        if not re.fullmatch(rf'[\s{valid_pattern}]*', expr):
            raise ValueError("Invalid tokens")

        tokens = []
        for match in re.finditer(rf'{number_pattern}|{operator_pattern}', expr):
            value = match.group()
            if re.match(number_pattern, value):
                tokens.append(('NUM', float(value)))
            else:
                tokens.append(('OP', value))

        return tokens

    def _current_token(self) -> Tuple[str, Union[str, float]]:
        """
        Get the current token without advancing the position.

        Returns:
            The current token tuple.

        Raises:
            ValueError: If there are no more tokens.
        """
        if self._pos >= len(self._tokens):
            raise ValueError("Mismatched parentheses")
        return self._tokens[self._pos]

    def _consume(self) -> Tuple[str, Union[str, float]]:
        """
        Consume the current token and advance the position.

        Returns:
            The consumed token tuple.

        Raises:
            ValueError: If there are no more tokens.
        """
        token = self._current_token()
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parse an expression (handles + and -).

        Returns:
            The result of the expression evaluation.
        """
        value = self._parse_term()

        while True:
            token = self._current_token()
            if token[1] == '+':
                self._consume()
                value += self._parse_term()
            elif token[1] == '-':
                self._consume()
                value -= self._parse_term()
            else:
                break

        return value

    def _parse_term(self) -> float:
        """
        Parse a term (handles * and /).

        Returns:
            The result of the term evaluation.
        """
        value = self._parse_factor()

        while True:
            token = self._current_token()
            if token[1] == '*':
                self._consume()
                value *= self._parse_factor()
            elif token[1] == '/':
                self._consume()
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                value /= divisor
            else:
                break

        return value

    def _parse_factor(self) -> float:
        """
        Parse a factor (handles numbers, parentheses, and unary operators).

        Returns:
            The result of the factor evaluation.
        """
        token = self._current_token()

        # Handle unary minus or plus
        if token[1] == '-':
            self._consume()
            return -self._parse_factor()
        elif token[1] == '+':
            self._consume()
            return self._parse_factor()

        # Handle parentheses
        if token[1] == '(':
            self._consume()
            value = self._parse_expression()
            # Expect closing parenthesis
            close_token = self._current_token()
            if close_token[1] != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()
            return value

        # Handle numbers
        if token[0] == 'NUM':
            self._consume()
            return token[1]

        # If we reach here, it's an unexpected token
        raise ValueError("Invalid tokens")

import pytest


@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    """Test basic addition and subtraction."""
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("5 - 3") == 2.0
    assert evaluator.evaluate("10 + 20 - 5") == 25.0

def test_precedence(evaluator):
    """Test operator precedence (* and / before + and -)."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses(evaluator):
    """Test parentheses for grouping."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    assert evaluator.evaluate("((1 + 2) * 3)") == 9.0

def test_unary_minus(evaluator):
    """Test unary minus support."""
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("- (2 + 1)") == -3.0
    assert evaluator.evaluate("3 * -2") == -6.0
    assert evaluator.evaluate("- -5") == 5.0

def test_error_cases(evaluator):
    """Test error handling for invalid inputs."""
    # Empty expression
    with pytest.raises(ValueError, match="Empty expressions"):
        evaluator.evaluate("")
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid tokens"):
        evaluator.evaluate("1 @ 2")