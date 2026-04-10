import re
from typing import List, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser
    to evaluate arithmetic expressions with operator precedence and unary operators.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates the given mathematical expression string.

        Args:
            expr: The string expression to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens, 
                       has mismatched parentheses, or results in division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError("Expression contains no valid tokens.")

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at end of expression: '{self._tokens[self._pos]}'")

        return float(result)

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of meaningful tokens using regex.

        Args:
            expr: The raw input string.

        Returns:
            A list of strings representing numbers, operators, and parentheses.
        """
        # Pattern matches floating point numbers, operators, and parentheses
        token_pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        tokens = re.findall(token_pattern, expr)
        
        # Validate that the reconstructed string from tokens matches the non-whitespace input
        # This ensures no invalid characters (like 'a', '$', etc.) were skipped
        cleaned_expr = re.sub(r'\s+', '', expr)
        reconstructed = "".join(tokens)
        
        # Check if there are characters in the input that weren't captured by the pattern
        # We strip whitespace to compare the "content"
        if len(re.sub(r'\s+', '', expr)) != len(reconstructed):
            # Find the first invalid character for a better error message
            for char in expr:
                if not char.isspace() and not re.match(token_pattern, char):
                    raise ValueError(f"Invalid token detected: '{char}'")
            raise ValueError("Expression contains invalid tokens.")

        return tokens

    def _peek(self) -> Union[str, None]:
        """Returns the current token without advancing the pointer."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Returns the current token and advances the pointer."""
        token = self._peek()
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Handles addition and subtraction (lowest precedence).
        Grammar: expression -> term { (+|-) term }
        """
        node = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                node += right
            else:
                node -= right
        return node

    def _parse_term(self) -> float:
        """
        Handles multiplication and division.
        Grammar: term -> factor { (*|/) factor }
        """
        node = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero.")
                node /= right
        return node

    def _parse_factor(self) -> float:
        """
        Handles unary minus and parentheses.
        Grammar: factor -> '-' factor | primary
        """
        token = self._peek()

        if token == '-':
            self._consume()
            return -self._parse_factor()
        
        if token == '+':
            self._consume()
            return self._parse_factor()

        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Handles numbers and grouped expressions.
        Grammar: primary -> number | '(' expression ')'
        """
        token = self._peek()

        if token == '(':
            self._consume()
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._consume()  # consume ')'
            return result

        if token is not None and (token[0].isdigit() or (len(token) > 1 and token[1].isdigit())):
            return float(self._consume())
        
        if token == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")
            
        raise ValueError(f"Unexpected token: '{token if token else 'EOF'}'")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("15 / 4") == 3.75

def test_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0

def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_errors(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3")