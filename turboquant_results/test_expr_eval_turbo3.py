from typing import List, Tuple, Union
import re


class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        """Initialize the evaluator."""
        pass

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is invalid, empty, has mismatched parentheses,
                        division by zero, or contains invalid tokens.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Empty expression")

        result, pos = self._parse_expression(tokens, 0)
        if pos != len(tokens):
            raise ValueError(f"Unexpected token at position {pos}: {tokens[pos]}")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert an expression string into a list of tokens.

        Args:
            expr: The expression string to tokenize.

        Returns:
            A list of tokens (numbers, operators, parentheses).

        Raises:
            ValueError: If invalid characters are found.
        """
        tokens = []
        i = 0
        while i < len(expr):
            char = expr[i]
            if char.isspace():
                i += 1
                continue
            elif char in "+-*/()":
                tokens.append(char)
                i += 1
            elif char.isdigit() or char == '.':
                start = i
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                num_str = expr[start:i]
                if num_str.count('.') > 1:
                    raise ValueError(f"Invalid number format: {num_str}")
                tokens.append(num_str)
            else:
                raise ValueError(f"Invalid character: '{char}'")
        return tokens

    def _parse_expression(self, tokens: List[str], pos: int) -> Tuple[float, int]:
        """
        Parse addition and subtraction (lowest precedence).

        Args:
            tokens: List of tokens.
            pos: Current position in the token list.

        Returns:
            Tuple of (result, new position).

        Raises:
            ValueError: If mismatched parentheses or invalid expression.
        """
        left, pos = self._parse_term(tokens, pos)
        while pos < len(tokens) and tokens[pos] in "+-":
            op = tokens[pos]
            pos += 1
            right, pos = self._parse_term(tokens, pos)
            if op == "+":
                left = left + right
            else:
                left = left - right
        return left, pos

    def _parse_term(self, tokens: List[str], pos: int) -> Tuple[float, int]:
        """
        Parse multiplication and division (higher precedence).

        Args:
            tokens: List of tokens.
            pos: Current position in the token list.

        Returns:
            Tuple of (result, new position).

        Raises:
            ValueError: If division by zero.
        """
        left, pos = self._parse_factor(tokens, pos)
        while pos < len(tokens) and tokens[pos] in "*/":
            op = tokens[pos]
            pos += 1
            right, pos = self._parse_factor(tokens, pos)
            if op == "*":
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
        return left, pos

    def _parse_factor(self, tokens: List[str], pos: int) -> Tuple[float, int]:
        """
        Parse numbers, parentheses, and unary minus.

        Args:
            tokens: List of tokens.
            pos: Current position in the token list.

        Returns:
            Tuple of (result, new position).

        Raises:
            ValueError: If mismatched parentheses or invalid token.
        """
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")

        token = tokens[pos]

        if token == "(":
            pos += 1
            result, pos = self._parse_expression(tokens, pos)
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError("Mismatched parentheses")
            pos += 1
            return result, pos
        elif token == "-":
            pos += 1
            value, pos = self._parse_factor(tokens, pos)
            return -value, pos
        elif token == "+":
            pos += 1
            return self._parse_factor(tokens, pos)
        elif re.match(r"^\d+(\.\d+)?$", token):
            pos += 1
            return float(token), pos
        else:
            raise ValueError(f"Invalid token: '{token}'")

import pytest



@pytest.fixture
def evaluator():
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("5 - 3") == 2.0
    assert evaluator.evaluate("2 * 3") == 6.0
    assert evaluator.evaluate("10 / 2") == 5.0


def test_operator_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0


def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0


def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 + -3") == -1.0
    assert evaluator.evaluate("-(-3)") == 3.0


def test_error_cases(evaluator):
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 @ 3")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + abc")