from typing import List, Tuple, Union

Token = Tuple[str, Union[float, str, None]]

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, / with correct precedence, parentheses, unary minus,
    and floating point numbers.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        parser = self._Parser(tokens)
        result = parser.parse_expression()

        if parser.current()[0] != 'EOF':
            raise ValueError("Invalid expression: unexpected tokens after valid expression")

        return result

    @staticmethod
    def _tokenize(expr: str) -> List[Token]:
        """Converts an expression string into a list of tokens."""
        tokens: List[Token] = []
        i = 0
        n = len(expr)
        while i < n:
            c = expr[i]
            if c.isspace():
                i += 1
                continue
            if c.isdigit() or c == '.':
                j = i
                dot_count = 0
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        dot_count += 1
                    j += 1
                if dot_count > 1:
                    raise ValueError("Invalid number format")
                num_str = expr[i:j]
                try:
                    tokens.append(('NUM', float(num_str)))
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")
                i = j
            elif c in '+-*/()':
                tokens.append((c, c))
                i += 1
            else:
                raise ValueError(f"Invalid token: {c}")
        tokens.append(('EOF', None))
        return tokens

    class _Parser:
        """Internal recursive descent parser."""

        def __init__(self, tokens: List[Token]) -> None:
            self.tokens = tokens
            self.pos = 0

        def current(self) -> Token:
            return self.tokens[self.pos]

        def advance(self) -> Token:
            token = self.tokens[self.pos]
            self.pos += 1
            return token

        def parse_expression(self) -> float:
            """Handles addition and subtraction (lowest precedence)."""
            result = self.parse_term()
            while self.current()[0] in ('+', '-'):
                op = self.advance()[0]
                right = self.parse_term()
                if op == '+':
                    result += right
                else:
                    result -= right
            return result

        def parse_term(self) -> float:
            """Handles multiplication and division (higher precedence)."""
            result = self.parse_factor()
            while self.current()[0] in ('*', '/'):
                op = self.advance()[0]
                right = self.parse_factor()
                if op == '*':
                    result *= right
                else:
                    if right == 0.0:
                        raise ValueError("Division by zero")
                    result /= right
            return result

        def parse_factor(self) -> float:
            """Handles numbers, unary minus, and parentheses (highest precedence)."""
            token_type, token_val = self.current()
            if token_type == 'NUM':
                self.advance()
                return float(token_val)
            elif token_type == '-':
                self.advance()
                return -self.parse_factor()
            elif token_type == '(':
                self.advance()
                result = self.parse_expression()
                if self.current()[0] != ')':
                    raise ValueError("Mismatched parentheses")
                self.advance()
                return result
            else:
                raise ValueError(f"Unexpected token: {token_val}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / bind tighter than + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping(evaluator):
    """Test that parentheses correctly override precedence."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus_and_floats(evaluator):
    """Test unary minus and floating point number support."""
    assert evaluator.evaluate("-3.14") == -3.14
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 + -2.5") == 2.5
    assert evaluator.evaluate("- -4") == 4.0

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("(5 - 5) / 2")

def test_error_handling(evaluator):
    """Test various invalid input scenarios."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("3..14")