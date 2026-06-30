from typing import Iterator, Tuple, List, Union, Optional

# Token type: (token_type, token_value)
Token = Tuple[str, Union[float, str]]

class ExpressionEvaluator:
    """A recursive descent parser for evaluating mathematical expressions.
    
    Supports +, -, *, / with correct operator precedence, parentheses for grouping,
    unary minus, and floating-point numbers. Raises ValueError for invalid inputs.
    """

    def __init__(self) -> None:
        self.tokens: List[Token] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = list(self._tokenize(expr))
        self.pos = 0

        result = self._parse_expr()

        if self.pos < len(self.tokens):
            token_type = self.tokens[self.pos][0]
            if token_type in ('(', ')'):
                raise ValueError("Mismatched parentheses")
            raise ValueError(f"Unexpected token: {token_type}")

        return result

    def _tokenize(self, expr: str) -> Iterator[Token]:
        """Convert expression string into a stream of tokens."""
        i = 0
        n = len(expr)
        while i < n:
            if expr[i].isspace():
                i += 1
                continue
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format")
                        has_dot = True
                    j += 1
                num_str = expr[i:j]
                if num_str == '.':
                    raise ValueError("Invalid number format")
                yield ('NUM', float(num_str))
                i = j
            elif expr[i] in '+-*/()':
                yield (expr[i], expr[i])
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")
        yield ('EOF', None)

    def _peek(self) -> Token:
        """Return the current token without consuming it."""
        return self.tokens[self.pos]

    def _consume(self, expected_type: Optional[str] = None) -> Token:
        """Consume and return the current token. Optionally validate its type."""
        token = self._peek()
        if expected_type is not None and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[0]}")
        self.pos += 1
        return token

    def _parse_expr(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._peek()[0] in ('+', '-'):
            op = self._consume()[0]
            right = self._parse_term()
            result = result + right if op == '+' else result - right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_unary()
        while self._peek()[0] in ('*', '/'):
            op = self._consume()[0]
            right = self._parse_unary()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_unary(self) -> float:
        """Parse unary minus (highest precedence)."""
        if self._peek()[0] == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        token = self._peek()
        if token[0] == 'NUM':
            self._consume()
            return token[1]
        if token[0] == '(':
            self._consume()
            result = self._parse_expr()
            if self._peek()[0] != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()
            return result
        raise ValueError(f"Unexpected token: {token[0]}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / bind tighter than + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping and unary minus in various positions."""
    assert evaluator.evaluate("-(2 + 3) * 4") == -20.0
    assert evaluator.evaluate("-3 + 2") == -1.0
    assert evaluator.evaluate("2 * -3") == -6.0
    assert evaluator.evaluate("-( -(5) )") == 5.0

def test_floating_point_numbers(evaluator):
    """Test support for decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("10.0 / 4.0") == pytest.approx(2.5)
    assert evaluator.evaluate("1.5 * 2.0 + 0.5") == pytest.approx(3.5)

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("(2 + 3) / 0.0")

def test_error_handling(evaluator):
    """Test ValueError for mismatched parentheses, invalid tokens, and empty input."""
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")