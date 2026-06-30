from typing import List, Optional

class Token:
    """Represents a lexical token in the expression."""
    def __init__(self, type_: str, value: Optional[float]) -> None:
        self.type = type_
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value})"


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Addition (+), Subtraction (-), Multiplication (*), Division (/)
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14)
    
    Raises ValueError for invalid syntax, mismatched parentheses, 
    division by zero, or empty expressions.
    """
    def __init__(self) -> None:
        self.tokens: List[Token] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        result = self._parse_expression()

        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")

        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Converts the input string into a list of tokens."""
        tokens: List[Token] = []
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
                            raise ValueError("Invalid number format: multiple decimal points")
                        has_dot = True
                    j += 1
                num_str = expr[i:j]
                if num_str == '.':
                    raise ValueError("Invalid number format: lone decimal point")
                tokens.append(Token('NUMBER', float(num_str)))
                i = j
            elif expr[i] == '+':
                tokens.append(Token('PLUS', None))
                i += 1
            elif expr[i] == '-':
                tokens.append(Token('MINUS', None))
                i += 1
            elif expr[i] == '*':
                tokens.append(Token('MULTIPLY', None))
                i += 1
            elif expr[i] == '/':
                tokens.append(Token('DIVIDE', None))
                i += 1
            elif expr[i] == '(':
                tokens.append(Token('LPAREN', None))
                i += 1
            elif expr[i] == ')':
                tokens.append(Token('RPAREN', None))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")

        tokens.append(Token('EOF', None))
        return tokens

    def _current_token(self) -> Token:
        """Returns the token at the current parsing position."""
        return self.tokens[self.pos]

    def _consume(self, expected_type: str) -> Token:
        """Consumes the current token if it matches the expected type."""
        token = self._current_token()
        if token.type != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token.type}")
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parses addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token().type in ('PLUS', 'MINUS'):
            op = self._current_token().type
            self.pos += 1
            right = self._parse_term()
            if op == 'PLUS':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parses multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token().type in ('MULTIPLY', 'DIVIDE'):
            op = self._current_token().type
            self.pos += 1
            right = self._parse_factor()
            if op == 'MULTIPLY':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parses unary operators, numbers, and parenthesized expressions."""
        token = self._current_token()

        if token.type == 'MINUS':
            self.pos += 1
            return -self._parse_factor()

        if token.type == 'NUMBER':
            self.pos += 1
            return token.value

        if token.type == 'LPAREN':
            self.pos += 1
            result = self._parse_expression()
            self._consume('RPAREN')
            return result

        raise ValueError(f"Unexpected token: {token}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Tests that * and / are evaluated before + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus(evaluator):
    """Tests grouping with parentheses and unary minus operator."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 1) * 4") == -12.0
    assert evaluator.evaluate("--5") == 5.0

def test_floating_point_numbers(evaluator):
    """Tests support for decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("1.5 + 2.5") == 4.0
    assert evaluator.evaluate("10.0 / 4.0") == pytest.approx(2.5)

def test_error_handling(evaluator):
    """Tests ValueError raising for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 + a")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

def test_mismatched_parentheses(evaluator):
    """Tests ValueError raising for unbalanced parentheses."""
    with pytest.raises(ValueError, match="Expected RPAREN"):
        evaluator.evaluate("(3 + 4")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("3 + )")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("()")