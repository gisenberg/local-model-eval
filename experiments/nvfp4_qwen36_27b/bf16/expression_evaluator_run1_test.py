from typing import List, Tuple, Optional

Token = Tuple[str, Optional[str]]

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, / with correct precedence, parentheses, unary minus,
    and floating point numbers.
    """

    def __init__(self) -> None:
        self.tokens: List[Token] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.

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

        self.tokens = self._tokenize(expr)
        self.pos = 0
        result = self._parse_expression()

        if self.pos < len(self.tokens) and self.tokens[self.pos][0] != 'EOF':
            raise ValueError(f"Unexpected token: {self.tokens[self.pos][1]}")

        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Convert expression string into a list of tokens."""
        tokens: List[Token] = []
        i = 0
        n = len(expr)
        while i < n:
            if expr[i].isspace():
                i += 1
                continue
            if expr[i].isdigit() or expr[i] == '.':
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                num_str = expr[start:i]
                if num_str == '.' or num_str.count('.') > 1:
                    raise ValueError(f"Invalid number format: {num_str}")
                tokens.append(('NUMBER', num_str))
            elif expr[i] in '+-*/()':
                type_map = {'+': 'PLUS', '-': 'MINUS', '*': 'MULT', '/': 'DIV', '(': 'LPAREN', ')': 'RPAREN'}
                tokens.append((type_map[expr[i]], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: {expr[i]}")
        tokens.append(('EOF', None))
        return tokens

    def _peek(self) -> Token:
        """Return the current token without consuming it."""
        return self.tokens[self.pos]

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._peek()[0] in ('PLUS', 'MINUS'):
            op = self._peek()[0]
            self.pos += 1
            right = self._parse_term()
            result = result + right if op == 'PLUS' else result - right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._peek()[0] in ('MULT', 'DIV'):
            op = self._peek()[0]
            self.pos += 1
            right = self._parse_factor()
            if op == 'MULT':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary operators, numbers, and parenthesized expressions."""
        token = self._peek()
        if token[0] == 'PLUS':
            self.pos += 1
            return self._parse_factor()
        elif token[0] == 'MINUS':
            self.pos += 1
            return -self._parse_factor()
        elif token[0] == 'NUMBER':
            self.pos += 1
            return float(token[1])
        elif token[0] == 'LPAREN':
            self.pos += 1
            result = self._parse_expression()
            if self._peek()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result
        else:
            raise ValueError(f"Unexpected token: {token[1]}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / bind tighter than + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping and unary minus handling."""
    assert evaluator.evaluate("-(2 + 3) * 2") == -10.0
    assert evaluator.evaluate("-3 + --4") == 1.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_floating_point_numbers(evaluator):
    """Test support for decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + 1.5") == pytest.approx(2.0)

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

def test_error_handling(evaluator):
    """Test various invalid inputs raise ValueError."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(3 + 4")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("3 + 4)")