from typing import List, Tuple, Any

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, / with correct operator precedence, parentheses for grouping,
    unary minus, and floating-point numbers.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

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

        if self._current_token()[0] != 'EOF':
            raise ValueError(f"Unexpected token: {self._current_token()[1]}")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, Any]]:
        """Converts the input string into a list of tokens."""
        tokens = []
        i = 0
        n = len(expr)
        while i < n:
            if expr[i].isspace():
                i += 1
                continue
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or (expr[j] == '.' and not has_dot)):
                    if expr[j] == '.':
                        has_dot = True
                    j += 1
                num_str = expr[i:j]
                if num_str == '.' or num_str == '':
                    raise ValueError(f"Invalid token: '{num_str}'")
                tokens.append(('NUMBER', float(num_str)))
                i = j
            elif expr[i] == '+':
                tokens.append(('PLUS', '+'))
                i += 1
            elif expr[i] == '-':
                tokens.append(('MINUS', '-'))
                i += 1
            elif expr[i] == '*':
                tokens.append(('MULT', '*'))
                i += 1
            elif expr[i] == '/':
                tokens.append(('DIV', '/'))
                i += 1
            elif expr[i] == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif expr[i] == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")
        tokens.append(('EOF', None))
        return tokens

    def _current_token(self) -> Tuple[str, Any]:
        """Returns the token at the current position."""
        return self.tokens[self.pos]

    def _advance(self) -> None:
        """Moves the position pointer to the next token."""
        self.pos += 1

    def _parse_expression(self) -> float:
        """Parses addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        while self._current_token()[0] in ('PLUS', 'MINUS'):
            op = self._current_token()[0]
            self._advance()
            right = self._parse_term()
            if op == 'PLUS':
                left = left + right
            else:
                left = left - right
        return left

    def _parse_term(self) -> float:
        """Parses multiplication and division (higher precedence)."""
        left = self._parse_factor()
        while self._current_token()[0] in ('MULT', 'DIV'):
            op = self._current_token()[0]
            self._advance()
            right = self._parse_factor()
            if op == 'MULT':
                left = left * right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                left = left / right
        return left

    def _parse_factor(self) -> float:
        """Parses unary operators (highest precedence before primary)."""
        if self._current_token()[0] == 'MINUS':
            self._advance()
            return -self._parse_factor()
        if self._current_token()[0] == 'PLUS':
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parses numbers and parenthesized expressions."""
        token = self._current_token()
        if token[0] == 'NUMBER':
            self._advance()
            return token[1]
        if token[0] == 'LPAREN':
            self._advance()
            result = self._parse_expression()
            if self._current_token()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        raise ValueError(f"Unexpected token: {token[1]}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_precedence(evaluator):
    """Tests correct operator precedence for +, -, *, /"""
    assert evaluator.evaluate("3 + 4 * 2") == 11.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary(evaluator):
    """Tests parentheses grouping and unary minus support"""
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("- - 3") == 3.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("-(2 * 3) + 1") == -5.0

def test_float_numbers(evaluator):
    """Tests floating-point number parsing and arithmetic"""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + 1.5") == 2.0
    assert evaluator.evaluate("10.0 / 4.0") == pytest.approx(2.5)

def test_division_by_zero(evaluator):
    """Tests that division by zero raises ValueError"""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 * 2 / 0.0")

def test_invalid_expressions(evaluator):
    """Tests error handling for empty, mismatched, and invalid tokens"""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 & 3")