from typing import List, Tuple

Token = Tuple[str, str]

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
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

        self.tokens = self._tokenize(expr)
        self.pos = 0

        result = self._parse_expression()

        if self._current_token()[0] != 'EOF':
            raise ValueError("Invalid expression: unexpected tokens at end")

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
                            raise ValueError(f"Invalid token: multiple decimal points in number")
                        has_dot = True
                    j += 1
                tokens.append(('NUMBER', expr[i:j]))
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

        tokens.append(('EOF', ''))
        return tokens

    def _current_token(self) -> Token:
        return self.tokens[self.pos]

    def _eat(self, token_type: str) -> str:
        token = self._current_token()
        if token[0] != token_type:
            raise ValueError(f"Expected {token_type}, got {token[0]}")
        self.pos += 1
        return token[1]

    def _parse_expression(self) -> float:
        """Parses addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token()[0] in ('PLUS', 'MINUS'):
            op = self._current_token()[0]
            self._eat(op)
            right = self._parse_term()
            if op == 'PLUS':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parses multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token()[0] in ('MULT', 'DIV'):
            op = self._current_token()[0]
            self._eat(op)
            right = self._parse_factor()
            if op == 'MULT':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parses unary operators."""
        if self._current_token()[0] == 'MINUS':
            self._eat('MINUS')
            return -self._parse_factor()
        if self._current_token()[0] == 'PLUS':
            self._eat('PLUS')
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parses numbers and parenthesized expressions (highest precedence)."""
        token = self._current_token()
        if token[0] == 'NUMBER':
            self._eat('NUMBER')
            return float(token[1])
        if token[0] == 'LPAREN':
            self._eat('LPAREN')
            result = self._parse_expression()
            if self._current_token()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses: missing closing parenthesis")
            self._eat('RPAREN')
            return result
        raise ValueError(f"Unexpected token: {token[0]}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Tests that * and / bind tighter than + and -"""
    assert evaluator.evaluate("2 + 3 * 4 - 8 / 2") == pytest.approx(10.0)

def test_parentheses_and_unary_minus(evaluator):
    """Tests grouping and unary minus support"""
    assert evaluator.evaluate("-(2 + 3) * 2") == pytest.approx(-10.0)
    assert evaluator.evaluate("--3 + -(-2)") == pytest.approx(5.0)

def test_floating_point_numbers(evaluator):
    """Tests decimal number parsing and arithmetic"""
    assert evaluator.evaluate("3.14 * 2 + 0.5") == pytest.approx(6.78)
    assert evaluator.evaluate(".5 / 2") == pytest.approx(0.25)

def test_division_by_zero(evaluator):
    """Tests that division by zero raises ValueError"""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0.0")

def test_error_handling(evaluator):
    """Tests mismatched parentheses, invalid tokens, and empty expressions"""
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + 3 & 4")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")