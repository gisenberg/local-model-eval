from typing import Tuple, List

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Grammar:
        expr   -> term (('+' | '-') term)*
        term   -> factor (('*' | '/') factor)*
        factor -> ('-' | '+') factor | primary
        primary -> NUMBER | '(' expr ')'
    """

    def __init__(self) -> None:
        self._tokens: List[Tuple[str, str]] = []
        self._pos: int = 0

    def _tokenize(self, expr: str) -> None:
        """Converts the input string into a list of tokens."""
        self._tokens = []
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
                self._tokens.append(('NUMBER', expr[i:j]))
                i = j
            elif expr[i] in '+-*/()':
                self._tokens.append((expr[i], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: {expr[i]}")
        self._tokens.append(('EOF', ''))
        self._pos = 0

    def _peek(self) -> Tuple[str, str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos]

    def _consume(self, expected: str = None) -> Tuple[str, str]:
        """Consumes and returns the current token. Optionally validates type."""
        token = self._peek()
        if expected and token[0] != expected:
            raise ValueError(f"Expected {expected}, got {token[0]}")
        self._pos += 1
        return token

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.

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
        
        self._tokenize(expr)
        result = self._parse_expr()
        
        if self._peek()[0] != 'EOF':
            raise ValueError("Unexpected token after expression")
        return result

    def _parse_expr(self) -> float:
        """Parses addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._peek()[0] in ('+', '-'):
            op = self._consume()[0]
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parses multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._peek()[0] in ('*', '/'):
            op = self._consume()[0]
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parses unary plus and minus."""
        if self._peek()[0] in ('+', '-'):
            op = self._consume()[0]
            right = self._parse_factor()
            return right if op == '+' else -right
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parses numbers and parenthesized expressions."""
        token = self._peek()
        if token[0] == 'NUMBER':
            self._consume()
            return float(token[1])
        elif token[0] == '(':
            self._consume()
            result = self._parse_expr()
            if self._peek()[0] != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()
            return result
        else:
            raise ValueError(f"Unexpected token: {token[0]}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test correct precedence: * and / before + and -"""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping(evaluator):
    """Test parentheses override default precedence"""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == 15.0
    assert evaluator.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

def test_unary_minus(evaluator):
    """Test unary minus at start, after operators, and chained"""
    assert evaluator.evaluate("-3 + 2") == -1.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_floating_point_numbers(evaluator):
    """Test support for decimal numbers"""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("10.0 / 4.0") == pytest.approx(2.5)

def test_error_handling(evaluator):
    """Test ValueError raising for invalid inputs"""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
        
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 & 3")
        
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
        
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3 )")