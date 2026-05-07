from typing import List, Tuple

Token = Tuple[str, str]

class ExpressionEvaluator:
    """A recursive descent parser for evaluating mathematical expressions.
    
    Supports +, -, *, / with standard precedence, parentheses, unary minus,
    and floating-point numbers. Raises ValueError for invalid inputs.
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
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        result = self._parse_expr()

        if self.pos < len(self.tokens) and self.tokens[self.pos][0] != 'EOF':
            raise ValueError("Invalid expression: unexpected tokens after valid expression")

        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Convert expression string into a list of tokens."""
        tokens: List[Token] = []
        i = 0
        n = len(expr)

        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue

            if ch.isdigit() or ch == '.':
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                num_str = expr[start:i]
                if num_str.count('.') > 1 or num_str == '.':
                    raise ValueError(f"Invalid number format: {num_str}")
                tokens.append(('NUMBER', num_str))
            elif ch == '+':
                tokens.append(('PLUS', '+'))
                i += 1
            elif ch == '-':
                tokens.append(('MINUS', '-'))
                i += 1
            elif ch == '*':
                tokens.append(('MULT', '*'))
                i += 1
            elif ch == '/':
                tokens.append(('DIV', '/'))
                i += 1
            elif ch == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif ch == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{ch}'")

        tokens.append(('EOF', ''))
        return tokens

    def _current_token(self) -> Token:
        """Return the current token."""
        return self.tokens[self.pos]

    def _parse_expr(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token()[0] in ('PLUS', 'MINUS'):
            op = self._current_token()[0]
            self.pos += 1
            right = self._parse_term()
            if op == 'PLUS':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token()[0] in ('MULT', 'DIV'):
            op = self._current_token()[0]
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
        """Parse unary operators, numbers, and parentheses (highest precedence)."""
        token = self._current_token()

        if token[0] == 'MINUS':
            self.pos += 1
            return -self._parse_factor()
        if token[0] == 'PLUS':
            self.pos += 1
            return self._parse_factor()
        if token[0] == 'NUMBER':
            self.pos += 1
            return float(token[1])
        if token[0] == 'LPAREN':
            self.pos += 1
            result = self._parse_expr()
            if self._current_token()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result

        raise ValueError(f"Unexpected token: {token[0]}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / are evaluated before + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0

def test_parentheses_grouping(evaluator):
    """Test that parentheses override default precedence."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == 15.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    assert evaluator.evaluate("(10 - 2) / (4 - 2)") == 4.0

def test_unary_minus(evaluator):
    """Test unary minus at start, after operators, and nested."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 + -2 * 3") == -1.0
    assert evaluator.evaluate("- - 3") == 3.0
    assert evaluator.evaluate("10 * -2.5") == -25.0

def test_floating_point_numbers(evaluator):
    """Test support for decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("0.1 + 0.2") == pytest.approx(0.3)
    assert evaluator.evaluate("10.5 / 2.1") == pytest.approx(5.0)
    assert evaluator.evaluate("-.5 + 1.5") == pytest.approx(1.0)

def test_error_handling(evaluator):
    """Test ValueError raises for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
        
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError, match="Invalid number format"):
        evaluator.evaluate("2.3.4")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("()")
        
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / (2 - 2)")