from typing import List, Tuple, Any

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Addition (+), Subtraction (-), Multiplication (*), Division (/)
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14)
    """

    def __init__(self) -> None:
        self.tokens: List[Tuple[str, Any]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        result = self._parse_expression()

        # Ensure the entire expression was consumed
        if self.pos < len(self.tokens) and self.tokens[self.pos][0] != 'EOF':
            raise ValueError("Invalid expression: unexpected tokens after valid expression")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, Any]]:
        """Convert expression string into a list of (type, value) tokens."""
        tokens: List[Tuple[str, Any]] = []
        i = 0
        n = len(expr)

        while i < n:
            if expr[i].isspace():
                i += 1
                continue

            # Numbers (integers and floats)
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format")
                        has_dot = True
                    j += 1
                try:
                    num = float(expr[i:j])
                except ValueError:
                    raise ValueError(f"Invalid number: '{expr[i:j]}'")
                tokens.append(('NUMBER', num))
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
        """Return the token at the current position."""
        return self.tokens[self.pos]

    def _eat(self, token_type: str) -> None:
        """Consume the current token if it matches the expected type."""
        if self._current_token()[0] == token_type:
            self.pos += 1
        else:
            raise ValueError(f"Expected '{token_type}', got '{self._current_token()[0]}'")

    def _parse_expression(self) -> float:
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
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Delegate to unary operator parsing."""
        return self._parse_unary()

    def _parse_unary(self) -> float:
        """Parse unary plus and minus operators."""
        if self._current_token()[0] == 'MINUS':
            self.pos += 1
            return -self._parse_unary()
        if self._current_token()[0] == 'PLUS':
            self.pos += 1
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions (highest precedence)."""
        token = self._current_token()
        if token[0] == 'NUMBER':
            self.pos += 1
            return token[1]
        if token[0] == 'LPAREN':
            self.pos += 1
            result = self._parse_expression()
            self._eat('RPAREN')
            return result
        raise ValueError(f"Unexpected token: '{token[0]}'")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / are evaluated before + and -"""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping with parentheses and unary minus operator"""
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-3 * -4") == 12.0
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0
    assert evaluator.evaluate("--5") == 5.0

def test_floating_point_numbers(evaluator):
    """Test support for decimal numbers"""
    assert evaluator.evaluate("3.14 + 2.86") == pytest.approx(6.0)
    assert evaluator.evaluate("10.5 / 2.1") == pytest.approx(5.0)
    assert evaluator.evaluate(".5 * 4") == pytest.approx(2.0)

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError"""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("2 * (4 / 0)")

def test_error_handling(evaluator):
    """Test mismatched parentheses, invalid tokens, and empty expressions"""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("3..14")