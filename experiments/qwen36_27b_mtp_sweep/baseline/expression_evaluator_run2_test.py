from typing import List, Tuple, Any

class ExpressionEvaluator:
    """A recursive descent parser for evaluating mathematical expressions.
    
    Supports +, -, *, / with standard precedence, parentheses, unary minus,
    and floating-point numbers.
    """

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

        tokens = self._tokenize(expr)
        self._tokens = tokens
        self._pos = 0
        self._current_token = tokens[0]

        result = self._parse_expression()

        if self._current_token[0] != 'EOF':
            raise ValueError(f"Unexpected token after expression: {self._current_token}")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, Any]]:
        """Convert expression string into a list of tokens."""
        tokens: List[Tuple[str, Any]] = []
        i = 0
        while i < len(expr):
            char = expr[i]
            if char.isspace():
                i += 1
                continue
            if char.isdigit() or char == '.':
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                num_str = expr[i:j]
                if num_str.count('.') > 1:
                    raise ValueError(f"Invalid number format: {num_str}")
                tokens.append(('NUM', float(num_str)))
                i = j
                continue
            if char in '+-*/()':
                tokens.append((char, char))
                i += 1
                continue
            raise ValueError(f"Invalid token: '{char}'")
        tokens.append(('EOF', None))
        return tokens

    def _advance(self) -> None:
        """Move to the next token."""
        self._pos += 1
        if self._pos < len(self._tokens):
            self._current_token = self._tokens[self._pos]
        else:
            self._current_token = ('EOF', None)

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token[0] in ('+', '-'):
            op = self._current_token[0]
            self._advance()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token[0] in ('*', '/'):
            op = self._current_token[0]
            self._advance()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary plus and minus."""
        if self._current_token[0] in ('+', '-'):
            op = self._current_token[0]
            self._advance()
            val = self._parse_factor()
            return val if op == '+' else -val
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions (highest precedence)."""
        token = self._current_token
        if token[0] == 'NUM':
            self._advance()
            return token[1]
        if token[0] == '(':
            self._advance()
            result = self._parse_expression()
            if self._current_token[0] != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._advance()
            return result
        raise ValueError(f"Unexpected token: {token}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / bind tighter than + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping(evaluator):
    """Test that parentheses override default precedence."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert evaluator.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

def test_unary_minus(evaluator):
    """Test unary minus on numbers and parenthesized expressions."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("- - 5") == 5.0
    assert evaluator.evaluate("10 + -3 * 2") == 4.0

def test_floating_point_numbers(evaluator):
    """Test parsing and arithmetic with decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("1.5 + 2.5") == 4.0
    assert evaluator.evaluate("10.0 / 3.0") == pytest.approx(3.3333333333333335)

def test_error_handling(evaluator):
    """Test that appropriate ValueErrors are raised for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
        
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
        
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + @ 3")