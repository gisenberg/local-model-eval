from typing import List, Tuple, Optional

class ExpressionEvaluator:
    """A recursive descent parser for evaluating mathematical expressions.
    
    Supports +, -, *, / with standard precedence, parentheses, unary minus,
    and floating-point numbers.
    """

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        self.tokens = self._tokenize(expr)
        self.pos = 0

        if self.tokens[0][0] == 'EOF':
            raise ValueError("Empty expression")

        result = self._parse_expression()

        if self.tokens[self.pos][0] != 'EOF':
            if self.tokens[self.pos][0] == ')':
                raise ValueError("Mismatched parentheses")
            raise ValueError("Unexpected token after expression")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, Optional[float]]]:
        """Convert expression string into a list of tokens."""
        tokens: List[Tuple[str, Optional[float]]] = []
        i = 0
        n = len(expr)

        while i < n:
            if expr[i].isspace():
                i += 1
                continue

            if expr[i].isdigit() or expr[i] == '.':
                j = i
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                num_str = expr[i:j]
                try:
                    tokens.append(('NUM', float(num_str)))
                except ValueError:
                    raise ValueError(f"Invalid number format: '{num_str}'")
                i = j
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], None))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")

        tokens.append(('EOF', None))
        return tokens

    def _current_token(self) -> Tuple[str, Optional[float]]:
        """Return the current token."""
        return self.tokens[self.pos]

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token()[0] in ('+', '-'):
            op = self._current_token()[0]
            self.pos += 1
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token()[0] in ('*', '/'):
            op = self._current_token()[0]
            self.pos += 1
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
        token = self._current_token()
        if token[0] == '+':
            self.pos += 1
            return self._parse_factor()
        if token[0] == '-':
            self.pos += 1
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions (highest precedence)."""
        token = self._current_token()
        if token[0] == 'NUM':
            self.pos += 1
            return token[1]  # type: ignore[return-value]
        if token[0] == '(':
            self.pos += 1
            result = self._parse_expression()
            if self._current_token()[0] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result
        raise ValueError(f"Unexpected token: {token[0]}")

import pytest

def test_basic_arithmetic_and_precedence():
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("10 - 2 / 2") == 9.0

def test_unary_minus_and_floats():
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate("--5.5") == 5.5

def test_division_by_zero():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("5 / (2 - 2)")

def test_mismatched_parentheses():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("2 + 3)")

def test_invalid_tokens_and_empty():
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + a")
    with pytest.raises(ValueError, match="Invalid number format"):
        ev.evaluate("2..3")