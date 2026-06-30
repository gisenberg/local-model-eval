from typing import List, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Grammar:
        expression := term (('+' | '-') term)*
        term       := unary (('*' | '/') unary)*
        unary      := ('-')* primary
        primary    := NUMBER | '(' expression ')'
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

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        # Ensure the entire expression was consumed
        if self._pos < len(self._tokens) and self._tokens[self._pos][0] != 'EOF':
            raise ValueError(f"Unexpected token: {self._tokens[self._pos][1]}")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, str]]:
        """Convert expression string into a list of (type, value) tokens."""
        tokens: List[Tuple[str, str]] = []
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
                if num_str == '.' or num_str == '':
                    raise ValueError("Invalid number format")
                tokens.append(('NUMBER', num_str))
                i = j
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")

        tokens.append(('EOF', ''))
        return tokens

    def _current_token(self) -> Tuple[str, str]:
        """Return the token at the current position."""
        return self._tokens[self._pos]

    def _consume(self, expected_type: str = None) -> Tuple[str, str]:
        """Consume and return the current token, optionally validating its type."""
        token = self._current_token()
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[0]}")
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token()[0] in ('+', '-'):
            op = self._consume()[0]
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_unary()
        while self._current_token()[0] in ('*', '/'):
            op = self._consume()[0]
            right = self._parse_unary()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_unary(self) -> float:
        """Parse unary minus operators."""
        if self._current_token()[0] == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions (highest precedence)."""
        token = self._current_token()
        if token[0] == 'NUMBER':
            self._consume()
            return float(token[1])
        elif token[0] == '(':
            self._consume()
            result = self._parse_expression()
            if self._current_token()[0] != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()
            return result
        else:
            raise ValueError(f"Unexpected token: {token[0]}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator: ExpressionEvaluator):
    """Test that * and / are evaluated before + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping(evaluator: ExpressionEvaluator):
    """Test that parentheses override default precedence."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == 15.0
    assert evaluator.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

def test_unary_minus(evaluator: ExpressionEvaluator):
    """Test unary minus on numbers and grouped expressions."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-2 * -3") == 6.0

def test_floating_point_numbers(evaluator: ExpressionEvaluator):
    """Test support for decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + 1.5") == 2.0
    assert evaluator.evaluate("10 / 3.0") == pytest.approx(3.3333333333333335)

def test_error_conditions(evaluator: ExpressionEvaluator):
    """Test that appropriate ValueErrors are raised for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
        
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3)")
        
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + * 3")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")