from typing import List, Tuple


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Addition (+), subtraction (-), multiplication (*), division (/)
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating-point numbers (e.g., '3.14', '.5', '5.')
    
    Raises ValueError for:
    - Empty expressions
    - Invalid tokens
    - Mismatched parentheses
    - Division by zero
    """

    def __init__(self) -> None:
        self.tokens: List[Tuple[str, str]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: The mathematical expression string to evaluate.

        Returns:
            The computed result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        result = self._parse_expression()

        if self.pos < len(self.tokens) and self.tokens[self.pos][0] != 'EOF':
            raise ValueError("Invalid expression: unexpected tokens after valid expression")

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
                has_digit = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid token: multiple decimal points")
                        has_dot = True
                    else:
                        has_digit = True
                    j += 1

                if not has_digit:
                    raise ValueError(f"Invalid token: '{expr[i:j]}'")
                tokens.append(('NUMBER', expr[i:j]))
                i = j
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")

        tokens.append(('EOF', ''))
        return tokens

    def _current_token(self) -> Tuple[str, str]:
        """Return the current token."""
        return self.tokens[self.pos]

    def _advance(self) -> Tuple[str, str]:
        """Advance to the next token and return the current one."""
        token = self._current_token()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token()[0] in ('+', '-'):
            op = self._advance()[0]
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
            op = self._advance()[0]
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary plus and minus operators."""
        token = self._current_token()
        if token[0] == '+':
            self._advance()
            return self._parse_factor()
        if token[0] == '-':
            self._advance()
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        token = self._current_token()
        if token[0] == 'NUMBER':
            self._advance()
            return float(token[1])
        if token[0] == '(':
            self._advance()
            result = self._parse_expression()
            if self._current_token()[0] != ')':
                raise ValueError("Mismatched parentheses: missing closing parenthesis")
            self._advance()
            return result
        raise ValueError(f"Unexpected token: {token[1]}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence_and_parentheses(evaluator):
    """Test correct precedence of * / over + - and parentheses grouping."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0

def test_unary_minus_and_floats(evaluator):
    """Test unary minus support and floating-point number parsing."""
    assert evaluator.evaluate("-3.14") == -3.14
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate(".5 * 2") == 1.0

def test_complex_nested_expressions(evaluator):
    """Test deeply nested parentheses and mixed operations."""
    assert evaluator.evaluate("((1 + 2) * 3 - 4) / 2") == 2.5
    assert evaluator.evaluate("1.5 * (2.0 + 3.0) - 1") == 6.5
    assert evaluator.evaluate("-(-(-5))") == -5.0

def test_error_empty_and_invalid_tokens(evaluator):
    """Test ValueError for empty expressions and invalid characters."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 & 4")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3..5")

def test_error_parentheses_and_division_by_zero(evaluator):
    """Test ValueError for mismatched parentheses and division by zero."""
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(3 + 4")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="unexpected tokens"):
        evaluator.evaluate("3 + 4)")