from typing import List, Tuple, Optional

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14, .5, 2.)
    
    Raises ValueError for empty expressions, invalid tokens, 
    mismatched parentheses, and division by zero.
    """

    def __init__(self) -> None:
        self.tokens: List[Tuple[str, Optional[str]]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing the mathematical expression.

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

        result = self._parse_expr()

        if self.pos < len(self.tokens):
            raise ValueError("Invalid token or mismatched parentheses")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, Optional[str]]]:
        """Convert expression string into a list of (type, value) tokens."""
        tokens: List[Tuple[str, Optional[str]]] = []
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
                if not any(c.isdigit() for c in num_str):
                    raise ValueError("Invalid number format")
                tokens.append(('NUMBER', num_str))
                i = j
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: {expr[i]}")

        tokens.append(('EOF', None))
        return tokens

    def _current_token(self) -> Tuple[str, Optional[str]]:
        """Return the token at the current position."""
        return self.tokens[self.pos]

    def _advance(self) -> None:
        """Move the parser to the next token."""
        self.pos += 1

    def _parse_expr(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token()[0] in ('+', '-'):
            op = self._current_token()[1]
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
        while self._current_token()[0] in ('*', '/'):
            op = self._current_token()[1]
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
        """Parse unary operators and delegate to primary expressions."""
        if self._current_token()[0] == '-':
            self._advance()
            return -self._parse_factor()
        if self._current_token()[0] == '+':
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions (highest precedence)."""
        token = self._current_token()
        if token[0] == 'NUMBER':
            self._advance()
            return float(token[1])
        elif token[0] == '(':
            self._advance()
            result = self._parse_expr()
            if self._current_token()[0] != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        else:
            raise ValueError(f"Invalid token: {token[1] if token[1] is not None else 'EOF'}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_precedence_and_parentheses(evaluator):
    """Test standard operator precedence and grouping."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    assert evaluator.evaluate("2 * (3 + 4) / 7") == pytest.approx(2.0)

def test_unary_minus(evaluator):
    """Test unary minus in various positions."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("- - 3.5") == 3.5
    assert evaluator.evaluate("10 + -2 * 3") == 4.0

def test_floating_point_numbers(evaluator):
    """Test parsing and arithmetic with floats."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("1.5 + 2.5") == 4.0
    assert evaluator.evaluate(".5") == 0.5
    assert evaluator.evaluate("5.") == 5.0
    assert evaluator.evaluate("1.1 + 2.2") == pytest.approx(3.3)

def test_error_conditions(evaluator):
    """Test ValueError raising for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError, match="Invalid number format"):
        evaluator.evaluate("3..14")

def test_complex_nested_expressions(evaluator):
    """Test deeply nested and mixed expressions."""
    assert evaluator.evaluate("((10 - 2) / 4) * (3.5 + 1.5)") == pytest.approx(10.0)
    assert evaluator.evaluate("-2 * (3 + -4)") == pytest.approx(2.0)
    assert evaluator.evaluate("-( -( -( 5 ) ) )") == -5.0
    assert evaluator.evaluate("1 + 2 * 3 - 4 / 2") == pytest.approx(5.0)