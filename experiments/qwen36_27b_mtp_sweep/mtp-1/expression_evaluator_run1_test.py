from typing import List, Tuple, Optional

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Addition (+), Subtraction (-), Multiplication (*), Division (/)
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating-point numbers (e.g., 3.14, .5, 5.)
    
    Raises ValueError for empty expressions, invalid tokens, 
    mismatched parentheses, and division by zero.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The numerical result of the expression as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        result = self._parse_expression()

        # Ensure no trailing tokens remain after EOF
        if self.pos < len(self.tokens) and self.tokens[self.pos][0] != 'EOF':
            raise ValueError(f"Unexpected token: {self.tokens[self.pos][1]}")

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
                if i == j - 1 and expr[i] == '.':
                    raise ValueError("Invalid token: '.'")
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

        tokens.append(('EOF', None))
        return tokens

    def _current_token(self) -> Tuple[str, Optional[str]]:
        """Return the token at the current position."""
        return self.tokens[self.pos]

    def _consume(self, expected_type: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Consume and return the current token, optionally validating its type."""
        token = self._current_token()
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[0]}")
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token()[0] in ('PLUS', 'MINUS'):
            op = self._consume()[0]
            right = self._parse_term()
            result = result + right if op == 'PLUS' else result - right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token()[0] in ('MULT', 'DIV'):
            op = self._consume()[0]
            right = self._parse_factor()
            if op == 'MULT':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary operators (highest precedence before primary)."""
        if self._current_token()[0] == 'MINUS':
            self._consume()
            return -self._parse_factor()
        if self._current_token()[0] == 'PLUS':
            self._consume()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        token = self._current_token()
        if token[0] == 'NUMBER':
            self._consume()
            return float(token[1])
        if token[0] == 'LPAREN':
            self._consume()
            result = self._parse_expression()
            if self._current_token()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self._consume()
            return result
        raise ValueError(f"Unexpected token: {token[1] if token[1] else token[0]}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic_and_precedence(evaluator):
    """Test standard operators and correct precedence (* / before + -)"""
    assert evaluator.evaluate("3 + 4 * 2") == 11.0
    assert evaluator.evaluate("(3 + 4) * 2") == 14.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_unary_minus_and_floats(evaluator):
    """Test unary minus and floating-point number support"""
    assert evaluator.evaluate("-3.14") == -3.14
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("3.5 * 2.0") == 7.0
    assert evaluator.evaluate(".5 + 1.5") == 2.0

def test_empty_and_invalid_tokens(evaluator):
    """Test error handling for empty strings and invalid characters"""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3 @ 4")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("3 +")

def test_mismatched_parentheses(evaluator):
    """Test error handling for unbalanced parentheses"""
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(3 + 4")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("3 + 4)")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("((3 + 4)")

def test_division_by_zero(evaluator):
    """Test error handling for division by zero"""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (2 - 2)")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("0 / 0")