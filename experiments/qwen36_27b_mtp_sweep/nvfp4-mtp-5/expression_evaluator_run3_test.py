import re
from typing import List

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    
    Supports:
    - Binary operators: +, -, *, / with correct precedence
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating-point numbers (e.g., '3.14')
    
    Raises ValueError for:
    - Empty expressions
    - Invalid tokens
    - Mismatched parentheses
    - Division by zero
    """

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.pos: int = 0

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the raw expression string into a list of lexical tokens."""
        tokens: List[str] = []
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
                tokens.append(expr[i:j])
                i = j
            elif expr[i] in '+-*/()':
                tokens.append(expr[i])
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")
                
        return tokens

    def _current_token(self) -> str:
        """Returns the current token or 'EOF' if at the end of the token stream."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return 'EOF'

    def _advance(self) -> str:
        """Consumes the current token and returns it."""
        token = self._current_token()
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """Parses addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token() in ('+', '-'):
            op = self._advance()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parses multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token() in ('*', '/'):
            op = self._advance()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parses unary plus/minus and delegates to primary expressions."""
        if self._current_token() == '+':
            self._advance()
            return self._parse_factor()
        if self._current_token() == '-':
            self._advance()
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parses atomic values: numbers and parenthesized expressions."""
        token = self._current_token()
        
        if token == '(':
            self._advance()
            result = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
            
        if token == 'EOF':
            raise ValueError("Unexpected end of expression")
            
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Invalid token: '{token}'")

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: A string containing a valid mathematical expression.
            
        Returns:
            The evaluated result as a float.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        self.tokens = self._tokenize(expr)
        if not self.tokens:
            raise ValueError("Empty expression")
            
        self.pos = 0
        result = self._parse_expression()
        
        if self._current_token() != 'EOF':
            raise ValueError("Invalid expression: unexpected tokens after valid expression")
            
        return result

import pytest

class TestExpressionEvaluator:
    @pytest.fixture(autouse=True)
    def evaluator(self) -> ExpressionEvaluator:
        return ExpressionEvaluator()

    def test_operator_precedence(self, evaluator: ExpressionEvaluator) -> None:
        """Tests that * and / bind tighter than + and -."""
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        assert evaluator.evaluate("10 / 2 - 1") == 4.0
        assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

    def test_unary_minus(self, evaluator: ExpressionEvaluator) -> None:
        """Tests unary minus on numbers and parenthesized expressions."""
        assert evaluator.evaluate("-3") == -3.0
        assert evaluator.evaluate("-(2 + 1)") == -3.0
        assert evaluator.evaluate("--5") == 5.0
        assert evaluator.evaluate("- - 3.14") == pytest.approx(3.14)

    def test_parentheses_grouping(self, evaluator: ExpressionEvaluator) -> None:
        """Tests that parentheses correctly override default precedence."""
        assert evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
        assert evaluator.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

    def test_floating_point_numbers(self, evaluator: ExpressionEvaluator) -> None:
        """Tests support for decimal numbers."""
        assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
        assert evaluator.evaluate("1.5 / 0.5") == 3.0
        assert evaluator.evaluate(".5 + .5") == 1.0

    def test_error_handling(self, evaluator: ExpressionEvaluator) -> None:
        """Tests that appropriate ValueErrors are raised for invalid inputs."""
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("   ")
            
        with pytest.raises(ValueError, match="Invalid token"):
            evaluator.evaluate("2 & 3")
            
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("2 + 3)")
            
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("1 / 0")