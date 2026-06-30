from __future__ import annotations
from typing import Tuple, List


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus/plus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14)
    
    Raises ValueError for:
    - Empty expressions
    - Mismatched parentheses
    - Division by zero
    - Invalid tokens or malformed numbers
    """

    def __init__(self) -> None:
        self.tokens: List[Tuple[str, str]] = []
        self.pos: int = 0

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
            
        self.tokens = self._tokenize(expr)
        self.pos = 0
        result = self._parse_expr()
        
        if self.tokens[self.pos][0] != 'EOF':
            raise ValueError("Invalid expression: unexpected tokens after end of expression")
            
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
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                num_str = expr[i:j]
                if num_str.count('.') > 1:
                    raise ValueError(f"Invalid number format: '{num_str}'")
                tokens.append(('NUM', num_str))
                i = j
            elif expr[i] == '+':
                tokens.append(('PLUS', '+'))
                i += 1
            elif expr[i] == '-':
                tokens.append(('MINUS', '-'))
                i += 1
            elif expr[i] == '*':
                tokens.append(('MUL', '*'))
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
                
        tokens.append(('EOF', ''))
        return tokens

    def _parse_expr(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        while self.tokens[self.pos][0] in ('PLUS', 'MINUS'):
            op = self.tokens[self.pos][0]
            self.pos += 1
            right = self._parse_term()
            left = left + right if op == 'PLUS' else left - right
        return left

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        left = self._parse_factor()
        while self.tokens[self.pos][0] in ('MUL', 'DIV'):
            op = self.tokens[self.pos][0]
            self.pos += 1
            right = self._parse_factor()
            if op == 'MUL':
                left *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_factor(self) -> float:
        """Parse unary plus/minus and delegate to primary expressions."""
        if self.tokens[self.pos][0] in ('PLUS', 'MINUS'):
            op = self.tokens[self.pos][0]
            self.pos += 1
            val = self._parse_factor()
            return val if op == 'PLUS' else -val
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        token = self.tokens[self.pos]
        if token[0] == 'NUM':
            self.pos += 1
            try:
                return float(token[1])
            except ValueError:
                raise ValueError(f"Invalid number: '{token[1]}'")
        elif token[0] == 'LPAREN':
            self.pos += 1
            val = self._parse_expr()
            if self.tokens[self.pos][0] != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return val
        else:
            raise ValueError(f"Unexpected token: {token[1] if token[1] else 'EOF'}")

import pytest


class TestExpressionEvaluator:
    @pytest.fixture
    def evaluator(self):
        return ExpressionEvaluator()

    def test_operator_precedence(self, evaluator):
        """Test that * and / bind tighter than + and -"""
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        assert evaluator.evaluate("10 / 2 - 3") == 2.0
        assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

    def test_parentheses_and_unary_minus(self, evaluator):
        """Test grouping and unary minus handling"""
        assert evaluator.evaluate("-(2 + 3)") == -5.0
        assert evaluator.evaluate("(-3) * 2") == -6.0
        assert evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert evaluator.evaluate("--5") == 5.0

    def test_floating_point_numbers(self, evaluator):
        """Test decimal number parsing and arithmetic"""
        assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
        assert evaluator.evaluate("1.5 / 0.5") == 3.0
        assert evaluator.evaluate(".5 + .5") == 1.0

    def test_division_by_zero(self, evaluator):
        """Test that division by zero raises ValueError"""
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("10 / 0")
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("2 * (3 / 0)")

    def test_invalid_expressions(self, evaluator):
        """Test error handling for malformed inputs"""
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")
        with pytest.raises(ValueError, match="Invalid token"):
            evaluator.evaluate("2 + a")
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("(2 + 3")
        with pytest.raises(ValueError, match="Invalid number"):
            evaluator.evaluate("1.2.3 + 4")