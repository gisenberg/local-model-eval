from typing import Optional


class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus
    - Floating point numbers
    
    Raises:
        ValueError: For invalid expressions, mismatched parentheses, division by zero, etc.
    """

    def __init__(self):
        self.pos: int = 0
        self.expr: str = ""
        self.current_char: Optional[str] = None

    def evaluate(self, expr: str) -> float:
        """
        Evaluate the mathematical expression and return the result as a float.
        
        Args:
            expr (str): The mathematical expression to evaluate.
            
        Returns:
            float: The result of the evaluated expression.
            
        Raises:
            ValueError: If the expression is invalid.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")
        
        self.expr = expr.replace(" ", "")
        self.pos = 0
        self.current_char = self.expr[self.pos] if self.expr else None

        result = self.parse_expression()
        
        if self.current_char is not None:
            raise ValueError(f"Invalid token at position {self.pos}: '{self.current_char}'")
        
        return result

    def advance(self):
        """Move to the next character in the expression."""
        self.pos += 1
        if self.pos < len(self.expr):
            self.current_char = self.expr[self.pos]
        else:
            self.current_char = None

    def parse_expression(self) -> float:
        """Parse addition and subtraction."""
        return self.parse_term()

    def parse_term(self) -> float:
        """Parse multiplication and division."""
        left = self.parse_factor()

        while self.current_char in ('*', '/'):
            op = self.current_char
            self.advance()
            right = self.parse_factor()
            if op == '*':
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero.")
                left /= right
        return left

    def parse_factor(self) -> float:
        """Parse numbers, parentheses, and unary minus."""
        if self.current_char is None:
            raise ValueError("Unexpected end of expression.")

        if self.current_char == '-':
            self.advance()
            return -self.parse_power()

        if self.current_char == '(':
            self.advance()
            result = self.parse_expression()
            if self.current_char != ')':
                raise ValueError("Mismatched parentheses.")
            self.advance()
            return result

        return self.parse_number()

    def parse_number(self) -> float:
        """Parse a number (integer or floating point)."""
        start = self.pos
        if self.current_char is None or not self.current_char.isdigit() and self.current_char != '.':
            raise ValueError(f"Invalid token at position {self.pos}: '{self.current_char}'")
        
        has_dot = False
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if has_dot:
                    raise ValueError("Invalid number format: multiple decimal points.")
                has_dot = True
            self.advance()

        number_str = self.expr[start:self.pos]
        try:
            return float(number_str)
        except ValueError:
            raise ValueError(f"Invalid number at position {start}: '{number_str}'")

import pytest


def test_simple_addition():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 4") == 7.0


def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2+1)") == -3.0


def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 + (3 * 4)") == 14.0


def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")


def test_invalid_expression():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate("3 + * 4")