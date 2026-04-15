import re
from typing import List, Tuple, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator supporting +, -, *, /, parentheses,
    and unary minus with correct operator precedence.
    """

    def __init__(self):
        self.pos = 0
        self.tokens: List[str] = []
        self.error_message: Optional[str] = None

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string.

        Args:
            expr: A string containing a valid mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is invalid, has mismatched parentheses,
                       involves division by zero, or contains invalid tokens.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")

        # Tokenize the expression
        self.tokens = self._tokenize(expr)
        if self.error_message:
            raise ValueError(self.error_message)

        self.pos = 0
        result = self._parse_expression()

        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: {self.tokens[self.pos]}")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of tokens (numbers, operators, parentheses).
        """
        token_pattern = r'\d+\.?\d*|[+\-*/()]'
        tokens = []
        
        # Regex to match numbers (int or float) and operators/parentheses
        for match in re.finditer(token_pattern, expr):
            token = match.group()
            if token == '-' and tokens and tokens[-1] in ('+', '-', '*', '/'):
                # Handle unary minus by converting to a special token or logic
                # We will handle unary minus in the parser logic specifically
                pass
            
            tokens.append(token)

        # Check for invalid characters
        if re.search(r'[^0-9+\-*/().\s]', expr):
            raise ValueError("Invalid character in expression.")

        return tokens

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Handles unary minus implicitly via the term parser.
        """
        left = self._parse_term()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_term()
            
            if op == '+':
                left = left + right
            else:
                left = left - right
                
        return left

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        """
        left = self._parse_factor()
        
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self._parse_factor()
            
            if op == '*':
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero.")
                left = left / right
                
        return left

    def _parse_factor(self) -> float:
        """
        Parses numbers, parentheses, and unary minus (highest precedence).
        """
        # Handle Unary Minus: if current token is '-', consume it and negate the result
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1
            value = self._parse_factor()
            return -value

        # Handle Unary Plus (optional, but good for completeness)
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '+':
            self.pos += 1
            return self._parse_factor()

        token = self.tokens[self.pos]
        
        if token == '(':
            self.pos += 1  # Consume '('
            value = self._parse_expression()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing parenthesis.")
            self.pos += 1  # Consume ')'
            return value
        elif token == ')':
            raise ValueError("Mismatched parentheses: unexpected closing parenthesis.")
        elif token.isdigit() or (token.startswith('.') and token[1:].isdigit()):
            self.pos += 1
            return float(token)
        else:
            raise ValueError(f"Invalid token: {token}")

import pytest

class TestExpressionEvaluator:
    def test_basic_arithmetic(self):
        """Tests basic addition, subtraction, multiplication, and division."""
        evaluator = ExpressionEvaluator()
        
        assert evaluator.evaluate("2 + 3") == 5.0
        assert evaluator.evaluate("10 - 4") == 6.0
        assert evaluator.evaluate("3 * 4") == 12.0
        assert evaluator.evaluate("15 / 3") == 5.0
        assert evaluator.evaluate("2 + 3 * 4") == 14.0  # Precedence check
        assert evaluator.evaluate("10 / (2 + 3)") == 2.0

    def test_parentheses_grouping(self):
        """Tests correct handling of parentheses for grouping."""
        evaluator = ExpressionEvaluator()
        
        # Standard grouping
        assert evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert evaluator.evaluate("2 * (3 + 4)") == 14.0
        
        # Nested parentheses
        assert evaluator.evaluate("((2 + 3) * 4) - 1") == 19.0
        assert evaluator.evaluate("2 + (3 * (4 + 5))") == 29.0
        
        # Complex nesting
        assert evaluator.evaluate("(1 + 2) * (3 + 4)") == 21.0

    def test_unary_minus(self):
        """Tests support for unary minus before numbers and groups."""
        evaluator = ExpressionEvaluator()
        
        # Unary minus before number
        assert evaluator.evaluate("-5 + 3") == -2.0
        assert evaluator.evaluate("-3 * 4") == -12.0
        assert evaluator.evaluate("5 + -3") == 2.0
        
        # Unary minus before parenthesis
        assert evaluator.evaluate("-(2 + 3)") == -5.0
        assert evaluator.evaluate("10 - (2 + 3)") == 5.0
        assert evaluator.evaluate("-(-(2 + 3))") == 5.0
        
        # Mixed unary and binary
        assert evaluator.evaluate("-3 + -4") == -7.0
        assert evaluator.evaluate("-(3 * 4)") == -12.0

    def test_floating_point_numbers(self):
        """Tests support for floating point literals."""
        evaluator = ExpressionEvaluator()
        
        assert evaluator.evaluate("3.14 + 2.86") == pytest.approx(6.0, rel=1e-9)
        assert evaluator.evaluate("1.5 * 2") == 3.0
        assert evaluator.evaluate("10 / 3.0") == pytest.approx(3.3333333333333335)
        assert evaluator.evaluate("-3.5 * 2") == -7.0
        assert evaluator.evaluate("1.1 + 2.2") == pytest.approx(3.3, rel=1e-9)

    def test_error_cases(self):
        """Tests ValueError raising for invalid inputs."""
        evaluator = ExpressionEvaluator()
        
        # Empty expression
        with pytest.raises(ValueError):
            evaluator.evaluate("")
        
        with pytest.raises(ValueError):
            evaluator.evaluate("   ")
        
        # Mismatched parentheses
        with pytest.raises(ValueError):
            evaluator.evaluate("(2 + 3")
        
        with pytest.raises(ValueError):
            evaluator.evaluate("2 + 3)")
        
        with pytest.raises(ValueError):
            evaluator.evaluate("((2 + 3)")
        
        # Division by zero
        with pytest.raises(ValueError):
            evaluator.evaluate("10 / 0")
        
        with pytest.raises(ValueError):
            evaluator.evaluate("5 / (2 - 2)")
        
        # Invalid tokens
        with pytest.raises(ValueError):
            evaluator.evaluate("2 + a")
        
        with pytest.raises(ValueError):
            evaluator.evaluate("2 + * 3")
        
        # Unexpected token at end (trailing operator)
        with pytest.raises(ValueError):
            evaluator.evaluate("2 +")