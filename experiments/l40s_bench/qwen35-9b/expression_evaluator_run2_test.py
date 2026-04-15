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
                       involves division by zero, or is empty.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty.")

        # Tokenize the input
        self.tokens = self._tokenize(expr)
        self.pos = 0
        self.error_message = None

        try:
            result = self._parse_expression()
            # Ensure we consumed all tokens
            if self.pos < len(self.tokens):
                raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
            return result
        except ZeroDivisionError:
            raise ValueError("Division by zero is not allowed.")
        except Exception as e:
            if self.error_message:
                raise ValueError(self.error_message)
            raise

    def _tokenize(self, expr: str) -> List[str]:
        """Converts the expression string into a list of tokens."""
        # Regex to match numbers (int or float), operators, and parentheses
        pattern = r'\d+\.?\d*|[+\-*/()]'
        tokens = re.findall(pattern, expr)

        # Check for invalid characters
        if re.search(r'[^0-9+\-*/().\s]', expr):
            raise ValueError("Invalid character in expression.")

        return tokens

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Handles unary minus by delegating to _parse_term.
        """
        left = self._parse_term()

        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if token == '+':
                self.pos += 1
                right = self._parse_term()
                left += right
            elif token == '-':
                self.pos += 1
                right = self._parse_term()
                left -= right
            else:
                break
        return left

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Handles unary minus here before processing binary operators.
        """
        # Check for unary minus
        if self.tokens[self.pos] == '-':
            self.pos += 1
            return -self._parse_term()
        
        if self.tokens[self.pos] == '+':
            self.pos += 1
            return self._parse_term()

        return self._parse_factor()

    def _parse_factor(self) -> float:
        """
        Parses numbers and parenthesized expressions (highest precedence).
        """
        token = self.tokens[self.pos]

        if token == '(':
            self.pos += 1  # Consume '('
            result = self._parse_expression()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing parenthesis.")
            self.pos += 1  # Consume ')'
            return result
        elif token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
            # Handle integer tokens like "123"
            # Note: Our tokenizer separates '.', so we need to reconstruct if needed,
            # but our regex \d+\.?\d* captures the whole number in one token usually.
            # Let's handle standard float parsing.
            val_str = token
            return float(val_str)
        else:
            raise ValueError(f"Invalid token: {token}")

import pytest

class TestExpressionEvaluator:
    def test_basic_arithmetic(self):
        """Test basic addition and subtraction."""
        evaluator = ExpressionEvaluator()
        assert evaluator.evaluate("5 + 3") == 8.0
        assert evaluator.evaluate("10 - 4") == 6.0
        assert evaluator.evaluate("2 + 3 - 1") == 4.0

    def test_operator_precedence(self):
        """Test that multiplication happens before addition."""
        evaluator = ExpressionEvaluator()
        # 2 + 3 * 4 should be 2 + 12 = 14
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        # (2 + 3) * 4 should be 20
        assert evaluator.evaluate("(2 + 3) * 4") == 20.0
        # 10 / 2 * 5 should be 25 (left to right for equal precedence)
        assert evaluator.evaluate("10 / 2 * 5") == 25.0

    def test_parentheses_grouping(self):
        """Test nested parentheses and grouping."""
        evaluator = ExpressionEvaluator()
        assert evaluator.evaluate("(2 + 3) * (4 - 1)") == 15.0
        assert evaluator.evaluate("((1 + 2) * 3)") == 9.0
        assert evaluator.evaluate("1 + (2 + (3 + 4))") == 10.0

    def test_unary_minus(self):
        """Test unary minus for negative numbers and grouped expressions."""
        evaluator = ExpressionEvaluator()
        assert evaluator.evaluate("-5 + 3") == -2.0
        assert evaluator.evaluate("3 - -5") == 8.0
        assert evaluator.evaluate("- (2 + 3)") == -5.0
        assert evaluator.evaluate("-(5 * 2)") == -10.0
        assert evaluator.evaluate("-3.5") == -3.5

    def test_error_cases(self):
        """Test ValueError for invalid inputs, division by zero, and empty strings."""
        evaluator = ExpressionEvaluator()

        # Empty expression
        with pytest.raises(ValueError):
            evaluator.evaluate("")

        # Mismatched parentheses
        with pytest.raises(ValueError):
            evaluator.evaluate("(1 + 2")
        
        with pytest.raises(ValueError):
            evaluator.evaluate("1 + 2)")

        # Division by zero
        with pytest.raises(ValueError):
            evaluator.evaluate("10 / 0")

        # Invalid token (e.g., letter)
        with pytest.raises(ValueError):
            evaluator.evaluate("1 + a")

        # Unexpected token at end
        with pytest.raises(ValueError):
            evaluator.evaluate("1 + 2 *")