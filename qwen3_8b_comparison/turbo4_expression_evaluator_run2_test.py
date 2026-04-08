from typing import Optional, Union
import re

class ExpressionEvaluator:
    """
    A class to evaluate mathematical expressions with support for:
    - Basic operations (+, -, *, /)
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers
    - Correct operator precedence
    - Raises ValueError for invalid expressions or errors
    """

    def __init__(self):
        self.tokens = []

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.

        Args:
            expr (str): The expression to evaluate.

        Returns:
            float: The result of the evaluation.

        Raises:
            ValueError: For invalid expressions, mismatched parentheses, division by zero, etc.
        """
        # Preprocess the expression
        expr = self.preprocess(expr)
        if not expr:
            raise ValueError("Empty expression")

        # Tokenize the expression
        self.tokenize(expr)

        # Parse and evaluate the expression
        return self.parse_expression()

    def preprocess(self, expr: str) -> str:
        """
        Preprocess the expression to handle unary minus and remove spaces.

        Args:
            expr (str): The input expression.

        Returns:
            str: The preprocessed expression.
        """
        # Remove all whitespace
        expr = expr.replace(" ", "")

        # Handle unary minus
        # Replace '-(' with ' - ('
        expr = re.sub(r'(-)(\()', r' \1 (', expr)
        # Replace '(-' with ' -'
        expr = re.sub(r'(-)([0-9.]+)', r' \1 \2', expr)
        # Replace '(-' with ' -'
        expr = re.sub(r'(-)([a-zA-Z])', r' \1 \2', expr)

        return expr

    def tokenize(self, expr: str) -> None:
        """
        Tokenize the expression into numbers, operators, and parentheses.

        Args:
            expr (str): The preprocessed expression.
        """
        # Token pattern: numbers, operators, parentheses
        token_pattern = r'([+-]?\d*\.?\d+|[+-*/()])'
        tokens = re.findall(token_pattern, expr)

        # Validate tokens
        if not tokens:
            raise ValueError("Invalid tokens")

        # Check for mismatched parentheses
        open_count = 0
        for token in tokens:
            if token == '(':
                open_count += 1
            elif token == ')':
                open_count -= 1
                if open_count < 0:
                    raise ValueError("Mismatched parentheses")
        if open_count != 0:
            raise ValueError("Mismatched parentheses")

        self.tokens = tokens

    def parse_expression(self) -> float:
        """
        Parse and evaluate the expression using recursive descent.

        Returns:
            float: The result of the expression.
        """
        return self.parse_additive()

    def parse_additive(self) -> float:
        """
        Parse additive expressions (addition and subtraction).

        Returns:
            float: The result of the additive expression.
        """
        result = self.parse_multiplicative()
        while self.current_token in ('+', '-'):
            op = self.current_token
            self.advance()
            right = self.parse_multiplicative()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def parse_multiplicative(self) -> float:
        """
        Parse multiplicative expressions (multiplication and division).

        Returns:
            float: The result of the multiplicative expression.
        """
        result = self.parse_unary()
        while self.current_token in ('*', '/'):
            op = self.current_token
            self.advance()
            right = self.parse_unary()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def parse_unary(self) -> float:
        """
        Parse unary expressions (unary minus and numbers).

        Returns:
            float: The result of the unary expression.
        """
        if self.current_token == '-':
            self.advance()
            return -self.parse_unary()
        elif self.current_token == '(':
            self.advance()
            result = self.parse_expression()
            if self.current_token != ')':
                raise ValueError("Mismatched parentheses")
            self.advance()
            return result
        else:
            return self.parse_number()

    def parse_number(self) -> float:
        """
        Parse a number (integer or float).

        Returns:
            float: The parsed number.
        """
        token = self.current_token
        if token in ('+', '-'):
            return self.parse_unary()
        else:
            return float(token)

    @property
    def current_token(self) -> str:
        """Return the current token."""
        return self.tokens[0] if self.tokens else None

    def advance(self) -> None:
        """Advance to the next token."""
        self.tokens.pop(0)