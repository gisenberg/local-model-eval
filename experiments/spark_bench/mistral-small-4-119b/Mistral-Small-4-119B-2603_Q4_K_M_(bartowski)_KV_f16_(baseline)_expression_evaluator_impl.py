from typing import Union, List, Tuple

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports +, -, *, / with correct precedence,
    parentheses, unary minus, and floating point numbers. Raises ValueError for errors.
    """

    def __init__(self):
        self.pos = 0
        self.expr = ""

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression to evaluate

        Returns:
            The result of the evaluated expression

        Raises:
            ValueError: If the expression contains mismatched parentheses, division by zero,
                       invalid tokens, or is empty
        """
        if not expr.strip():
            raise ValueError("Empty expression provided")

        self.expr = expr
        self.pos = 0
        result = self._parse_expression()
        return result

    def _parse_expression(self) -> float:
        """
        Parse an expression which may contain addition and subtraction operations.

        Returns:
            The result of the parsed expression
        """
        result = self._parse_term()

        while self.pos < len(self.expr):
            if self._match('+'):
                result += self._parse_term()
            elif self._match('-'):
                result -= self._parse_term()
            else:
                break

        return result

    def _parse_term(self) -> float:
        """
        Parse a term which may contain multiplication and division operations.

        Returns:
            The result of the parsed term
        """
        result = self._parse_factor()

        while self.pos < len(self.expr):
            if self._match('*'):
                result *= self._parse_factor()
            elif self._match('/'):
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                result /= divisor
            else:
                break

        return result

    def _parse_factor(self) -> float:
        """
        Parse a factor which may contain parentheses or unary minus.

        Returns:
            The result of the parsed factor
        """
        if self._match('('):
            result = self._parse_expression()
            if not self._match(')'):
                raise ValueError("Mismatched parentheses - missing closing parenthesis")
            return result

        return self._parse_number()

    def _parse_number(self) -> float:
        """
        Parse a number (integer or floating point) from the expression.

        Returns:
            The parsed number as a float

        Raises:
            ValueError: If no valid number is found at the current position
        """
        if self.pos >= len(self.expr):
            raise ValueError("Invalid expression - unexpected end of input")

        # Skip whitespace
        while self.pos < len(self.expr) and self.expr[self.pos].isspace():
            self.pos += 1

        if self.pos >= len(self.expr):
            raise ValueError("Invalid expression - unexpected end of input")

        # Handle unary minus
        sign = 1
        if self.expr[self.pos] == '-':
            sign = -1
            self.pos += 1
            # Skip whitespace after minus
            while self.pos < len(self.expr) and self.expr[self.pos].isspace():
                self.pos += 1

        # Extract number
        num_str = ""
        while self.pos < len(self.expr) and (self.expr[self.pos].isdigit() or self.expr[self.pos] == '.'):
            num_str += self.expr[self.pos]
            self.pos += 1

        if not num_str:
            raise ValueError("Invalid expression - expected a number")

        try:
            return sign * float(num_str)
        except ValueError:
            raise ValueError(f"Invalid expression - could not parse number '{num_str}'")

    def _match(self, expected: str) -> bool:
        """
        Check if the current position matches the expected character and advance if it does.

        Args:
            expected: The character to match against

        Returns:
            True if the character matches, False otherwise
        """
        if self.pos < len(self.expr) and self.expr[self.pos] == expected:
            self.pos += 1
            return True
        return False
