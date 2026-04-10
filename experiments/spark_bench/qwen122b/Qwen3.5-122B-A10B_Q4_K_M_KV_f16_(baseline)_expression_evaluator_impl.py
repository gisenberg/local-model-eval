from typing import List, Union, Optional
import re

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        self.expression: str = ""
        self.pos: int = 0
        self.length: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.expression = expr.replace(" ", "")
        self.pos = 0
        self.length = len(self.expression)

        if self.length == 0:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        if self.pos < self.length:
            raise ValueError(f"Invalid token at position {self.pos}: '{self.expression[self.pos]}'")

        return result

    def _parse_expression(self) -> float:
        """
        Parses an expression handling addition and subtraction (lowest precedence).
        """
        result = self._parse_term()

        while self.pos < self.length:
            char = self.expression[self.pos]
            if char == '+':
                self.pos += 1
                result += self._parse_term()
            elif char == '-':
                self.pos += 1
                result -= self._parse_term()
            else:
                break
        
        return result

    def _parse_term(self) -> float:
        """
        Parses a term handling multiplication and division (higher precedence).
        """
        result = self._parse_factor()

        while self.pos < self.length:
            char = self.expression[self.pos]
            if char == '*':
                self.pos += 1
                result *= self._parse_factor()
            elif char == '/':
                self.pos += 1
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                result /= divisor
            else:
                break
        
        return result

    def _parse_factor(self) -> float:
        """
        Parses a factor handling unary operators and primary values (numbers/parentheses).
        """
        # Handle unary minus or plus
        if self.pos < self.length and self.expression[self.pos] in ('+', '-'):
            sign = 1
            if self.expression[self.pos] == '-':
                sign = -1
            self.pos += 1
            return sign * self._parse_factor()

        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parses a primary value: a number or a parenthesized expression.
        """
        if self.pos >= self.length:
            raise ValueError("Unexpected end of expression")

        char = self.expression[self.pos]

        if char == '(':
            self.pos += 1
            result = self._parse_expression()
            
            if self.pos >= self.length or self.expression[self.pos] != ')':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            
            self.pos += 1
            return result

        if char.isdigit() or char == '.':
            return self._parse_number()

        raise ValueError(f"Invalid token at position {self.pos}: '{char}'")

    def _parse_number(self) -> float:
        """
        Parses a floating point number from the current position.
        """
        start = self.pos
        has_dot = False

        while self.pos < self.length:
            char = self.expression[self.pos]
            if char.isdigit():
                self.pos += 1
            elif char == '.' and not has_dot:
                has_dot = True
                self.pos += 1
            else:
                break

        if start == self.pos:
            raise ValueError(f"Expected number at position {start}")

        number_str = self.expression[start:self.pos]
        
        # Validate format (e.g., prevent multiple dots if logic above failed, though loop handles it)
        if number_str.count('.') > 1:
            raise ValueError(f"Invalid number format: '{number_str}'")

        try:
            return float(number_str)
        except ValueError:
            raise ValueError(f"Invalid number: '{number_str}'")
