from __future__ import annotations
from typing import List, Union, Tuple
import re


class ExpressionEvaluator:
    """
    A recursive descent parser-based mathematical expression evaluator.
    Supports +, -, *, /, parentheses, unary minus, and floating-point numbers.
    Raises ValueError for invalid expressions or runtime errors.
    """

    def __init__(self) -> None:
        self._tokens: List[Tuple[str, Union[str, float]]] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr (str): The expression to evaluate.

        Returns:
            float: The result of the evaluation.

        Raises:
            ValueError: If the expression is invalid (e.g., mismatched parens, division by zero, etc.)
        """
        if not expr or expr.strip() == "":
            raise ValueError("Empty expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        try:
            result = self._parse_expression()
            if self._pos < len(self._tokens):
                raise ValueError(f"Unexpected token '{self._tokens[self._pos][0]}' after expression")
            return result
        except ZeroDivisionError:
            raise ValueError("Division by zero")

    def _tokenize(self, expr: str) -> List[Tuple[str, Union[str, float]]]:
        """
        Tokenize the input expression into a list of (type, value) tuples.

        Token types: 'NUMBER', 'PLUS', 'MINUS', 'MULTIPLY', 'DIVIDE', 'LPAREN', 'RPAREN'

        Returns:
            List[Tuple[str, Union[str, float]]]: List of (token_type, token_value)

        Raises:
            ValueError: If an invalid token is encountered.
        """
        tokens = []
        i = 0
        while i < len(expr):
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            elif ch.isdigit() or ch == '.':
                # Match number (including decimals)
                match = re.match(r'\d+(\.\d+)?', expr[i:])
                if not match:
                    raise ValueError(f"Invalid number at position {i}")
                num_str = match.group(0)
                tokens.append(('NUMBER', float(num_str)))
                i += len(num_str)
            elif ch == '+':
                tokens.append(('PLUS', '+'))
                i += 1
            elif ch == '-':
                tokens.append(('MINUS', '-'))
                i += 1
            elif ch == '*':
                tokens.append(('MULTIPLY', '*'))
                i += 1
            elif ch == '/':
                tokens.append(('DIVIDE', '/'))
                i += 1
            elif ch == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif ch == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
            else:
                raise ValueError(f"Invalid character '{ch}' at position {i}")
        return tokens

    def _current_token(self) -> Tuple[str, Union[str, float]]:
        """Return the current token or raise IndexError if at end."""
        return self._tokens[self._pos]

    def _consume(self, expected_type: str) -> Union[str, float]:
        """
        Consume and return the current token if it matches expected_type.

        Args:
            expected_type (str): Expected token type.

        Returns:
            Union[str, float]: Token value.

        Raises:
            ValueError: If token type doesn't match.
        """
        if self._pos >= len(self._tokens):
            raise ValueError(f"Unexpected end of expression")
        token_type, token_value = self._current_token()
        if token_type != expected_type:
            raise ValueError(f"Expected '{expected_type}', got '{token_type}'")
        self._pos += 1
        return token_value

    def _parse_expression(self) -> float:
        """
        Parse an expression: term { ('+' | '-') term }.

        Returns:
            float: Result of the expression.
        """
        result = self._parse_term()
        while self._pos < len(self._tokens) and self._current_token()[0] in ('PLUS', 'MINUS'):
            op = self._consume(self._current_token()[0])
            if op == '+':
                result += self._parse_term()
            else:  # op == '-'
                result -= self._parse_term()
        return result

    def _parse_term(self) -> float:
        """
        Parse a term: factor { ('*' | '/') factor }.

        Returns:
            float: Result of the term.
        """
        result = self._parse_factor()
        while self._pos < len(self._tokens) and self._current_token()[0] in ('MULTIPLY', 'DIVIDE'):
            op = self._consume(self._current_token()[0])
            if op == '*':
                result *= self._parse_factor()
            else:  # op == '/'
                divisor = self._parse_factor()
                if divisor == 0.0:
                    raise ZeroDivisionError()
                result /= divisor
        return result

    def _parse_factor(self) -> float:
        """
        Parse a factor: [ '+' | '-' ] factor | NUMBER | '(' expression ')'.

        Handles unary plus/minus and recursive unary operators.

        Returns:
            float: Result of the factor.
        """
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression")

        token_type, token_value = self._current_token()

        if token_type == 'PLUS':
            self._consume('PLUS')
            return +self._parse_factor()
        elif token_type == 'MINUS':
            self._consume('MINUS')
            return -self._parse_factor()
        elif token_type == 'NUMBER':
            self._consume('NUMBER')
            return token_value
        elif token_type == 'LPAREN':
            self._consume('LPAREN')
            result = self._parse_expression()
            if self._pos >= len(self._tokens) or self._current_token()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._consume('RPAREN')
            return result
        else:
            raise ValueError(f"Unexpected token '{token_type}' in factor")
