import re
from typing import List, Union, Optional

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        self.tokens: List[Union[str, float]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: A string containing the mathematical expression.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: {self.tokens[self.pos]}")
            
        return result

    def _tokenize(self, expr: str) -> None:
        """
        Converts the input string into a list of tokens (numbers and operators).
        Handles floating point numbers and unary minus detection.
        """
        # Regex to match numbers (int or float), operators, and parentheses
        # Note: We handle unary minus during parsing, but we need to distinguish 
        # it from subtraction. We'll tokenize '-' as an operator and resolve 
        # context in the parser.
        token_pattern = r'\s*(?:(\d+\.?\d*)|([+\-*/()]))\s*'
        matches = re.findall(token_pattern, expr)
        
        if not matches:
            # Check if there are non-whitespace characters that didn't match
            if re.search(r'\S', expr):
                raise ValueError(f"Invalid token in expression: {expr}")
            return

        for number, operator in matches:
            if number:
                self.tokens.append(float(number))
            elif operator:
                self.tokens.append(operator)
            else:
                # This case should theoretically not happen with the regex above
                # unless there's a logic gap, but good for safety.
                pass

    def _current_token(self) -> Optional[Union[str, float]]:
        """Returns the current token or None if end of input."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self) -> Union[str, float]:
        """Consumes and returns the current token, advancing the position."""
        token = self._current_token()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (lowest precedence).
        Grammar: expression -> term (('+' | '-') term)*
        """
        left = self._parse_term()
        
        while True:
            token = self._current_token()
            if token == '+':
                self._consume()
                right = self._parse_term()
                left += right
            elif token == '-':
                self._consume()
                right = self._parse_term()
                left -= right
            else:
                break
                
        return left

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (higher precedence).
        Grammar: term -> factor (('*' | '/') factor)*
        """
        left = self._parse_factor()
        
        while True:
            token = self._current_token()
            if token == '*':
                self._consume()
                right = self._parse_factor()
                left *= right
            elif token == '/':
                self._consume()
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
            else:
                break
                
        return left

    def _parse_factor(self) -> float:
        """
        Parses unary operators and primary values (highest precedence).
        Grammar: factor -> ('-' | '+') factor | '(' expression ')' | number
        """
        token = self._current_token()
        
        if token == '-':
            self._consume()
            return -self._parse_factor()
        elif token == '+':
            self._consume()
            return self._parse_factor()
        elif token == '(':
            self._consume()
            result = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses: expected ')'")
            self._consume()
            return result
        elif isinstance(token, float):
            self._consume()
            return token
        else:
            if token is None:
                raise ValueError("Unexpected end of expression")
            raise ValueError(f"Invalid token: {token}")
