import re
from typing import List, Optional, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers
    
    Raises ValueError for invalid syntax, mismatched parentheses, or division by zero.
    """

    def __init__(self):
        self.tokens: List[Tuple[str, str]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: The mathematical expression string.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at end of expression: {self.tokens[self.pos][1]}")
            
        return result

    def _tokenize(self, expr: str) -> None:
        """
        Converts the input string into a list of tokens.
        Tokens are tuples of (type, value).
        Types: 'NUMBER', 'PLUS', 'MINUS', 'MUL', 'DIV', 'LPAREN', 'RPAREN'
        """
        token_pattern = r'\s*(?:(\d+\.?\d*)|([+\-*/()]))'
        matches = re.findall(token_pattern, expr)
        
        self.tokens = []
        for num, op in matches:
            if num:
                self.tokens.append(('NUMBER', num))
            elif op:
                if op == '+':
                    self.tokens.append(('PLUS', op))
                elif op == '-':
                    self.tokens.append(('MINUS', op))
                elif op == '*':
                    self.tokens.append(('MUL', op))
                elif op == '/':
                    self.tokens.append(('DIV', op))
                elif op == '(':
                    self.tokens.append(('LPAREN', op))
                elif op == ')':
                    self.tokens.append(('RPAREN', op))
        
        # Check for invalid characters (those not matched by regex)
        # We reconstruct the string from tokens to see if anything was skipped
        reconstructed = "".join(t[1] for t in self.tokens)
        if reconstructed != "".join(expr.split()):
            # Find the first character that doesn't match
            clean_expr = "".join(expr.split())
            for i, char in enumerate(clean_expr):
                if char not in "0123456789.+-*/()":
                    raise ValueError(f"Invalid token: '{char}'")

    def _current_token(self) -> Optional[Tuple[str, str]]:
        """Returns the current token or None if at the end."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self, expected_type: Optional[str] = None) -> Tuple[str, str]:
        """
        Consumes the current token. If expected_type is provided, validates it.
        Raises ValueError if the token doesn't match or if end of input is reached.
        """
        token = self._current_token()
        if token is None:
            raise ValueError("Unexpected end of expression")
        
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[0]}")
            
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses an expression: Term { ('+' | '-') Term }
        Handles addition and subtraction (lowest precedence).
        """
        left = self._parse_term()
        
        while True:
            token = self._current_token()
            if token and token[0] in ('PLUS', 'MINUS'):
                op = token[0]
                self._consume()
                right = self._parse_term()
                if op == 'PLUS':
                    left += right
                else:
                    left -= right
            else:
                break
                
        return left

    def _parse_term(self) -> float:
        """
        Parses a term: Factor { ('*' | '/') Factor }
        Handles multiplication and division.
        """
        left = self._parse_factor()
        
        while True:
            token = self._current_token()
            if token and token[0] in ('MUL', 'DIV'):
                op = token[0]
                self._consume()
                right = self._parse_factor()
                if op == 'MUL':
                    left *= right
                else:
                    if right == 0:
                        raise ValueError("Division by zero")
                    left /= right
            else:
                break
                
        return left

    def _parse_factor(self) -> float:
        """
        Parses a factor: NUMBER | '(' Expression ')' | ('+' | '-') Factor
        Handles numbers, parentheses, and unary operators.
        """
        token = self._current_token()
        
        if token is None:
            raise ValueError("Unexpected end of expression")

        # Handle unary plus or minus
        if token[0] in ('PLUS', 'MINUS'):
            self._consume()
            val = self._parse_factor()
            if token[0] == 'MINUS':
                return -val
            return val

        # Handle numbers
        if token[0] == 'NUMBER':
            self._consume()
            return float(token[1])

        # Handle parentheses
        if token[0] == 'LPAREN':
            self._consume()
            val = self._parse_expression()
            if self._current_token() is None or self._current_token()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses: missing closing ')'")
            self._consume() # Consume ')'
            return val

        raise ValueError(f"Invalid token: {token[1]}")
