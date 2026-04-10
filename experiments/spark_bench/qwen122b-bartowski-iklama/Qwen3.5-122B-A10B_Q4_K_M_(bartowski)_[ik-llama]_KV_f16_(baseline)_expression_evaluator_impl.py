import re
from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self) -> None:
        self.tokens: List[Tuple[str, Union[str, float]]] = []
        self.pos: int = 0

    def _tokenize(self, expr: str) -> List[Tuple[str, Union[str, float]]]:
        """
        Converts the input string into a list of tokens.
        
        Args:
            expr: The mathematical expression string.
            
        Returns:
            A list of tuples (token_type, value).
            
        Raises:
            ValueError: If invalid characters are found or expression is empty.
        """
        if not expr or expr.isspace():
            raise ValueError("Empty expression")

        tokens = []
        i = 0
        length = len(expr)
        
        # Regex pattern for numbers (integers and floats)
        number_pattern = re.compile(r'\d+(\.\d+)?')
        
        while i < length:
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
            
            # Check for numbers
            if char.isdigit() or char == '.':
                match = number_pattern.match(expr, i)
                if match:
                    num_str = match.group()
                    # Validate number format (e.g., prevent multiple dots)
                    if num_str.count('.') > 1:
                        raise ValueError(f"Invalid number format: '{num_str}'")
                    tokens.append(('NUMBER', float(num_str)))
                    i = match.end()
                    continue
                else:
                    raise ValueError(f"Invalid token at position {i}: '{char}'")
            
            # Check for operators and parentheses
            if char == '+':
                tokens.append(('PLUS', '+'))
            elif char == '-':
                tokens.append(('MINUS', '-'))
            elif char == '*':
                tokens.append(('MUL', '*'))
            elif char == '/':
                tokens.append(('DIV', '/'))
            elif char == '(':
                tokens.append(('LPAREN', '('))
            elif char == ')':
                tokens.append(('RPAREN', ')'))
            else:
                raise ValueError(f"Invalid token at position {i}: '{char}'")
            
            i += 1
            
        return tokens

    def _peek(self) -> Tuple[str, Union[str, float]]:
        """
        Returns the current token without consuming it.
        
        Returns:
            The current token tuple.
        """
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ('EOF', None)

    def _consume(self) -> Tuple[str, Union[str, float]]:
        """
        Returns the current token and advances the position.
        
        Returns:
            The current token tuple.
        """
        token = self._peek()
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses an expression handling addition and subtraction (lowest precedence).
        Grammar: Expression -> Term { (+|-) Term }
        
        Returns:
            The evaluated float result.
        """
        left = self._parse_term()
        
        while True:
            token_type, _ = self._peek()
            if token_type == 'PLUS':
                self._consume()
                right = self._parse_term()
                left = left + right
            elif token_type == 'MINUS':
                self._consume()
                right = self._parse_term()
                left = left - right
            else:
                break
                
        return left

    def _parse_term(self) -> float:
        """
        Parses a term handling multiplication and division (higher precedence).
        Grammar: Term -> Factor { (*|/) Factor }
        
        Returns:
            The evaluated float result.
            
        Raises:
            ValueError: If division by zero occurs.
        """
        left = self._parse_factor()
        
        while True:
            token_type, _ = self._peek()
            if token_type == 'MUL':
                self._consume()
                right = self._parse_factor()
                left = left * right
            elif token_type == 'DIV':
                self._consume()
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
            else:
                break
                
        return left

    def _parse_factor(self) -> float:
        """
        Parses a factor handling numbers, parentheses, and unary operators.
        Grammar: Factor -> Number | ( Expression ) | (+|-) Factor
        
        Returns:
            The evaluated float result.
            
        Raises:
            ValueError: If parentheses are mismatched or unexpected tokens found.
        """
        token_type, value = self._peek()
        
        # Handle Unary Plus or Minus
        if token_type == 'PLUS':
            self._consume()
            return self._parse_factor()
        elif token_type == 'MINUS':
            self._consume()
            return -self._parse_factor()
        
        # Handle Numbers
        elif token_type == 'NUMBER':
            self._consume()
            return float(value)
        
        # Handle Parentheses
        elif token_type == 'LPAREN':
            self._consume()
            result = self._parse_expression()
            next_token_type, _ = self._peek()
            if next_token_type != 'RPAREN':
                raise ValueError("Mismatched parentheses: expected ')'")
            self._consume()
            return result
        
        else:
            raise ValueError(f"Unexpected token: '{value}'")

    def evaluate(self, expr: str) -> float:
        """
        Public method to evaluate a mathematical expression string.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The calculated float result.
            
        Raises:
            ValueError: For empty expressions, invalid tokens, mismatched 
                        parentheses, or division by zero.
        """
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Empty expression")
            
        result = self._parse_expression()
        
        # Ensure all tokens were consumed
        if self.pos < len(self.tokens):
            remaining = self.tokens[self.pos]
            raise ValueError(f"Unexpected token at end of expression: '{remaining[1]}'")
            
        return result
