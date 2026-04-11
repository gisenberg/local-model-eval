import re
from typing import List, Optional, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports basic arithmetic operations,
    operator precedence, parentheses, and unary minus.
    
    Uses recursive descent parsing to evaluate expressions without using eval().
    """
    
    def __init__(self):
        """Initialize the tokenizer and parser."""
        self.tokens: List[str] = []
        self.current_token_index: int = 0
        self.current_token: Optional[str] = None
    
    def tokenize(self, expr: str) -> List[str]:
        """
        Tokenize the input expression into numbers, operators, and parentheses.
        
        Args:
            expr: The mathematical expression to tokenize
            
        Returns:
            List of tokens
            
        Raises:
            ValueError: If invalid tokens are found
        """
        # Remove all whitespace
        expr = expr.replace(' ', '')
        
        # Regular expression to match numbers (including negative numbers at start)
        # and operators/parentheses
        token_pattern = r"""
            (?P<NUMBER>-?\d+\.\d+|-?\d+\.|\d+\.?\d*)  # Numbers (including floats)
            |(?P<PLUS>\+)
            |(?P<MINUS>-)
            |(?P<MUL>\*)
            |(?P<DIV>/)
            |(?P<LPAREN>\()
            |(?P<RPAREN>\))
        """
        
        tokens = []
        for match in re.finditer(token_pattern, expr, re.VERBOSE):
            token_type = match.lastgroup
            token_value = match.group(token_type)
            
            if token_type == 'NUMBER':
                # Convert to float if it has a decimal point, otherwise int
                if '.' in token_value:
                    tokens.append(float(token_value))
                else:
                    tokens.append(int(token_value))
            else:
                tokens.append(token_value)
        
        return tokens
    
    def next_token(self) -> None:
        """Advance to the next token in the token list."""
        self.current_token_index += 1
        if self.current_token_index < len(self.tokens):
            self.current_token = self.tokens[self.current_token_index]
        else:
            self.current_token = None
    
    def parse_expression(self) -> float:
        """
        Parse and evaluate an expression (entry point for the parser).
        
        Returns:
            The result of the evaluated expression
            
        Raises:
            ValueError: For various parsing errors
        """
        if not self.tokens:
            raise ValueError("Empty expression")
        
        self.current_token_index = -1
        self.next_token()
        
        result = self.parse_additive()
        
        if self.current_token is not None:
            raise ValueError(f"Unexpected token: {self.current_token}")
        
        return result
    
    def parse_additive(self) -> float:
        """
        Parse additive operations (+ and -) with correct precedence.
        
        Returns:
            The result of the parsed additive expression
        """
        left = self.parse_multiplicative()
        
        while self.current_token in ('+', '-'):
            op = self.current_token
            self.next_token()
            right = self.parse_multiplicative()
            
            if op == '+':
                left += right
            else:
                left -= right
        
        return left
    
    def parse_multiplicative(self) -> float:
        """
        Parse multiplicative operations (* and /) with correct precedence.
        
        Returns:
            The result of the parsed multiplicative expression
        """
        left = self.parse_unary()
        
        while self.current_token in ('*', '/'):
            op = self.current_token
            self.next_token()
            right = self.parse_unary()
            
            if op == '*':
                left *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
        
        return left
    
    def parse_unary(self) -> float:
        """
        Parse unary minus operations.
        
        Returns:
            The result of the parsed unary expression
        """
        if self.current_token == '-':
            self.next_token()
            return -self.parse_primary()
        
        return self.parse_primary()
    
    def parse_primary(self) -> float:
        """
        Parse primary expressions (numbers and parenthesized expressions).
        
        Returns:
            The result of the parsed primary expression
        """
        if self.current_token is None:
            raise ValueError("Unexpected end of expression")
        
        if self.current_token == '(':
            self.next_token()
            expr = self.parse_additive()
            if self.current_token != ')':
                raise ValueError("Mismatched parentheses")
            self.next_token()
            return expr
        
        if isinstance(self.current_token, (int, float)):
            value = self.current_token
            self.next_token()
            return value
        
        raise ValueError(f"Invalid token: {self.current_token}")
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Args:
            expr: The mathematical expression to evaluate
            
        Returns:
            The result of the evaluated expression
            
        Raises:
            ValueError: For various parsing errors
        """
        try:
            self.tokens = self.tokenize(expr)
            return self.parse_expression()
        except ValueError as e:
            raise ValueError(f"Error evaluating expression '{expr}': {str(e)}")
