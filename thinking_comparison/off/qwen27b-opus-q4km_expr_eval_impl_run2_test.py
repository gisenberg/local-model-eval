import re
from typing import List, Tuple, Iterator

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus
    - Floating point numbers
    """
    
    class Token:
        """Represents a single token in the expression."""
        def __init__(self, type_: str, value: float | str):
            self.type = type_
            self.value = value
        
        def __repr__(self):
            return f"Token({self.type}, {self.value!r})"
    
    def __init__(self):
        """Initialize the evaluator."""
        self.tokens: List[ExpressionEvaluator.Token] = []
        self.pos: int = 0
    
    def _tokenize(self, expr: str) -> List[ExpressionEvaluator.Token]:
        """
        Convert the input string into a list of tokens.
        
        Args:
            expr: The input expression string
            
        Returns:
            A list of Token objects
            
        Raises:
            ValueError: If an invalid character is encountered
        """
        tokens: List[ExpressionEvaluator.Token] = []
        pattern = r'(\d+\.?\d*|\.\d+|[+\-*/()])'
        matches = re.finditer(pattern, expr)
        
        prev_end = 0
        for match in matches:
            start, end = match.span()
            token_str = match.group(1)
            
            # Check for invalid characters between tokens
            if start > prev_end:
                invalid_char = expr[prev_end:start]
                raise ValueError(f"Invalid character: '{invalid_char}'")
            
            # Classify the token
            if token_str in '+-*/()':
                tokens.append(ExpressionEvaluator.Token(token_str, token_str))
            else:
                # It's a number
                try:
                    num_value = float(token_str)
                    tokens.append(ExpressionEvaluator.Token('NUMBER', num_value))
                except ValueError:
                    raise ValueError(f"Invalid number: '{token_str}'")
            
            prev_end = end
        
        return tokens
    
    def _peek(self) -> ExpressionEvaluator.Token | None:
        """
        Return the current token without consuming it.
        
        Returns:
            The current token, or None if at end of input
        """
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None
    
    def _consume(self) -> ExpressionEvaluator.Token:
        """
        Return and consume the current token.
        
        Returns:
            The current token
            
        Raises:
            ValueError: If at end of input
        """
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self.pos += 1
        return token
    
    def _expect(self, type_: str) -> ExpressionEvaluator.Token:
        """
        Consume a token of the expected type.
        
        Args:
            type_: The expected token type
            
        Returns:
            The consumed token
            
        Raises:
            ValueError: If the token type doesn't match
        """
        token = self._consume()
        if token.type != type_:
            raise ValueError(f"Expected '{type_}', got '{token.type}'")
        return token
    
    def _parse_expression(self) -> float:
        """
        Parse an expression (handles + and - with lowest precedence).
        
        Grammar: Expression -> Term (('+' | '-') Term)*
        
        Returns:
            The evaluated result as a float
        """
        result = self._parse_term()
        
        while True:
            token = self._peek()
            if token is None or token.type not in '+-':
                break
            
            op = self._consume().type
            
            if op == '+':
                result += self._parse_term()
            else:  # op == '-'
                result -= self._parse_term()
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse a term (handles * and / with medium precedence).
        
        Grammar: Term -> Factor (('*' | '/') Factor)*
        
        Returns:
            The evaluated result as a float
        """
        result = self._parse_factor()
        
        while True:
            token = self._peek()
            if token is None or token.type not in '*/':
                break
            
            op = self._consume().type
            right = self._parse_factor()
            
            if op == '*':
                result *= right
            else:  # op == '/'
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parse a factor (handles numbers, parentheses, and unary minus).
        
        Grammar: Factor -> Number | '(' Expression ')' | '-' Factor
        
        Returns:
            The evaluated result as a float
        """
        token = self._peek()
        
        if token is None:
            raise ValueError("Unexpected end of expression")
        
        # Handle unary minus
        if token.type == '-':
            self._consume()
            return -self._parse_factor()
        
        # Handle numbers
        if token.type == 'NUMBER':
            self._consume()
            return token.value  # type: ignore
        
        # Handle parentheses
        if token.type == '(':
            self._consume()
            result = self._parse_expression()
            self._expect(')')
            return result
        
        # Invalid token
        raise ValueError(f"Unexpected token: '{token.type}'")
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Args:
            expr: The expression string to evaluate
            
        Returns:
            The result of the evaluation as a float
            
        Raises:
            ValueError: For various error conditions including:
                - Empty or whitespace-only expressions
                - Invalid tokens
                - Mismatched parentheses
                - Division by zero
        """
        # Handle empty or whitespace-only expressions
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        # Tokenize the expression
        self.tokens = self._tokenize(expr)
        
        # Check for empty token list (shouldn't happen after above check, but be safe)
        if not self.tokens:
            raise ValueError("Empty expression")
        
        # Reset position
        self.pos = 0
        
        # Parse and evaluate
        result = self._parse_expression()
        
        # Check for leftover tokens (mismatched parentheses or extra content)
        if self.pos < len(self.tokens):
            remaining = self.tokens[self.pos:]
            raise ValueError(f"Unexpected token(s): {remaining}")
        
        return result

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("15 / 4") == 3.75

def test_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0

def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_errors(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3")