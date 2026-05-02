import re
from enum import Enum, auto
from typing import List, Union
from dataclasses import dataclass


class TokenType(Enum):
    """Enumeration for token types supported by the evaluator."""
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


@dataclass
class Token:
    """Represents a lexical token."""
    type: TokenType
    value: Union[float, None] = None  # For NUMBER tokens


class ExpressionEvaluator:
    """
    A recursive descent mathematical expression evaluator.
    
    Supports basic arithmetic (+, -, *, /), parentheses, unary negation, 
    and floating point numbers. Uses a two-stage process: tokenization
    followed by recursive descent parsing.
    
    Attributes:
        tokens (List[Token]): The list of parsed tokens from the expression string.
        pos (int): Current position in the token stream.
        
    Methods:
        evaluate(str) -> float: Evaluates the input expression string.
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr (str): The mathematical expression to evaluate.
            
        Returns:
            float: The calculated result.
            
        Raises:
            ValueError: If the expression contains invalid syntax, mismatched
                        parentheses, division by zero, or is empty.
        """
        # Handle empty or whitespace-only input
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        # Phase 1: Tokenization
        self._tokenize(expr)
        
        # Phase 2: Parse and Evaluate
        self.pos = 0
        result = self._parse_expression()
        
        # Check for trailing content after parsing
        if self._current_type() != TokenType.EOF:
            raise ValueError(f"Invalid tokens found after valid expression: {self._current_value()}")
            
        return result

    def _tokenize(self, expr: str) -> None:
        """
        Converts the raw string into a list of Token objects.
        
        Raises:
            ValueError: If unknown characters are found in the expression.
        """
        # Pattern matches:
        # 1. Numbers (int or float): \d+(\.\d+)?
        # 2. Operators and parens: [+\-*/()]
        # 3. Whitespace is skipped
        pattern = r'\s*(\d+(?:\.\d+)?|[+\-*/()])'
        matcher = re.finditer(pattern, expr)
        
        collected_tokens = []
        
        for match in matcher:
            matched_str = match.group(1)
            try:
                num_val = float(matched_str)
                collected_tokens.append(Token(TokenType.NUMBER, num_val))
            except ValueError:
                # It's an operator or parenthesis
                char_map = {
                    '+': TokenType.PLUS,
                    '-': TokenType.MINUS,
                    '*': TokenType.MUL,
                    '/': TokenType.DIV,
                    '(': TokenType.LPAREN,
                    ')': TokenType.RPAREN
                }
                collected_tokens.append(Token(char_map[matched_str]))
        
        # Validate that the entire string was covered by tokens (catches invalid chars)
        # We reconstruct the cleaned string to compare length
        # Simpler approach: Iterate over the original string skipping whitespace, ensure matches align
        # A cleaner way for strict validation given our pattern:
        clean_expr = "".join(re.split(r'\s+', expr.replace(' ', '')))
        reconstructed = "".join(m.group(1) for m in re.finditer(pattern, expr))
        
        # Since finditer skips whitespace, we need to ensure invalid chars weren't ignored
        # We'll scan manually to detect "invalid chars"
        scanner_index = 0
        for token_match in re.finditer(pattern, expr):
            # Advance scanner index to start of match
            if token_match.start() > scanner_index:
                segment = expr[scanner_index:token_match.start()]
                if segment.strip(): # If non-empty non-whitespace exists between tokens
                     raise ValueError(f"Invalid characters in expression: '{segment}'")
            scanner_index = token_match.end()
            
        # Check end of string
        if scanner_index < len(expr):
            remaining = expr[scanner_index:]
            if remaining.strip():
                raise ValueError(f"Invalid characters in expression: '{remaining}'")

        self.tokens = collected_tokens + [Token(TokenType.EOF)]

    # --- Parsing Logic (Recursive Descent) ---

    def _current_token(self) -> Token:
        """Returns the current token at the parse head."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1]

    def _current_type(self) -> TokenType:
        """Returns the type of the current token."""
        return self._current_token().type

    def _current_value(self) -> Union[float, None]:
        """Returns the value of the current token if applicable."""
        return self._current_token().value

    def _consume(self) -> Token:
        """Consumes the current token and advances position."""
        token = self._current_token()
        self.pos += 1
        return token

    def _expect(self, expected_type: TokenType) -> None:
        """
        Ensures the current token matches the expected type.
        
        Raises:
            ValueError: If token types do not match.
        """
        if self._current_type() != expected_type:
            raise ValueError(f"Expected {expected_type.name}, got {self._current_type().name}")
        self._consume()

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction (Lowest precedence).
        Expr -> Term { ('+'|'-') Term }*
        """
        left = self._parse_term()
        
        while self._current_type() in (TokenType.PLUS, TokenType.MINUS):
            op = self._current_type()
            self._consume()
            right = self._parse_term()
            if op == TokenType.PLUS:
                left = left + right
            else:
                left = left - right
                
        return left

    def _parse_term(self) -> float:
        """
        Parses multiplication and division (Higher precedence).
        Term -> Factor { ('*'|'/') Factor }*
        """
        left = self._parse_factor()
        
        while self._current_type() in (TokenType.MUL, TokenType.DIV):
            op = self._current_type()
            self._consume()
            right = self._parse_factor()
            
            if op == TokenType.MUL:
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
                
        return left

    def _parse_factor(self) -> float:
        """
        Parses atoms, grouping, and unary operators (Highest precedence).
        Factor -> Number | '(' Expr ')' | ('+'|'-') Factor
        """
        token_type = self._current_type()
        
        # Handle Unary Plus/Minus
        if token_type == TokenType.PLUS:
            self._consume()
            return self._parse_factor()
            
        if token_type == TokenType.MINUS:
            self._consume()
            return -self._parse_factor()
            
        # Handle Number
        if token_type == TokenType.NUMBER:
            value = self._current_value()
            self._consume()
            return value
            
        # Handle Grouping (Parentheses)
        if token_type == TokenType.LPAREN:
            self._consume()
            result = self._parse_expression()
            self._expect(TokenType.RPAREN)
            return result
            
        # Catch-all for unexpected start of factor
        raise ValueError(f"Unexpected token in expression: {self._current_type().name}")


def run_tests():
    """Simple test runner script (run directly to verify basic functionality)."""
    import pytest
    
    # Define test functions here for standalone execution compatibility
    # though the prompt requests a separate file structure conceptually.
    pass


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------

def test_addition_and_multiplication_precedence():
    """Tests that multiplication occurs before addition."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0

def test_unary_minus_and_parentheses():
    """Tests unary operators and parenthesis grouping."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 * -(4 + 1)") == -15.0

def test_division_and_division_by_zero():
    """Tests floating point division and ValueError on div by zero."""
    evaluator = ExpressionEvaluator()
    assert abs(evaluator.evaluate("10 / 4") - 2.5) < 0.0001
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

def test_mismatched_parentheses_and_invalid_syntax():
    """Tests syntax error handling for parens and invalid tokens."""
    evaluator = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="mismatched|parentheses|Unexpected token"):
        evaluator.evaluate("(1 + 2")
        
    with pytest.raises(ValueError, match="Expected"):
        evaluator.evaluate("5 + ) 2")

    with pytest.raises(ValueError, match="Invalid characters"):
        evaluator.evaluate("1 + b")

def test_empty_expression_and_whitespace_handling():
    """Tests behavior on empty input and complex spacing."""
    evaluator = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Empty"):
        evaluator.evaluate("")
        
    with pytest.raises(ValueError, match="Empty"):
        evaluator.evaluate("   ")

    assert evaluator.evaluate("  10   +   20  ") == 30.0


if __name__ == "__main__":
    # To run these tests with pytest from command line:
    # pytest filename.py -v
    print("Run with 'pytest [filename].py' to execute tests.")