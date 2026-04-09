import re
, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports +, -, *, /, parentheses,
    floating-point numbers, and unary minus, implemented using a 
    recursive descent parser.
    """

    def __init__(self):
        # Token types
        self.TOKEN_NUMBER = 'NUMBER'
        self.TOKEN_PLUS = 'PLUS'
        self.TOKEN_MINUS = 'MINUS'
        self.TOKEN_MULTIPLY = 'MULTIPLY'
        self.TOKEN_DIVIDE = 'DIVIDE'
        self.TOKEN_LPAREN = 'LPAREN'
        self.TOKEN_RPAREN = 'RPAREN'
        self.TOKEN_EOF = 'EOF'

    def _tokenize(self, expr: str) -> List[tuple[str, str]]:
        """
        Converts the input string expression into a list of tokens.
        Handles numbers, operators, and parentheses.
        """
        if not expr.strip():
            raise ValueError("Empty expression provided.")

        # Regex to match numbers (including floats), operators, and parentheses
        # We use lookarounds to ensure operators/parentheses are matched as separate tokens
        token_specification = [
            (self.TOKEN_NUMBER, r'\d+(\.\d*)?|\.\d+'),  # Floating point numbers
            (self.TOKEN_PLUS, r'\+'),
            (self.TOKEN_MINUS, r'-'),
            (self.TOKEN_MULTIPLY, r'\*'),
            (self.TOKEN_DIVIDE, r'/'),
            (self.TOKEN_LPAREN, r'\('),
            (self.TOKEN_RPAREN, r'\)'),
            (self.TOKEN_SKIP, r'\s+'),  # Skip whitespace
        ]
        
        tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
        
        tokens = []
        for mo in re.finditer(tok_regex, expr):
            kind = mo.lastgroup
            value = mo.group()
            
            if kind == self.TOKEN_SKIP:
                continue
            elif kind == self.TOKEN_NUMBER:
                tokens.append((self.TOKEN_NUMBER, float(value)))
            elif kind in [self.TOKEN_PLUS, self.TOKEN_MINUS, self.TOKEN_MULTIPLY, self.TOKEN_DIVIDE, self.TOKEN_LPAREN, self.TOKEN_RPAREN]:
                tokens.append((kind, value))
            else:
                # This should ideally not be reached if the regex is exhaustive
                raise ValueError(f"Invalid token found: {value}")

        tokens.append((self.TOKEN_EOF, None))
        return tokens

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the given mathematical expression string.

        Args:
            expr: The mathematical expression string.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is invalid (e.g., syntax error, 
                        division by zero, mismatched parentheses).
        """
        tokens = self._tokenize(expr)
        self.tokens = tokens
        self.pos = 0
        
        try:
            result = self._parse_expression()
            
            # Check if we consumed all tokens (except EOF)
            if self.peek()[0] != self.TOKEN_EOF:
                raise ValueError("Invalid expression: unexpected tokens remaining.")
                
            return result
        except IndexError:
            # Catches issues like trying to read past EOF unexpectedly
            raise ValueError("Invalid expression syntax.")

    # --- Parser Helper Methods ---

    def peek(self) -> tuple[str, Union[float, None]]:
        """Returns the current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (self.TOKEN_EOF, None)

    def consume(self, expected_type: str = None) -> tuple[str, Union[float, None]]:
        """Consumes and returns the current token, optionally checking its type."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression.")
            
        token = self.tokens[self.pos]
        self.pos += 1
        
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Syntax error: Expected {expected_type} but found {token[0]} ('{token[1]}')")
        return token

    # Grammar Rules (Precedence: Expression -> Term -> Factor)

    def _parse_expression(self) -> float:
        """
        Handles addition (+) and subtraction (-). (Lowest Precedence)
        E -> T { (+|-) T }
        """
        left = self._parse_term()
        
        while self.peek()[0] in (self.TOKEN_PLUS, self.TOKEN_MINUS):
            op_type, op_val = self.consume()
            right = self._parse_term()
            
            if op_type == self.TOKEN_PLUS:
                left += right
            elif op_type == self.TOKEN_MINUS:
                left -= right
                
        return left

    def _parse_term(self) -> float:
        """
        Handles multiplication (*) and division (/). (Medium Precedence)
        T -> F { (*|/) F }
        """
        left = self._parse_factor()
        
        while self.peek()[0] in (self.TOKEN_MULTIPLY, self.TOKEN_DIVIDE):
            op_type, op_val = self.consume()
            right = self._parse_factor()
            
            if op_type == self.TOKEN_MULTIPLY:
                left *= right
            elif op_type == self.TOKEN_DIVIDE:
                if right == 0.0:
                    raise ValueError("Division by zero.")
                left /= right
                
        return left

    def _parse_factor(self) -> float:
        """
        Handles numbers, parentheses, and unary minus. (Highest Precedence)
        F -> NUMBER | '(' E ')' | - F
        """
        token_type, token_value = self.peek()

        # 1. Unary Minus Check
        # A minus sign is unary if it is the first token, or if it follows 
        # an opening parenthesis or another operator.
        if token_type == self.TOKEN_MINUS:
            # Check context: If the previous token was an operator or '(' or start of expression
            # Since we don't explicitly track the previous token type easily here, 
            # we check if the current token is the start of a factor that needs negation.
            # A simpler heuristic for this recursive structure: if the next token is 
            # not a number or ')' (which would imply binary subtraction), treat it as unary.
            
            # We check if the token *after* the minus sign is a number or '('
            next_token_type, _ = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else (self.TOKEN_EOF, None)
            
            if next_token_type in (self.TOKEN_NUMBER, self.TOKEN_LPAREN):
                # Consume the unary minus
                self.consume(self.TOKEN_MINUS)
                # Recursively evaluate the rest, then negate the result
                return -self._parse_factor()
            # If it's not unary (e.g., '3 - 4'), it will be handled by _parse_expression
        
        # 2. Parentheses
        if token_type == self.TOKEN_LPAREN:
            self.consume(self.TOKEN_LPAREN)
            result = self._parse_expression()
            self.consume(self.TOKEN_RPAREN)
            return result

        # 3. Number
        if token_type == self.TOKEN_NUMBER:
            return self.consume(self.TOKEN_NUMBER)[1]

        # 4. Error Case
        raise ValueError(f"Invalid token or syntax error at position {self.pos}: {token_type}")


if __name__ == '__main__':
    evaluator = ExpressionEvaluator()
    
    tests = [
        ("2 + 3 * 4", 14.0),
        ("(2 + 3) * 4", 20.0),
        ("10 / 2 - 1", 4.0),
        ("3.14 * (2 + 1) / 3", 3.14),
        ("-5 + 10", 5.0),
        ("-(2 + 3) * 4", -20.0),
        ("10 / -2", -5.0),
        ("-3.5 * 2", -7.0),
        ("10", 10.0),
    ]

    print("--- Running Basic Tests ---")
    for expr, expected in tests:
        try:
            result = evaluator.evaluate(expr)
            assert abs(result - expected) < 1e-9, f"FAIL: {expr} -> Got {result}, Expected {expected}"
            print(f"PASS: {expr} = {result}")
        except ValueError as e:
            print(f"ERROR: {expr} raised unexpected error: {e}")

    print("\n--- Running Error Tests ---")
    error_tests = [
        ("3 + (4 * 5", ValueError, "mismatched parentheses"),
        ("3 + 4))", ValueError, "mismatched parentheses"),
        ("10 / 0", ValueError, "Division by zero"),
        ("3 $ 4", ValueError, "Invalid token"),
        ("", ValueError, "Empty expression"),
        ("3 + * 4", ValueError, "Syntax error"),
    ]
    
    for expr, expected_exc, expected_msg in error_tests:
        try:
            evaluator.evaluate(expr)
            print(f"FAIL: {expr} -> Expected {expected_exc.__name__} but succeeded.")
        except expected_exc as e:
            if expected_msg in str(e):
                print(f"PASS: {expr} correctly raised {expected_exc.__name__}")
            else:
                print(f"FAIL: {expr} raised {expected_exc.__name__}, but message was unexpected: {e}")
        except Exception as e:
            print(f"FAIL: {expr} raised wrong exception: {type(e).__name__}")

import pytest


@pytest.fixture
def evaluator():
    """Fixture to provide a fresh evaluator instance for each test."""
    return ExpressionEvaluator()

# --- Test Cases for Correct Evaluation ---

@pytest.mark.parametrize("expression, expected", [
    # Basic arithmetic and precedence
    ("2 + 3 * 4", 14.0),
    ("(2 + 3) * 4", 20.0),
    ("10 / 2 - 1", 4.0),
    ("5 * (2 + 3) / 5", 5.0),
    
    # Floating point support
    ("3.14 * (2 + 1) / 3", 3.14),
    ("1.5 + 2.5", 4.0),
    
    # Unary Minus Support
    ("-5 + 10", 5.0),
    ("10 - (-3)", 13.0),
    ("-(2 + 3) * 4", -20.0),
    ("-3.5 * 2", -7.0),
    ("10 / -2", -5.0),
    
    # Complex combination
    ("100 / (2 * (5 - 3)) + -1", 51.0),
])
def test_valid_expressions(evaluator, expression, expected):
    """Tests various valid mathematical expressions."""
    result = evaluator.evaluate(expression)
    # Use pytest.approx for safe floating-point comparison
    assert result == pytest.approx(expected)

# --- Test Cases for Error Handling ---

@pytest.mark.parametrize("expression, expected_error_substring", [
    # Mismatched Parentheses
    ("3 + (4 * 5", "mismatched parentheses"),
    ("(3 + 4", "mismatched parentheses"),
    ("3 + 4))", "mismatched parentheses"),
    
    # Division by Zero
    ("10 / 0", "Division by zero"),
    ("5 / (2 - 2)", "Division by zero"),
    
    # Invalid Tokens/Syntax
    ("3 $ 4", "Invalid token"),
    ("3 + * 4", "Syntax error"), # Two operators in a row
    ("3 4", "Syntax error"),     # Missing operator
    
    # Empty Expression
    ("", "Empty expression provided"),
])
def test_invalid_expressions(evaluator, expression, expected_error_substring):
    """Tests that invalid expressions raise ValueError with expected messages."""
    with pytest.raises(ValueError) as excinfo:
        evaluator.evaluate(expression)
    
    assert expected_error_substring in str(excinfo.value)

def test_trailing_tokens_error(evaluator):
    """Tests for expressions that finish but have extra garbage tokens."""
    with pytest.raises(ValueError, match="unexpected tokens remaining"):
        evaluator.evaluate("2 + 3 4")

def test_empty_string_error(evaluator):
    """Tests the specific requirement for an empty input string."""
    with pytest.raises(ValueError, match="Empty expression provided"):
        evaluator.evaluate("  ")