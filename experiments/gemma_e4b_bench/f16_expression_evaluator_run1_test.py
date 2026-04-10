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

        self.tokens: List[tuple[str, str]] = []
        self.pos: int = 0

    def _tokenize(self, expr: str) -> List[tuple[str, str]]:
        """
        Converts the input string expression into a list of tokens.
        Handles numbers, operators, and parentheses.
        """
        tokens = []
        i = 0
        n = len(expr)

        while i < n:
            char = expr[i]

            if char.isspace():
                i += 1
                continue

            # Numbers (including floats)
            if char.isdigit() or char == '.':
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                value = expr[start:i]
                tokens.append((self.TOKEN_NUMBER, value))
                continue

            # Operators and Parentheses
            elif char in '+-*/()':
                if char == '+':
                    tokens.append((self.TOKEN_PLUS, char))
                elif char == '-':
                    # We treat '-' as a potential unary or binary operator during tokenization.
                    # The parser will distinguish its role.
                    tokens.append((self.TOKEN_MINUS, char))
                elif char == '*':
                    tokens.append((self.TOKEN_MULTIPLY, char))
                elif char == '/':
                    tokens.append((self.TOKEN_DIVIDE, char))
                elif char == '(':
                    tokens.append((self.TOKEN_LPAREN, char))
                elif char == ')':
                    tokens.append((self.TOKEN_RPAREN, char))
                i += 1
                continue
            
            # Invalid token
            else:
                raise ValueError(f"Invalid token found: '{char}' at position {i}")

        tokens.append((self.TOKEN_EOF, 'EOF'))
        return tokens

    def _peek(self) -> tuple[str, str]:
        """Returns the current token without advancing."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (self.TOKEN_EOF, "")

    def _consume(self, expected_type: str = None, expected_value: str = None) -> tuple[str, str]:
        """
        Advances the pointer and returns the current token.
        Raises ValueError if the token does not match expectations.
        """
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression.")
        
        token = self.tokens[self.pos]
        self.pos += 1

        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected token type {expected_type}, but got {token[0]}")
        if expected_value and token[1] != expected_value:
            raise ValueError(f"Expected token value '{expected_value}', but got '{token[1]}'")
            
        return token

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the given mathematical expression string.

        Args:
            expr: The expression string (e.g., "3.14 * (2 + -1)").

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is invalid (e.g., division by zero, 
                        mismatched parentheses, invalid syntax).
        """
        if not expr.strip():
            raise ValueError("Expression cannot be empty.")

        # Reset state for a new evaluation
        self.tokens = self._tokenize(expr)
        self.pos = 0

        try:
            # Start parsing at the lowest precedence level (addition/subtraction)
            result = self._parse_expression()
            
            # Ensure we consumed the entire expression
            self._consume(self.TOKEN_EOF)
            return result
        
        except ValueError as e:
            # Re-raise parsing errors with context if necessary, or just re-raise
            raise e


    # --- Recursive Descent Parsing Methods ---
    # Grammar structure (simplified):
    # Expression -> Term { ('+' | '-') Term }
    # Term       -> Factor { ('*' | '/') Factor }
    # Factor     -> NUMBER | '(' Expression ')' | '-' Factor  <- Handles Unary Minus
    
    def _parse_expression(self) -> float:
        """Handles addition and subtraction (+, -)."""
        result = self._parse_term()

        while True:
            token = self._peek()
            if token[0] == self.TOKEN_PLUS:
                self._consume(self.TOKEN_PLUS)
                right = self._parse_term()
                result += right
            elif token[0] == self.TOKEN_MINUS:
                self._consume(self.TOKEN_MINUS)
                right = self._parse_term()
                result -= right
            else:
                break
        return result

    def _parse_term(self) -> float:
        """Handles multiplication and division (*, /)."""
        result = self._parse_factor()

        while True:
            token = self._peek()
            if token[0] == self.TOKEN_MULTIPLY:
                self._consume(self.TOKEN_MULTIPLY)
                right = self._parse_factor()
                result *= right
            elif token[0] == self.TOKEN_DIVIDE:
                self._consume(self.TOKEN_DIVIDE)
                right = self._parse_factor()
                if right == 0.0:
                    raise ValueError("Division by zero.")
                result /= right
            else:
                break
        return result

    def _parse_factor(self) -> float:
        """Handles numbers, parentheses, and unary minus."""
        token = self._peek()
        token_type, token_value = token

        if token_type == self.TOKEN_NUMBER:
            self._consume(self.TOKEN_NUMBER)
            return float(token_value)

        elif token_type == self.TOKEN_LPAREN:
            self._consume(self.TOKEN_LPAREN)
            # Recursively evaluate the sub-expression
            result = self._parse_expression()
            self._consume(self.TOKEN_RPAREN)
            return result

        elif token_type == self.TOKEN_MINUS:
            # Unary Minus: Consume '-' and then parse the next factor
            self._consume(self.TOKEN_MINUS)
            # The result of the next factor is negated
            return -self._parse_factor()
        
        else:
            # If it's not a number, '(', or '-', it's an invalid start to a factor
            raise ValueError(f"Invalid syntax or unexpected token: {token_value}")

# Example Usage (optional, for testing locally)
if __name__ == '__main__':
    evaluator = ExpressionEvaluator()
    
    expressions = [
        "3 + 4 * 2 / ( 1 - 5 )",  # Precedence test
        "10.5 * (2 + -3.5)",      # Float and Unary Minus
        "-(2 + 3) * 4",           # Unary Minus with parentheses
        "10 / 0",                 # Division by zero
        "((1 + 2)",               # Mismatched parentheses
        "3 $ 4",                  # Invalid token
        ""                        # Empty expression
    ]

    for expr in expressions:
        try:
            result = evaluator.evaluate(expr)
            print(f"Expression: '{expr}' = {result}")
        except ValueError as e:
            print(f"Expression: '{expr}' -> ERROR: {e}")

import pytest


@pytest.fixture
def evaluator():
    """Fixture to provide a fresh ExpressionEvaluator instance for each test."""
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator: ExpressionEvaluator):
    """Tests basic addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 5") == 15.0
    assert evaluator.evaluate("20 / 4") == 5.0

def test_operator_precedence(evaluator: ExpressionEvaluator):
    """Tests that multiplication/division happens before addition/subtraction."""
    # 3 + (4 * 2) / (1 - 5) = 3 + 8 / -4 = 3 + (-2) = 1.0
    assert evaluator.evaluate("3 + 4 * 2 / (1 - 5)") == 1.0
    # 20 / 5 + 2 * 3 = 4 + 6 = 10.0
    assert evaluator.evaluate("20 / 5 + 2 * 3") == 10.0

def test_parentheses_grouping(evaluator: ExpressionEvaluator):
    """Tests correct grouping using parentheses."""
    # (3 + 4) * 2 = 14.0
    assert evaluator.evaluate("(3 + 4) * 2") == 14.0
    # 20 / (5 - 1) = 5.0
    assert evaluator.evaluate("20 / (5 - 1)") == 5.0

def test_floating_point_support(evaluator: ExpressionEvaluator):
    """Tests support for floating-point numbers."""
    # 3.14 * 2.0 = 6.28
    assert evaluator.evaluate("3.14 * 2.0") == 6.28
    # 10.0 / 3.0 = 3.333...
    assert abs(evaluator.evaluate("10.0 / 3.0") - (10.0 / 3.0)) < 1e-9

def test_unary_minus(evaluator: ExpressionEvaluator):
    """Tests support for unary minus at the start and within expressions."""
    # -3 + 5 = 2.0
    assert evaluator.evaluate("-3 + 5") == 2.0
    # 10 - (-2) = 12.0
    assert evaluator.evaluate("10 - (-2)") == 12.0
    # -(2 + 1) = -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    # - (5 * (3 - 1)) = -10.0
    assert evaluator.evaluate("- (5 * (3 - 1))") == -10.0

def test_error_division_by_zero(evaluator: ExpressionEvaluator):
    """Tests that division by zero raises a ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

def test_error_mismatched_parentheses(evaluator: ExpressionEvaluator):
    """Tests that mismatched parentheses raise a ValueError."""
    with pytest.raises(ValueError):
        evaluator.evaluate("((1 + 2)")
    with pytest.raises(ValueError):
        evaluator.evaluate("(1 + 2))")

def test_error_invalid_tokens(evaluator: ExpressionEvaluator):
    """Tests that invalid characters raise a ValueError."""
    with pytest.raises(ValueError, match="Invalid token found: '$'"):
        evaluator.evaluate("3 $ 4")

def test_error_empty_expression(evaluator: ExpressionEvaluator):
    """Tests that an empty input string raises a ValueError."""
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("  ")

# --- Additional required tests (Total 5 minimum, providing more for robustness) ---

def test_complex_expression(evaluator: ExpressionEvaluator):
    """Tests a complex combination of all features."""
    # Expression: 10.0 / (2 + -1) * (3 - 1.5)
    # 10.0 / (1) * (1.5) = 10.0 * 1.5 = 15.0
    assert abs(evaluator.evaluate("10.0 / (2 + -1) * (3 - 1.5)") - 15.0) < 1e-9

def test_unary_minus_at_start_of_expression(evaluator: ExpressionEvaluator):
    """Tests unary minus as the very first element."""
    assert evaluator.evaluate("-5") == -5.0