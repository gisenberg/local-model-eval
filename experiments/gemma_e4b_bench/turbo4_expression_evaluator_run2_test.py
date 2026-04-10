import re
, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser.

    Supports +, -, *, /, parentheses, floating-point numbers, and unary minus.
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
        tokens = []
        i = 0
        n = len(expr)
        
        # Regex to match numbers (including decimals) or single characters (operators/parens)
        token_specification = [
            (r'\d+\.\d+|\d+', self.TOKEN_NUMBER),  # Numbers
            (r'\+', self.TOKEN_PLUS),                # Addition
            (r'-', self.TOKEN_MINUS),                # Subtraction/Unary Minus
            (r'\*', self.TOKEN_MULTIPLY),          # Multiplication
            (r'/', self.TOKEN_DIVIDE),               # Division
            (r'\(', self.TOKEN_LPAREN),             # Left Parenthesis
            (r'\)', self.TOKEN_RPAREN),             # Right Parenthesis
            (r'\s+', None)                           # Whitespace (to be ignored)
        ]
        
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification if pair[1] is not None)
        
        for mo in re.finditer(tok_regex, expr):
            kind = mo.lastgroup
            value = mo.group()
            
            if kind == self.TOKEN_NUMBER:
                tokens.append((self.TOKEN_NUMBER, float(value)))
            elif kind == self.TOKEN_PLUS:
                tokens.append((self.TOKEN_PLUS, value))
            elif kind == self.TOKEN_MINUS:
                tokens.append((self.TOKEN_MINUS, value))
            elif kind == self.TOKEN_MULTIPLY:
                tokens.append((self.TOKEN_MULTIPLY, value))
            elif kind == self.TOKEN_DIVIDE:
                tokens.append((self.TOKEN_DIVIDE, value))
            elif kind == self.TOKEN_LPAREN:
                tokens.append((self.TOKEN_LPAREN, value))
            elif kind == self.TOKEN_RPAREN:
                tokens.append((self.TOKEN_RPAREN, value))
            # Whitespace is ignored by the regex structure
            
        tokens.append((self.TOKEN_EOF, None))
        return tokens

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the given mathematical expression string.

        Args:
            expr: The expression string (e.g., "3 * (4 + 5) / 2").

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is invalid (e.g., syntax error, 
                        division by zero, mismatched parentheses).
        """
        if not expr.strip():
            raise ValueError("Expression cannot be empty.")

        tokens = self._tokenize(expr)
        self.tokens = tokens
        self.pos = 0

        try:
            result = self._parse_expression()
            if self.peek_token()[0] != self.TOKEN_EOF:
                raise ValueError("Invalid tokens remaining after parsing.")
            return result
        except IndexError:
            raise ValueError("Mismatched parentheses or unexpected end of expression.")
        except Exception as e:
            # Catch any other parsing errors and wrap them as ValueErrors
            if not isinstance(e, ValueError):
                 raise ValueError(f"Parsing error: {e}")
            raise

    # --- Recursive Descent Parsing Methods ---

    def peek_token(self) -> tuple[str, Union[float, str]]:
        """Returns the current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (self.TOKEN_EOF, None)

    def consume_token(self, expected_type: str = None) -> tuple[str, Union[float, str]]:
        """Consumes and returns the current token, advancing the position."""
        if self.pos >= len(self.tokens):
            raise IndexError("Unexpected end of expression.")
        
        token = self.tokens[self.pos]
        self.pos += 1
        
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, but found {token[0]} ('{token[1]}')")
        
        return token

    def _parse_expression(self) -> float:
        """
        Handles addition (+) and subtraction (-). (Lowest precedence)
        Grammar: expression -> term { ('+' | '-') term }
        """
        result = self._parse_term()
        
        while self.peek_token()[0] in (self.TOKEN_PLUS, self.TOKEN_MINUS):
            op_token = self.consume_token()
            op = op_token[0]
            right = self._parse_term()
            
            if op == self.TOKEN_PLUS:
                result += right
            elif op == self.TOKEN_MINUS:
                result -= right
                
        return result

    def _parse_term(self) -> float:
        """
        Handles multiplication (*) and division (/). (Medium precedence)
        Grammar: term -> factor { ('*' | '/') factor }
        """
        result = self._parse_factor()
        
        while self.peek_token()[0] in (self.TOKEN_MULTIPLY, self.TOKEN_DIVIDE):
            op_token = self.consume_token()
            op = op_token[0]
            right = self._parse_factor()
            
            if op == self.TOKEN_MULTIPLY:
                result *= right
            elif op == self.TOKEN_DIVIDE:
                if right == 0.0:
                    raise ValueError("Division by zero.")
                result /= right
                
        return result

    def _parse_factor(self) -> float:
        """
        Handles numbers, parentheses, and unary minus. (Highest precedence)
        Grammar: factor -> NUMBER | '(' expression ')' | '-' factor
        """
        token = self.peek_token()
        token_type = token[0]
        
        # 1. Handle Unary Minus
        if token_type == self.TOKEN_MINUS:
            # Check if the minus is unary. It is unary if it's at the start, 
            # or follows an opening parenthesis, or follows another operator.
            # In this simplified parser structure, if we see a MINUS here, we treat it as unary
            # because the previous level (expression/term) would have consumed a value or operator.
            self.consume_token(self.TOKEN_MINUS) # Consume '-'
            # Recursively evaluate the next factor (e.g., -3 or -(2+1))
            return -self._parse_factor()

        # 2. Handle Parentheses
        elif token_type == self.TOKEN_LPAREN:
            self.consume_token(self.TOKEN_LPAREN) # Consume '('
            result = self._parse_expression()
            self.consume_token(self.TOKEN_RPAREN) # Consume ')'
            return result

        # 3. Handle Number
        elif token_type == self.TOKEN_NUMBER:
            # Consume the number token and return its value
            _, value = self.consume_token(self.TOKEN_NUMBER)
            return value
        
        else:
            # If it's not a number, '(', or '-', it's an invalid token in this context
            raise ValueError(f"Invalid token encountered: {token[0]} ('{token[1]}')")

# --- Pytest Setup ---
import pytest

@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Provides a fresh instance of the evaluator for each test."""
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator: ExpressionEvaluator):
    """Tests simple addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("2 * 5") == 10.0
    assert evaluator.evaluate("10 / 2") == 5.0

def test_operator_precedence(evaluator: ExpressionEvaluator):
    """Tests that multiplication/division takes precedence over addition/subtraction."""
    # 2 + 3 * 4 = 2 + 12 = 14
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # 10 / 2 - 1 = 5 - 1 = 4
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    # (3 + 5) * 2 = 16
    assert evaluator.evaluate("(3 + 5) * 2") == 16.0

def test_unary_minus_and_floating_point(evaluator: ExpressionEvaluator):
    """Tests unary minus and floating-point support."""
    # Unary minus at start
    assert evaluator.evaluate("-5.5") == -5.5
    # Unary minus after operator
    assert evaluator.evaluate("10 + -3") == 7.0
    # Complex unary minus
    assert evaluator.evaluate("-(2 + 1) * 3") == -9.0
    # Floating point division
    assert evaluator.evaluate("10.5 / 2") == 5.25

def test_error_handling(evaluator: ExpressionEvaluator):
    """Tests various error conditions."""
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Unexpected end of expression"):
        evaluator.evaluate("(5 + 2")
    
    with pytest.raises(ValueError, match="Invalid tokens remaining"):
        evaluator.evaluate("5 + 2)")

    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token encountered"):
        evaluator.evaluate("5 $ 2")

    # Empty expression
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate(" ")

# To run tests:
# 1. Save the code above as expression_evaluator.py
# 2. Install pytest: pip install pytest
# 3. Run: pytest expression_evaluator.py