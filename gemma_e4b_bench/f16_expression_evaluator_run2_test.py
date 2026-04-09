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
        tokens = []
        i = 0
        n = len(expr)
        
        # Regex to match numbers (including decimals) or operators/parentheses
        token_specification = [
            (self.TOKEN_NUMBER, r'\d+(\.\d*)?'),  # Floating point or integer
            (self.TOKEN_PLUS, r'\+'),
            (self.TOKEN_MINUS, r'-'),
            (self.TOKEN_MULTIPLY, r'\*'),
            (self.TOKEN_DIVIDE, r'/'),
            (self.TOKEN_LPAREN, r'\('),
            (self.TOKEN_RPAREN, r'\)'),
            (self.TOKEN_SKIP, r'\s+'),  # Whitespace to skip
        ]
        
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
        
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
                # Should not happen if regex is correct, but good for robustness
                raise ValueError(f"Invalid token found: {value}")

        tokens.append((self.TOKEN_EOF, ''))
        return tokens

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the given mathematical expression string.

        Args:
            expr: The mathematical expression string.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is invalid (e.g., division by zero, 
                        mismatched parentheses, invalid tokens, empty expression).
        """
        if not expr.strip():
            raise ValueError("Expression cannot be empty.")
            
        tokens = self._tokenize(expr)
        self.tokens = tokens
        self.pos = 0
        
        try:
            result = self._parse_expression()
            
            # Ensure all tokens were consumed
            if self.tokens[self.pos][0] != self.TOKEN_EOF:
                raise ValueError("Invalid expression structure: unexpected tokens remaining.")
                
            return result
        except IndexError:
            # Catches issues like unexpected EOF during parsing
            raise ValueError("Mismatched parentheses or incomplete expression.")
        except ZeroDivisionError:
            raise ValueError("Division by zero.")


    # --- Recursive Descent Parser Components ---
    # Grammar structure (simplified):
    # Expression -> Term { ('+' | '-') Term }
    # Term       -> Factor { ('*' | '/') Factor }
    # Factor     -> NUMBER | '(' Expression ')' | '-' Factor  <-- Handles unary minus

    def _peek(self) -> tuple[str, Union[str, float]]:
        """Returns the next token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (self.TOKEN_EOF, None)

    def _consume(self, expected_type: str = None) -> tuple[str, Union[str, float]]:
        """Consumes the current token and advances the position."""
        if self.pos >= len(self.tokens):
            raise IndexError("Unexpected end of expression.")
            
        token = self.tokens[self.pos]
        
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type} but found {token[0]} ('{token[1]}')")
        
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """Handles addition and subtraction (lowest precedence)."""
        result = self._parse_term()

        while self._peek()[0] in (self.TOKEN_PLUS, self.TOKEN_MINUS):
            op_type, op_val = self._consume()
            right = self._parse_term()
            
            if op_type == self.TOKEN_PLUS:
                result += right
            elif op_type == self.TOKEN_MINUS:
                result -= right
        
        return result

    def _parse_term(self) -> float:
        """Handles multiplication and division (medium precedence)."""
        result = self._parse_factor()

        while self._peek()[0] in (self.TOKEN_MULTIPLY, self.TOKEN_DIVIDE):
            op_type, op_val = self._consume()
            right = self._parse_factor()
            
            if op_type == self.TOKEN_MULTIPLY:
                result *= right
            elif op_type == self.TOKEN_DIVIDE:
                # Check for division by zero before performing the operation
                if right == 0.0:
                    raise ZeroDivisionError()
                result /= right
        
        return result

    def _parse_factor(self) -> float:
        """Handles numbers, parentheses, and unary minus (highest precedence)."""
        token = self._peek()
        token_type, token_value = token

        # 1. Unary Minus Check
        # A '-' is unary if it's the first token, or follows an opening parenthesis, 
        # or follows another operator.
        if token_type == self.TOKEN_MINUS:
            # Check context: if the previous token was an operator or '('
            # Since we don't explicitly track the previous token type easily here, 
            # we rely on the structure: if we are expecting a factor, and see '-', it's unary.
            # This is a common simplification in recursive descent for unary ops.
            self._consume(self.TOKEN_MINUS) # Consume the '-'
            # Recursively parse the next factor and negate it
            return -self._parse_factor()

        # 2. Parentheses
        elif token_type == self.TOKEN_LPAREN:
            self._consume(self.TOKEN_LPAREN)
            result = self._parse_expression()
            self._consume(self.TOKEN_RPAREN) # Must close the parenthesis
            return result

        # 3. Number
        elif token_type == self.TOKEN_NUMBER:
            self._consume(self.TOKEN_NUMBER)
            return token_value
        
        # 4. Error Case
        else:
            raise ValueError(f"Invalid token encountered where a factor was expected: {token_type} ('{token_value}')")

# --- Pytest Tests ---
import pytest

@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Fixture to provide a fresh evaluator instance for each test."""
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator: ExpressionEvaluator):
    """Tests basic addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 5") == 15.0
    assert evaluator.evaluate("10 / 2") == 5.0

def test_operator_precedence(evaluator: ExpressionEvaluator):
    """Tests correct operator precedence (* and / before + and -)."""
    # 2 + 3 * 4 = 2 + 12 = 14
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # 10 / 2 - 1 = 5 - 1 = 4
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    # 5 * (2 + 3) = 25
    assert evaluator.evaluate("5 * (2 + 3)") == 25.0

def test_parentheses_grouping(evaluator: ExpressionEvaluator):
    """Tests correct grouping using parentheses."""
    # (2 + 3) * 4 = 20
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # 10 / (2 + 3) = 2.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus(evaluator: ExpressionEvaluator):
    """Tests support for unary minus at the start and after operators."""
    # Unary minus at start
    assert evaluator.evaluate("-5") == -5.0
    # Unary minus after operator
    assert evaluator.evaluate("10 + -3") == 7.0
    # Unary minus with parentheses
    assert evaluator.evaluate("10 - (-2)") == 12.0
    # Complex unary minus
    assert evaluator.evaluate("-(2 + 1)") == -3.0

def test_floating_point_support(evaluator: ExpressionEvaluator):
    """Tests handling of floating-point numbers."""
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("10.0 / 3.0") == pytest.approx(3.3333333333333335)
    assert evaluator.evaluate("0.5 + 0.25") == 0.75

# --- Error Handling Tests ---

def test_division_by_zero(evaluator: ExpressionEvaluator):
    """Tests that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")

def test_mismatched_parentheses(evaluator: ExpressionEvaluator):
    """Tests mismatched opening or closing parentheses."""
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("1 + 2)")

def test_invalid_tokens(evaluator: ExpressionEvaluator):
    """Tests expressions containing invalid characters."""
    with pytest.raises(ValueError, match="Invalid token found"):
        evaluator.evaluate("1 $ 2")
    with pytest.raises(ValueError, match="Invalid token encountered"):
        evaluator.evaluate("1 + * 2")

def test_empty_expression(evaluator: ExpressionEvaluator):
    """Tests that an empty string raises ValueError."""
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("   ")

def test_incomplete_expression(evaluator: ExpressionEvaluator):
    """Tests expressions that end abruptly (e.g., missing operand)."""
    with pytest.raises(ValueError, match="Unexpected end of expression"):
        evaluator.evaluate("5 +")
    with pytest.raises(ValueError, match="Unexpected end of expression"):
        evaluator.evaluate("(")