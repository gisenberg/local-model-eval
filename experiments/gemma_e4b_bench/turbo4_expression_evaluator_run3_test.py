import re
, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser.

    Supports +, -, *, /, parentheses, floating-point numbers, and unary minus.
    """

    def __init__(self):
        # Token types for clarity
        self.TOKEN_NUMBER = 'NUMBER'
        self.TOKEN_PLUS = 'PLUS'
        self.TOKEN_MINUS = 'MINUS'
        self.TOKEN_MULTIPLY = 'MULTIPLY'
        self.TOKEN_DIVIDE = 'DIVIDE'
        self.TOKEN_LPAREN = 'LPAREN'
        self.TOKEN_RPAREN = 'RPAREN'

    def _tokenize(self, expr: str) -> List[tuple[str, str]]:
        """
        Converts the input string expression into a list of tokens.
        Handles whitespace and identifies numbers, operators, and parentheses.
        """
        if not expr:
            raise ValueError("Empty expression provided.")

        # Regex pattern to match numbers (including decimals), operators, and parentheses
        # We use lookarounds to ensure we capture operators correctly, even if they are adjacent to numbers.
        token_specification = [
            (r'\s+', None),  # Whitespace (skip)
            (r'\d+\.\d+|\d+', self.TOKEN_NUMBER),  # Floating point or integer
            (r'\(', self.TOKEN_LPAREN),
            (r'\)', self.TOKEN_RPAREN),
            (r'\+', self.TOKEN_PLUS),
            (r'-', self.TOKEN_MINUS), # '-' can be binary or unary
            (r'\*', self.TOKEN_MULTIPLY),
            (r'/', self.TOKEN_DIVIDE),
        ]
        
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
        tokens = []
        
        # Use re.finditer to process the string sequentially
        for mo in re.finditer(tok_regex, expr):
            kind = mo.lastgroup
            value = mo.group()

            if kind is None:  # Whitespace
                continue
            
            if kind == self.TOKEN_NUMBER:
                tokens.append((self.TOKEN_NUMBER, float(value)))
            elif kind == self.TOKEN_LPAREN:
                tokens.append((self.TOKEN_LPAREN, value))
            elif kind == self.TOKEN_RPAREN:
                tokens.append((self.TOKEN_RPAREN, value))
            elif kind == self.TOKEN_PLUS:
                tokens.append((self.TOKEN_PLUS, value))
            elif kind == self.TOKEN_MULTIPLY:
                tokens.append((self.TOKEN_MULTIPLY, value))
            elif kind == self.TOKEN_DIVIDE:
                tokens.append((self.TOKEN_DIVIDE, value))
            elif kind == self.TOKEN_MINUS:
                # We keep '-' as a generic MINUS token for now; the parser will distinguish unary vs binary.
                tokens.append((self.TOKEN_MINUS, value))
            else:
                # Should not happen if regex is correct
                raise ValueError(f"Invalid token encountered: {value}")
                
        return tokens

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the mathematical expression string.

        Args:
            expr: The mathematical expression string.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is invalid (e.g., mismatched parentheses, 
                        division by zero, invalid syntax).
        """
        tokens = self._tokenize(expr)
        
        # We use a simple iterator/index approach for the recursive descent parser
        self.tokens = tokens
        self.pos = 0
        
        try:
            result = self._parse_expression()
            
            # After parsing, ensure all tokens were consumed
            if self.pos != len(tokens):
                raise ValueError("Invalid expression: Extra tokens remaining after parsing.")
                
            return result
        
        except IndexError:
            # Catches premature end of input during parsing (e.g., '3+')
            raise ValueError("Invalid expression: Unexpected end of input.")


    # --- Recursive Descent Parser Methods ---

    def _peek(self) -> tuple[str, Union[float, str]]:
        """Returns the current token without consuming it."""
        if self.pos >= len(self.tokens):
            raise IndexError("Attempted to peek past end of tokens.")
        return self.tokens[self.pos]

    def _consume(self, expected_type: str = None) -> tuple[str, Union[float, str]]:
        """Consumes the current token and advances the position."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression.")
            
        token = self.tokens[self.pos]
        
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Syntax Error: Expected {expected_type} but found {token[0]} ('{token[1]}')")
        
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Handles addition (+) and subtraction (-). (Lowest Precedence)
        Expression -> Term { ('+' | '-') Term }
        """
        left = self._parse_term()

        while self.pos < len(self.tokens):
            token_type, _ = self._peek()
            
            if token_type in (self.TOKEN_PLUS, self.TOKEN_MINUS):
                op_type, _ = self._consume()
                right = self._parse_term()
                
                if op_type == self.TOKEN_PLUS:
                    left += right
                else: # TOKEN_MINUS
                    left -= right
            else:
                break
        return left

    def _parse_term(self) -> float:
        """
        Handles multiplication (*) and division (/). (Medium Precedence)
        Term -> Factor { ('*' | '/') Factor }
        """
        left = self._parse_factor()

        while self.pos < len(self.tokens):
            token_type, _ = self._peek()
            
            if token_type in (self.TOKEN_MULTIPLY, self.TOKEN_DIVIDE):
                op_type, _ = self._consume()
                right = self._parse_factor()
                
                if op_type == self.TOKEN_MULTIPLY:
                    left *= right
                else: # TOKEN_DIVIDE
                    if right == 0.0:
                        raise ValueError("Division by zero.")
                    left /= right
            else:
                break
        return left

    def _parse_factor(self) -> float:
        """
        Handles numbers, parentheses, and unary minus. (Highest Precedence)
        Factor -> Number | '(' Expression ')' | '-' Factor
        """
        token_type, value = self._peek()

        # 1. Handle Unary Minus
        if token_type == self.TOKEN_MINUS:
            self._consume(self.TOKEN_MINUS) # Consume the '-'
            # Recursively evaluate the rest, treating it as a negative factor
            return -self._parse_factor()

        # 2. Handle Parentheses
        elif token_type == self.TOKEN_LPAREN:
            self._consume(self.TOKEN_LPAREN) # Consume '('
            result = self._parse_expression()
            self._consume(self.TOKEN_RPAREN) # Consume ')'
            return result

        # 3. Handle Number
        elif token_type == self.TOKEN_NUMBER:
            return self._consume(self.TOKEN_NUMBER)[1]
        
        # 4. Error Case
        else:
            raise ValueError(f"Invalid token encountered at start of factor: {token_type}")

# =============================================================================
# PYTEST TESTS
# =============================================================================
import pytest

@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Fixture to provide a fresh evaluator instance for each test."""
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator: ExpressionEvaluator):
    """Tests simple addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 5") == 15.0
    assert evaluator.evaluate("10 / 2") == 5.0

def test_operator_precedence(evaluator: ExpressionEvaluator):
    """Tests that multiplication/division takes precedence over addition/subtraction."""
    # 2 + 3 * 4 = 2 + 12 = 14
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # 10 / 2 - 1 = 5 - 1 = 4
    assert evaluator.evaluate("10 / 2 - 1") == 4.0
    # 1 + 2 * 3 / 6 = 1 + 1 = 2
    assert evaluator.evaluate("1 + 2 * 3 / 6") == 2.0

def test_parentheses_grouping(evaluator: ExpressionEvaluator):
    """Tests correct grouping using parentheses."""
    # (2 + 3) * 4 = 5 * 4 = 20
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # 10 / (2 - 1) = 10 / 1 = 10
    assert evaluator.evaluate("10 / (2 - 1)") == 10.0

def test_unary_minus(evaluator: ExpressionEvaluator):
    """Tests support for unary minus at the start and within expressions."""
    # Unary minus at start
    assert evaluator.evaluate("-5") == -5.0
    # Unary minus applied to a number
    assert evaluator.evaluate("10 + -3") == 7.0
    # Unary minus applied to a parenthesized expression
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    # Complex unary minus
    assert evaluator.evaluate("--5") == 5.0 # Double negative

def test_floating_point_numbers(evaluator: ExpressionEvaluator):
    """Tests handling of floating-point numbers."""
    assert evaluator.evaluate("3.14 + 1.0") == 4.14
    assert evaluator.evaluate("0.5 * 2.5") == 1.25
    assert evaluator.evaluate("1.0 / 3.0") == pytest.approx(0.3333333333333333)

# --- Error Handling Tests ---

def test_division_by_zero(evaluator: ExpressionEvaluator):
    """Tests raising ValueError for division by zero."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

def test_mismatched_parentheses(evaluator: ExpressionEvaluator):
    """Tests raising ValueError for mismatched parentheses."""
    with pytest.raises(ValueError, match="Unexpected end of expression"):
        evaluator.evaluate("(5 + 2")
    with pytest.raises(ValueError, match="Syntax Error"):
        evaluator.evaluate("5 + 2)")

def test_invalid_tokens_and_syntax(evaluator: ExpressionEvaluator):
    """Tests raising ValueError for invalid syntax or tokens."""
    # Invalid token (e.g., '&&')
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("5 && 2")
    # Missing operand
    with pytest.raises(ValueError, match="Unexpected end of expression"):
        evaluator.evaluate("5 +")
    # Operator at end
    with pytest.raises(ValueError, match="Invalid expression"):
        evaluator.evaluate("5 *")

def test_empty_expression(evaluator: ExpressionEvaluator):
    """Tests raising ValueError for an empty input string."""
    with pytest.raises(ValueError, match="Empty expression provided"):
        evaluator.evaluate("")