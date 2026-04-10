import re
import pytest
from typing import List, Tuple, Union, Optional

# Define token types as constants for clarity
TOKEN_NUMBER = "NUMBER"
TOKEN_PLUS = "PLUS"
TOKEN_MINUS = "MINUS"
TOKEN_MUL = "MUL"
TOKEN_DIV = "DIV"
TOKEN_LPAREN = "LPAREN"
TOKEN_RPAREN = "RPAREN"
TOKEN_EOF = "EOF"

Token = Tuple[str, Union[float, str]]


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    Does not use eval() or ast.literal_eval().
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.tokens: List[Token] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or division by zero occurs.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        if not self.tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        # Ensure all tokens were consumed
        if self.pos < len(self.tokens):
            raise ValueError(f"Invalid syntax: unexpected token '{self._peek()}'")

        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """
        Convert the input string into a list of tokens.
        
        Args:
            expr: The raw expression string.
            
        Returns:
            A list of (type, value) tuples.
            
        Raises:
            ValueError: If an invalid character is encountered.
        """
        tokens: List[Token] = []
        i = 0
        length = len(expr)

        while i < length:
            char = expr[i]

            if char.isspace():
                i += 1
                continue

            if char.isdigit() or char == '.':
                # Parse number
                start = i
                has_dot = False
                while i < length and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format at index {i}")
                        has_dot = True
                    i += 1
                
                num_str = expr[start:i]
                # Validate number structure (e.g., prevent just "." or "..")
                if num_str == '.' or num_str.startswith('.') and len(num_str) == 1:
                     raise ValueError(f"Invalid number format at index {start}")
                
                try:
                    value = float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number '{num_str}' at index {start}")
                
                tokens.append((TOKEN_NUMBER, value))
                continue

            if char == '+':
                tokens.append((TOKEN_PLUS, '+'))
            elif char == '-':
                tokens.append((TOKEN_MINUS, '-'))
            elif char == '*':
                tokens.append((TOKEN_MUL, '*'))
            elif char == '/':
                tokens.append((TOKEN_DIV, '/'))
            elif char == '(':
                tokens.append((TOKEN_LPAREN, '('))
            elif char == ')':
                tokens.append((TOKEN_RPAREN, ')'))
            else:
                raise ValueError(f"Invalid token '{char}' at index {i}")

            i += 1

        tokens.append((TOKEN_EOF, None))
        return tokens

    def _peek(self) -> Token:
        """
        Return the current token without advancing.
        
        Returns:
            The current token tuple.
        """
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (TOKEN_EOF, None)

    def _consume(self) -> Token:
        """
        Return the current token and advance the position.
        
        Returns:
            The current token tuple.
        """
        token = self._peek()
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parse an expression handling addition and subtraction.
        Precedence: Lowest (+, -)
        
        Grammar: Expression -> Term { (+|-) Term }
        
        Returns:
            The calculated float value.
        """
        value = self._parse_term()

        while True:
            token_type, _ = self._peek()
            if token_type == TOKEN_PLUS:
                self._consume()
                value += self._parse_term()
            elif token_type == TOKEN_MINUS:
                self._consume()
                value -= self._parse_term()
            else:
                break
        return value

    def _parse_term(self) -> float:
        """
        Parse a term handling multiplication and division.
        Precedence: Medium (*, /)
        
        Grammar: Term -> Factor { (*|/) Factor }
        
        Returns:
            The calculated float value.
            
        Raises:
            ValueError: If division by zero occurs.
        """
        value = self._parse_factor()

        while True:
            token_type, _ = self._peek()
            if token_type == TOKEN_MUL:
                self._consume()
                value *= self._parse_factor()
            elif token_type == TOKEN_DIV:
                self._consume()
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                value /= divisor
            else:
                break
        return value

    def _parse_factor(self) -> float:
        """
        Parse a factor handling numbers, parentheses, and unary operators.
        Precedence: Highest (unary -, parentheses)
        
        Grammar: Factor -> Number | ( Expression ) | (+|-) Factor
        
        Returns:
            The calculated float value.
            
        Raises:
            ValueError: If parentheses are mismatched or token is invalid.
        """
        token_type, token_value = self._peek()

        # Handle Unary Plus/Minus
        if token_type in (TOKEN_PLUS, TOKEN_MINUS):
            self._consume()
            value = self._parse_factor()
            return -value if token_type == TOKEN_MINUS else value

        # Handle Parentheses
        if token_type == TOKEN_LPAREN:
            self._consume()
            value = self._parse_expression()
            if self._peek()[0] != TOKEN_RPAREN:
                raise ValueError("Mismatched parentheses: expected ')'")
            self._consume()
            return value

        # Handle Numbers
        if token_type == TOKEN_NUMBER:
            self._consume()
            return float(token_value)

        # Handle End of Input or Unexpected Token
        if token_type == TOKEN_EOF:
            raise ValueError("Unexpected end of expression")
        
        raise ValueError(f"Invalid token '{token_value}'")


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------

def test_basic_arithmetic():
    """Test basic addition, subtraction, multiplication, and division."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 2") == 5.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # Precedence check

def test_precedence_and_grouping():
    """Test operator precedence and parentheses grouping."""
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    # Parentheses override precedence
    assert evaluator.evaluate("(1 + 2) * 3") == 9.0
    # Nested parentheses
    assert evaluator.evaluate("((2 + 3) * 4) / 5") == 4.0
    # Complex precedence
    assert evaluator.evaluate("10 - 2 * 3 + 4") == 8.0

def test_unary_minus():
    """Test unary minus operator support."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-3.14") == -3.14
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 * -4") == -12.0
    assert evaluator.evaluate("3 - -4") == 7.0
    assert evaluator.evaluate("--5") == 5.0

def test_floating_point():
    """Test floating point number parsing."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("10. / 2.") == 5.0
    assert evaluator.evaluate("1.5 + 2.5") == 4.0

def test_error_cases():
    """Test error handling for invalid inputs."""
    evaluator = ExpressionEvaluator()
    
    # Empty expression
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("   ")

    # Mismatched parentheses
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + 3)")

    # Division by zero
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / (2 - 2)")

    # Invalid tokens
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3")
    
    # Syntax error (trailing garbage)
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + 2 3")
