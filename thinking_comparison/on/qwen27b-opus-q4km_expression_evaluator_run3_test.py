import re
from typing import List, Tuple, Any

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result as a float.
        
        Args:
            expr: A string containing the mathematical expression.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is invalid, empty, or contains errors.
        """
        if not expr or not expr.strip():
            raise ValueError("Expression cannot be empty")

        self._tokenize(expr)
        
        if not self.tokens:
            raise ValueError("Expression contains no valid tokens")

        self.pos = 0
        result = self._parse_expression()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token '{self.tokens[self.pos]}' at end of expression")
            
        return result

    def _tokenize(self, expr: str) -> None:
        """
        Converts the input string into a list of tokens.
        """
        # Regex to match numbers (integers and floats) and operators/parentheses
        token_pattern = r'(\d+\.?\d*|\+|\-|\*|\/|\(|\))'
        tokens = re.findall(token_pattern, expr.replace(" ", ""))
        
        if not tokens:
            raise ValueError("No valid tokens found in expression")
            
        self.tokens = tokens

    def _current_token(self) -> str:
        """Returns the current token or an empty string if end of input."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ""

    def _consume(self) -> str:
        """Consumes the current token and advances the position."""
        token = self._current_token()
        if token:
            self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses an expression: Term (('+' | '-') Term)*
        Handles addition and subtraction.
        """
        value = self._parse_term()
        
        while self._current_token() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                value += right
            else:
                value -= right
                
        return value

    def _parse_term(self) -> float:
        """
        Parses a term: Factor (('*' | '/') Factor)*
        Handles multiplication and division.
        """
        value = self._parse_factor()
        
        while self._current_token() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                value *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero is not allowed")
                value /= right
                
        return value

    def _parse_factor(self) -> float:
        """
        Parses a factor: Number | '(' Expression ')' | '-' Factor
        Handles numbers, parentheses, and unary minus.
        """
        token = self._current_token()
        
        # Handle unary minus
        if token == '-':
            self._consume()
            return -self._parse_factor()
            
        # Handle opening parenthesis
        if token == '(':
            self._consume()
            value = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses: expected closing ')'")
            self._consume()
            return value
            
        # Handle numbers
        if token and (token[0].isdigit() or token[0] == '.'):
            self._consume()
            try:
                return float(token)
            except ValueError:
                raise ValueError(f"Invalid number format: '{token}'")
                
        # If we reach here, it's an invalid token
        raise ValueError(f"Invalid token: '{token}'")

# --- Pytest Tests ---

def test_basic_arithmetic():
    """Test basic addition, subtraction, multiplication, and division."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 2") == 4.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 5") == 15.0
    assert evaluator.evaluate("20 / 4") == 5.0

def test_operator_precedence():
    """Test that multiplication/division happens before addition/subtraction."""
    evaluator = ExpressionEvaluator()
    # 2 + 3 * 4 should be 2 + 12 = 14, not 20
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # 10 / 2 + 3 should be 5 + 3 = 8
    assert evaluator.evaluate("10 / 2 + 3") == 8.0
    # 2 * 3 + 4 * 5 should be 6 + 20 = 26
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses():
    """Test grouping with parentheses."""
    evaluator = ExpressionEvaluator()
    # (2 + 3) * 4 should be 5 * 4 = 20
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # Nested parentheses
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0
    # Complex nesting
    assert evaluator.evaluate("2 * (3 + (4 * 5))") == 46.0

def test_unary_minus():
    """Test unary minus operator."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-3 + 2") == -1.0
    assert evaluator.evaluate("5 * -3") == -15.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0  # Double negative

def test_error_cases():
    """Test various error conditions."""
    evaluator = ExpressionEvaluator()
    
    # Empty expression
    try:
        evaluator.evaluate("")
        assert False, "Should have raised ValueError for empty string"
    except ValueError as e:
        assert "empty" in str(e).lower()
        
    # Division by zero
    try:
        evaluator.evaluate("10 / 0")
        assert False, "Should have raised ValueError for division by zero"
    except ValueError as e:
        assert "division by zero" in str(e).lower()
        
    # Mismatched parentheses
    try:
        evaluator.evaluate("(2 + 3")
        assert False, "Should have raised ValueError for mismatched parentheses"
    except ValueError as e:
        assert "parentheses" in str(e).lower()
        
    # Invalid token
    try:
        evaluator.evaluate("2 + a")
        assert False, "Should have raised ValueError for invalid token"
    except ValueError as e:
        assert "invalid token" in str(e).lower()