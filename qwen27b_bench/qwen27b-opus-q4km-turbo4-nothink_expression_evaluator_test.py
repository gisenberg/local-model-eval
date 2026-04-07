import re
from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Unary minus
    - Floating point numbers
    """
    
    # Token types
    TOKEN_NUMBER = 'NUMBER'
    TOKEN_PLUS = 'PLUS'
    TOKEN_MINUS = 'MINUS'
    TOKEN_MULTIPLY = 'MULTIPLY'
    TOKEN_DIVIDE = 'DIVIDE'
    TOKEN_LPAREN = 'LPAREN'
    TOKEN_RPAREN = 'RPAREN'
    TOKEN_EOF = 'EOF'
    
    def __init__(self):
        """Initialize the evaluator."""
        self.tokens: List[Tuple[str, Union[str, float]]] = []
        self.pos: int = 0
    
    def _tokenize(self, expr: str) -> List[Tuple[str, Union[str, float]]]:
        """
        Convert the input expression string into a list of tokens.
        
        Args:
            expr: The input expression string
            
        Returns:
            A list of (token_type, token_value) tuples
            
        Raises:
            ValueError: If an invalid character is encountered
        """
        tokens = []
        i = 0
        expr = expr.replace(' ', '')  # Remove whitespace
        
        while i < len(expr):
            char = expr[i]
            
            # Check for digits or decimal point (numbers)
            if char.isdigit() or char == '.':
                num_str = ''
                has_decimal = False
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_decimal:
                            raise ValueError(f"Invalid number format at position {i}: multiple decimal points")
                        has_decimal = True
                    num_str += expr[i]
                    i += 1
                tokens.append((self.TOKEN_NUMBER, float(num_str)))
                continue
            
            # Check for operators and parentheses
            if char == '+':
                tokens.append((self.TOKEN_PLUS, '+'))
            elif char == '-':
                tokens.append((self.TOKEN_MINUS, '-'))
            elif char == '*':
                tokens.append((self.TOKEN_MULTIPLY, '*'))
            elif char == '/':
                tokens.append((self.TOKEN_DIVIDE, '/'))
            elif char == '(':
                tokens.append((self.TOKEN_LPAREN, '('))
            elif char == ')':
                tokens.append((self.TOKEN_RPAREN, ')'))
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")
            i += 1
        
        tokens.append((self.TOKEN_EOF, None))
        return tokens
    
    def _current_token(self) -> Tuple[str, Union[str, float]]:
        """Return the current token without advancing."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (self.TOKEN_EOF, None)
    
    def _consume(self, expected_type: str = None) -> Tuple[str, Union[str, float]]:
        """
        Consume and return the current token, advancing the position.
        
        Args:
            expected_type: Optional expected token type for validation
            
        Returns:
            The consumed token
            
        Raises:
            ValueError: If the token type doesn't match expected_type
        """
        token = self._current_token()
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[0]}")
        self.pos += 1
        return token
    
    def _parse_expression(self) -> float:
        """
        Parse an expression (handles + and - operators).
        
        Expression → Term (('+' | '-') Term)*
        
        Returns:
            The evaluated result of the expression
        """
        result = self._parse_term()
        
        while self._current_token()[0] in (self.TOKEN_PLUS, self.TOKEN_MINUS):
            op = self._consume()[0]
            right = self._parse_term()
            if op == self.TOKEN_PLUS:
                result = result + right
            else:
                result = result - right
        
        return result
    
    def _parse_term(self) -> float:
        """
        Parse a term (handles * and / operators).
        
        Term → Factor (('*' | '/') Factor)*
        
        Returns:
            The evaluated result of the term
        """
        result = self._parse_factor()
        
        while self._current_token()[0] in (self.TOKEN_MULTIPLY, self.TOKEN_DIVIDE):
            op = self._consume()[0]
            right = self._parse_factor()
            if op == self.TOKEN_MULTIPLY:
                result = result * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result = result / right
        
        return result
    
    def _parse_factor(self) -> float:
        """
        Parse a factor (handles numbers, parentheses, and unary minus).
        
        Factor → Number | '(' Expression ')' | '-' Factor
        
        Returns:
            The evaluated result of the factor
        """
        token = self._current_token()
        
        # Handle unary minus
        if token[0] == self.TOKEN_MINUS:
            self._consume()
            return -self._parse_factor()
        
        # Handle numbers
        if token[0] == self.TOKEN_NUMBER:
            self._consume()
            return token[1]
        
        # Handle parentheses
        if token[0] == self.TOKEN_LPAREN:
            self._consume()  # Consume '('
            result = self._parse_expression()
            if self._current_token()[0] != self.TOKEN_RPAREN:
                raise ValueError("Missing closing parenthesis")
            self._consume()  # Consume ')'
            return result
        
        raise ValueError(f"Unexpected token: {token}")
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Args:
            expr: The mathematical expression string to evaluate
            
        Returns:
            The result of the evaluation as a float
            
        Raises:
            ValueError: For invalid expressions, division by zero, 
                       mismatched parentheses, or empty expressions
        """
        # Check for empty expression
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        # Tokenize the expression
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        # Check if expression is empty after tokenization
        if len(self.tokens) == 1 and self.tokens[0][0] == self.TOKEN_EOF:
            raise ValueError("Empty expression")
        
        # Parse and evaluate
        result = self._parse_expression()
        
        # Check for extra tokens (unmatched closing parenthesis)
        if self._current_token()[0] != self.TOKEN_EOF:
            raise ValueError(f"Unexpected token: {self._current_token()}")
        
        return result


# Pytest tests
def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    evaluator = ExpressionEvaluator()
    
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("20 / 4") == 5.0
    assert evaluator.evaluate("3.14 + 2.86") == 6.0
    assert evaluator.evaluate("1.5 * 2") == 3.0


def test_operator_precedence():
    """Test that operator precedence is correctly handled."""
    evaluator = ExpressionEvaluator()
    
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 2 + 12 = 14
    
    # Multiplication before subtraction
    assert evaluator.evaluate("10 - 2 * 3") == 4.0  # 10 - 6 = 4
    
    # Division before addition
    assert evaluator.evaluate("10 / 2 + 3") == 8.0  # 5 + 3 = 8
    
    # Mixed operators
    assert evaluator.evaluate("2 + 3 * 4 - 5 / 5") == 13.0  # 2 + 12 - 1 = 13
    
    # Left-to-right associativity
    assert evaluator.evaluate("10 - 5 - 2") == 3.0  # (10 - 5) - 2 = 3
    assert evaluator.evaluate("20 / 4 / 2") == 2.5  # (20 / 4) / 2 = 2.5


def test_parentheses():
    """Test parentheses for grouping."""
    evaluator = ExpressionEvaluator()
    
    # Basic parentheses
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0  # 5 * 4 = 20
    
    # Nested parentheses
    assert evaluator.evaluate("((2 + 3) * 4)") == 20.0
    
    # Multiple levels of nesting
    assert evaluator.evaluate("((2 + 3) * (4 + 5))") == 45.0  # 5 * 9 = 45
    
    # Complex expression with parentheses
    assert evaluator.evaluate("(1 + 2) * (3 + 4) / (5 - 2)") == 7.0  # 3 * 7 / 3 = 7


def test_unary_minus():
    """Test unary minus operator."""
    evaluator = ExpressionEvaluator()
    
    # Simple unary minus
    assert evaluator.evaluate("-5") == -5.0
    
    # Unary minus with number
    assert evaluator.evaluate("-3 + 2") == -1.0
    
    # Unary minus with parentheses
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    
    # Multiple unary minuses
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-(-3)") == 3.0
    
    # Unary minus in complex expression
    assert evaluator.evaluate("2 * -3 + 4") == -2.0  # 2 * (-3) + 4 = -2
    
    # Unary minus with division
    assert evaluator.evaluate("-10 / 2") == -5.0


def test_error_cases():
    """Test various error conditions."""
    evaluator = ExpressionEvaluator()
    
    # Empty expression
    try:
        evaluator.evaluate("")
        assert False, "Should raise ValueError for empty expression"
    except ValueError as e:
        assert "empty" in str(e).lower()
    
    # Whitespace only
    try:
        evaluator.evaluate("   ")
        assert False, "Should raise ValueError for whitespace-only expression"
    except ValueError as e:
        assert "empty" in str(e).lower()
    
    # Division by zero
    try:
        evaluator.evaluate("10 / 0")
        assert False, "Should raise ValueError for division by zero"
    except ValueError as e:
        assert "division by zero" in str(e).lower()
    
    # Mismatched parentheses - missing closing
    try:
        evaluator.evaluate("(2 + 3")
        assert False, "Should raise ValueError for missing closing parenthesis"
    except ValueError as e:
        assert "parenthesis" in str(e).lower()
    
    # Mismatched parentheses - extra closing
    try:
        evaluator.evaluate("2 + 3)")
        assert False, "Should raise ValueError for extra closing parenthesis"
    except ValueError as e:
        assert "unexpected" in str(e).lower()
    
    # Invalid character
    try:
        evaluator.evaluate("2 + a")
        assert False, "Should raise ValueError for invalid character"
    except ValueError as e:
        assert "invalid" in str(e).lower()
    
    # Invalid number format
    try:
        evaluator.evaluate("3.14.15")
        assert False, "Should raise ValueError for invalid number format"
    except ValueError as e:
        assert "invalid" in str(e).lower() or "decimal" in str(e).lower()


if __name__ == "__main__":
    # Run tests
    test_basic_arithmetic()
    print("✓ test_basic_arithmetic passed")
    
    test_operator_precedence()
    print("✓ test_operator_precedence passed")
    
    test_parentheses()
    print("✓ test_parentheses passed")
    
    test_unary_minus()
    print("✓ test_unary_minus passed")
    
    test_error_cases()
    print("✓ test_error_cases passed")
    
    print("\nAll tests passed!")