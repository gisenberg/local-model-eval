import re
, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser.

    Supports +, -, *, /, parentheses, floating-point numbers, and unary minus.
    """

    def __init__(self):
        # Tokenization is handled internally during parsing for simplicity in this structure
        pass

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of tokens (numbers, operators, parentheses).
        Handles implicit unary minus by checking context.
        """
        # Regex to match numbers (including floats), operators, and parentheses
        token_specification = [
            ('NUMBER', r'\d+(\.\d*)?'),  # Integers or decimals
            ('OP', r'[+\-*/]'),           # Operators
            ('LPAREN', r'\('),            # Left parenthesis
            ('RPAREN', r'\)'),            # Right parenthesis
            ('SKIP', r'[ \t]+'),          # Skip whitespace
            ('MISMATCH', r'.'),           # Any other character
        ]
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
        
        tokens = []
        for mo in re.finditer(tok_regex, expr):
            kind = mo.lastgroup
            value = mo.group()
            
            if kind == 'NUMBER':
                tokens.append(value)
            elif kind == 'OP' or kind == 'LPAREN' or kind == 'RPAREN':
                tokens.append(value)
            elif kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                raise ValueError(f"Invalid token found in expression: '{value}'")
        
        return tokens

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the given mathematical expression string.

        Args:
            expr: The mathematical expression string.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is invalid (e.g., mismatched parentheses, 
                        division by zero, invalid tokens, empty expression).
        """
        if not expr.strip():
            raise ValueError("Expression cannot be empty.")

        tokens = self._tokenize(expr)
        self.pos = 0
        
        try:
            result = self._parse_expression(tokens)
            
            # After parsing the main expression, we must have consumed all tokens
            if self.pos != len(tokens):
                raise ValueError("Unexpected tokens remaining after parsing.")
                
            return result
        
        except IndexError:
            # Catches cases where the parser expects a token but runs out (e.g., "3 + ")
            raise ValueError("Incomplete or malformed expression.")


    # --- Recursive Descent Parser Components ---

    def _peek(self, tokens: List[str]) -> Union[str, None]:
        """Returns the next token without consuming it."""
        if self.pos < len(tokens):
            return tokens[self.pos]
        return None

    def _consume(self, tokens: List[str], expected: str = None) -> str:
        """Consumes the current token and advances the pointer."""
        if self.pos >= len(tokens):
            raise ValueError("Unexpected end of expression.")
        
        token = tokens[self.pos]
        if expected and token != expected:
            raise ValueError(f"Expected '{expected}' but found '{token}'")
        
        self.pos += 1
        return token

    def _parse_primary(self, tokens: List[str]) -> float:
        """
        Handles numbers and parenthesized expressions. This is the base case.
        """
        token = self._peek(tokens)
        
        if token is None:
            raise ValueError("Unexpected end of expression while expecting a primary term.")

        # 1. Number
        if token.replace('.', '', 1).isdigit():
            self._consume(tokens)
            return float(token)

        # 2. Parenthesized Expression
        elif token == '(':
            self._consume(tokens, '(')
            result = self._parse_expression(tokens)
            self._consume(tokens, ')')
            return result
        
        # 3. Unary Minus (Handles '-3' or '-(2+1)')
        elif token == '-':
            self._consume(tokens, '-')
            # Recursively call primary to handle the operand after the minus sign
            operand = self._parse_primary(tokens)
            return -operand
        
        # 4. Unary Plus (Optional, but good practice)
        elif token == '+':
            self._consume(tokens, '+')
            operand = self._parse_primary(tokens)
            return operand
        
        else:
            raise ValueError(f"Invalid token encountered at primary level: {token}")


    def _parse_factor(self, tokens: List[str]) -> float:
        """
        Handles multiplication and division (*, /).
        Factor -> Primary ( ('*' | '/') Primary )*
        """
        result = self._parse_primary(tokens)
        
        while self._peek(tokens) in ('*', '/'):
            op = self._consume(tokens)
            right = self._parse_primary(tokens)
            
            if op == '*':
                result *= right
            elif op == '/':
                if right == 0.0:
                    raise ValueError("Division by zero.")
                result /= right
        
        return result

    def _parse_term(self, tokens: List[str]) -> float:
        """
        Handles addition and subtraction (+, -).
        Term -> Factor ( ('+' | '-') Factor )*
        """
        result = self._parse_factor(tokens)
        
        while self._peek(tokens) in ('+', '-'):
            op = self._consume(tokens)
            right = self._parse_factor(tokens)
            
            if op == '+':
                result += right
            elif op == '-':
                result -= right
                
        return result

    def _parse_expression(self, tokens: List[str]) -> float:
        """
        The entry point for the recursive descent parser (handles lowest precedence: + and -).
        Expression -> Term ( ('+' | '-') Term )*
        """
        # We reuse _parse_term as the main expression parser since + and - are the lowest precedence
        return self._parse_term(tokens)


# --- Pytest Tests ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("5 * 2") == 10.0
    assert evaluator.evaluate("20 / 5") == 4.0

def test_operator_precedence():
    evaluator = ExpressionEvaluator()
    # 2 + 3 * 4 = 2 + 12 = 14
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # 2 * 3 + 4 = 6 + 4 = 10
    assert evaluator.evaluate("2 * 3 + 4") == 10.0
    # 2 + 3 / 4 = 2 + 0.75 = 2.75
    assert evaluator.evaluate("2 + 3 / 4") == 2.75

def test_parentheses_grouping():
    evaluator = ExpressionEvaluator()
    # (2 + 3) * 4 = 5 * 4 = 20
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # 2 * (3 + 4) = 2 * 7 = 14
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    # Complex grouping
    assert evaluator.evaluate("10 / (2 + (4 - 2))") == 5.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    # Simple unary minus
    assert evaluator.evaluate("-5") == -5.0
    # Unary minus with number
    assert evaluator.evaluate("10 + -3") == 7.0
    # Unary minus with parentheses
    assert evaluator.evaluate("10 - (-3)") == 13.0
    # Nested unary minus
    assert evaluator.evaluate("--5") == 5.0
    # Unary minus in complex expression
    assert evaluator.evaluate("5 * (-2 + -1)") == 5.0

def test_floating_point_and_complex():
    evaluator = ExpressionEvaluator()
    # Floating point
    assert evaluator.evaluate("3.14 * 2") == 6.28
    # Mixed types
    assert evaluator.evaluate("10.5 / 2.5") == 4.2
    # Complex expression
    assert evaluator.evaluate("-(2.5 + (1.5 * 2))") == -5.5

# --- Error Handling Tests ---

def test_mismatched_parentheses():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Expected ')'"):
        evaluator.evaluate("(2 + 3")

def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (2 - 2)")

def test_invalid_tokens():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Invalid token found"):
        evaluator.evaluate("5 $ 3")

def test_empty_expression():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Expression cannot be empty"):
        evaluator.evaluate("  ")

def test_incomplete_expression():
    evaluator = ExpressionEvaluator()
    # Missing operand after operator
    with pytest.raises(ValueError, match="Unexpected end of expression"):
        evaluator.evaluate("5 + ")
    # Missing closing parenthesis
    with pytest.raises(ValueError, match="Expected ')'"):
        evaluator.evaluate("(5 + 3")