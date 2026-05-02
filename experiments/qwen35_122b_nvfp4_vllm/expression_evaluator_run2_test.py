import re
from typing import List, Any, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator supporting addition, subtraction,
    multiplication, division, parentheses, unary negation, and floats.
    Implemented using a recursive descent parser without eval().
    """

    # Token Constants
    TOK_NUM = "NUMBER"
    TOK_PLUS = "PLUS"
    TOK_MINUS = "MINUS"
    TOK_MUL = "MUL"
    TOK_DIV = "DIV"
    TOK_LPAREN = "LPAREN"
    TOK_RPAREN = "RPAREN"
    TOK_EOF = "EOF"

    def __init__(self):
        self._tokens: List[Tuple[str, Any]] = []
        self._pos: int = 0

    def _tokenize(self, expr: str) -> List[Tuple[str, Any]]:
        """Converts the expression string into a list of tokens."""
        if not expr.strip():
            raise ValueError("Empty expression")

        tokens = []
        i = 0
        length = len(expr)

        # Regex to match numbers (including floats), operators, and parentheses
        # Order matters: specific patterns first.
        pattern = re.compile(
            r'\s*(\d+\.\d+|\d+|[+\-*/()])',
            re.IGNORECASE
        )

        current_pos = 0
        for match in pattern.finditer(expr):
            text = match.group(1)
            start = match.start()
            
            # Check for gaps (invalid characters)
            if start > current_pos:
                raise ValueError(f"Invalid token near position {current_pos}: '{expr[current_pos:start]}'")
            
            if re.match(r'\d', text):
                tokens.append((self.TOK_NUM, float(text)))
            elif text == '+':
                tokens.append((self.TOK_PLUS, '+'))
            elif text == '-':
                tokens.append((self.TOK_MINUS, '-'))
            elif text == '*':
                tokens.append((self.TOK_MUL, '*'))
            elif text == '/':
                tokens.append((self.TOK_DIV, '/'))
            elif text == '(':
                tokens.append((self.TOK_LPAREN, '('))
            elif text == ')':
                tokens.append((self.TOK_RPAREN, ')'))
            
            current_pos = match.end()
        
        # Check for trailing invalid characters (not caught by gaps above if at end)
        if current_pos < length:
            raise ValueError(f"Invalid token at end of expression: '{expr[current_pos:]}'")
            
        return tokens

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates the mathematical expression.
        
        Args:
            expr (str): The mathematical expression string to evaluate.
            
        Returns:
            float: The result of the evaluation.
            
        Raises:
            ValueError: If the expression is invalid, empty, has mismatched parentheses,
                        or attempts division by zero.
        """
        self._tokens = self._tokenize(expr)
        self._pos = 0
        
        if not self._tokens:
            raise ValueError("Empty expression")

        try:
            result = self._parse_expr()
        except ZeroDivisionError:
            raise ValueError("Division by zero")
            
        # Ensure we consumed all tokens
        if self._pos < len(self._tokens):
            raise ValueError("Unexpected tokens after expression")
            
        return result

    def _current_token(self) -> Tuple[str, Any]:
        """Returns the current token or EOF."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return (self.TOK_EOF, None)

    def _consume(self, expected_type: str) -> Any:
        """Consumes the current token if it matches the expected type."""
        tok_type, value = self._current_token()
        if tok_type != expected_type:
            raise ValueError(f"Expected {expected_type}, got {tok_type}")
        self._pos += 1
        return value

    def _parse_expr(self) -> float:
        """
        Parses an expression (handles + and -).
        Grammar: Expr -> Term { ('+' | '-') Term }
        """
        left = self._parse_term()
        
        while True:
            token_type, _ = self._current_token()
            if token_type == self.TOK_PLUS:
                self._consume(self.TOK_PLUS)
                right = self._parse_term()
                left += right
            elif token_type == self.TOK_MINUS:
                self._consume(self.TOK_MINUS)
                right = self._parse_term()
                left -= right
            else:
                break
        return left

    def _parse_term(self) -> float:
        """
        Parses a term (handles * and /).
        Grammar: Term -> Factor { ('*' | '/') Factor }
        """
        left = self._parse_factor()
        
        while True:
            token_type, _ = self._current_token()
            if token_type == self.TOK_MUL:
                self._consume(self.TOK_MUL)
                right = self._parse_factor()
                left *= right
            elif token_type == self.TOK_DIV:
                self._consume(self.TOK_DIV)
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
            else:
                break
        return left

    def _parse_factor(self) -> float:
        """
        Parses a factor (handles unary +/-, parentheses, numbers).
        Grammar: Factor -> Unary | Primary
                 Unary -> ('+' | '-') Factor | Primary
                 Primary -> Number | '(' Expr ')'
        """
        token_type, _ = self._current_token()
        
        if token_type == self.TOK_MINUS:
            self._consume(self.TOK_MINUS)
            # Unary minus has high precedence, bind tightly
            return -self._parse_factor()
        elif token_type == self.TOK_PLUS:
            self._consume(self.TOK_PLUS)
            return self._parse_factor()
        elif token_type == self.TOK_LPAREN:
            self._consume(self.TOK_LPAREN)
            val = self._parse_expr()
            self._consume(self.TOK_RPAREN)
            return val
        elif token_type == self.TOK_NUM:
            self._consume(self.TOK_NUM)
            return _
        else:
            raise ValueError(f"Unexpected token: {token_type}")


# ----------------------------------------------------------------------
# Pytest Tests
# ----------------------------------------------------------------------

def test_basic_arithmetic_and_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication has higher precedence than addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Division and multiplication
    assert evaluator.evaluate("10 / 2 + 3") == 8.0
    # Pure addition/subtraction
    assert evaluator.evaluate("5 - 2 + 1") == 4.0

def test_parentheses_and_grouping():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # Nested parentheses
    assert evaluator.evaluate("((1 + 2) * 3) + 4") == 13.0
    # Complex nesting
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus_and_floats():
    evaluator = ExpressionEvaluator()
    # Simple unary minus
    assert evaluator.evaluate("-5") == -5.0
    # Unary minus in expression
    assert evaluator.evaluate("3 + -2") == 1.0
    # Unary minus with parens
    assert evaluator.evaluate("- (2 + 3)") == -5.0
    # Floating point inputs
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_error_mismatched_parentheses():
    evaluator = ExpressionEvaluator()
    # Missing closing paren
    try:
        evaluator.evaluate("(2 + 3")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
        
    # Extra closing paren
    try:
        evaluator.evaluate("2 + 3)")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_error_invalid_operations():
    evaluator = ExpressionEvaluator()
    # Division by zero
    try:
        evaluator.evaluate("1 / 0")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
        
    # Invalid tokens (letters)
    try:
        evaluator.evaluate("1 + x")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
        
    # Empty expression
    try:
        evaluator.evaluate("")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass