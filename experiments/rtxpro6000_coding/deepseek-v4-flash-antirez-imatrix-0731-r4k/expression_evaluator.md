```python
import pytest

class ExpressionEvaluator:
    """
    A recursive descent parser for arithmetic expressions.
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    Raises ValueError for invalid expressions.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result as a float.
        Raises ValueError for empty expressions, invalid tokens, mismatched parentheses,
        or division by zero.
        """
        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Empty expression")
        self._tokens = tokens
        self._pos = 0
        result = self._parse_expression()
        if self._pos != len(tokens):
            raise ValueError("Unexpected token")
        return result

    def _tokenize(self, expr: str) -> list:
        """
        Convert a string into a list of tokens.
        Each token is a tuple (type, value) where type is 'NUM', 'OP', 'LPAREN', or 'RPAREN'.
        Raises ValueError for invalid characters or malformed numbers.
        """
        tokens = []
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch.isspace():
                i += 1
                continue
            # Number: starts with digit or '.' followed by digit
            if ch.isdigit() or (ch == '.' and i + 1 < n and expr[i + 1].isdigit()):
                start = i
                has_dot = False
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format")
                        has_dot = True
                    i += 1
                # Ensure at least one digit
                if start == i or not any(c.isdigit() for c in expr[start:i]):
                    raise ValueError("Invalid number")
                tokens.append(('NUM', float(expr[start:i])))
                continue
            if ch in '+-*/':
                tokens.append(('OP', ch))
                i += 1
                continue
            if ch == '(':
                tokens.append(('LPAREN', '('))
                i += 1
                continue
            if ch == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
                continue
            raise ValueError(f"Invalid token: {ch}")
        return tokens

    def _peek(self) -> tuple:
        """Return the current token without consuming it, or None if at end."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _next(self) -> tuple:
        """Consume and return the current token, raising if at end."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parse an expression: term (('+' | '-') term)*.
        Handles binary addition and subtraction.
        """
        value = self._parse_term()
        while self._peek() == ('OP', '+') or self._peek() == ('OP', '-'):
            op = self._next()[1]
            right = self._parse_term()
            if op == '+':
                value += right
            else:
                value -= right
        return value

    def _parse_term(self) -> float:
        """
        Parse a term: factor (('*' | '/') factor)*.
        Handles multiplication and division, checking for division by zero.
        """
        value = self._parse_factor()
        while self._peek() == ('OP', '*') or self._peek() == ('OP', '/'):
            op = self._next()[1]
            right = self._parse_factor()
            if op == '*':
                value *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                value /= right
        return value

    def _parse_factor(self) -> float:
        """
        Parse a factor: unary minus, number, or parenthesized expression.
        Handles unary minus and parentheses.
        """
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        if token[0] == 'OP' and token[1] == '-':
            self._next()  # consume '-'
            value = self._parse_factor()
            return -value
        if token[0] == 'NUM':
            self._next()
            return token[1]
        if token[0] == 'LPAREN':
            self._next()  # consume '('
            value = self._parse_expression()
            if self._peek() != ('RPAREN', ')'):
                raise ValueError("Mismatched parentheses")
            self._next()  # consume ')'
            return value
        raise ValueError("Invalid token")


# ---------- Pytest Tests ----------

def test_basic_arithmetic():
    """Test simple addition, subtraction, multiplication, and division."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2+3") == 5.0
    assert evaluator.evaluate("10-4") == 6.0
    assert evaluator.evaluate("2*3") == 6.0
    assert evaluator.evaluate("8/2") == 4.0

def test_precedence():
    """Test that multiplication/division bind tighter than addition/subtraction."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2+3*4") == 14.0
    assert evaluator.evaluate("10-2*3") == 4.0
    assert evaluator.evaluate("20/2+5") == 15.0

def test_parentheses():
    """Test grouping with parentheses."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2+3)*4") == 20.0
    assert evaluator.evaluate("2*(3+4)") == 14.0
    assert evaluator.evaluate("(2+3)*(4-1)") == 15.0

def test_unary_minus():
    """Test unary minus in various positions."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2+1)") == -3.0
    assert evaluator.evaluate("3*-2") == -6.0
    assert evaluator.evaluate("3--2") == 5.0  # 3 - (-2)

def test_error_cases():
    """Test that invalid expressions raise ValueError."""
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate("")          # empty
    with pytest.raises(ValueError):
        evaluator.evaluate("   ")       # whitespace only
    with pytest.raises(ValueError):
        evaluator.evaluate("(2+3")      # missing closing parenthesis
    with pytest.raises(ValueError):
        evaluator.evaluate("2+3)")      # extra closing parenthesis
    with pytest.raises(ValueError):
        evaluator.evaluate("1/0")       # division by zero
    with pytest.raises(ValueError):
        evaluator.evaluate("2+abc")     # invalid token
```