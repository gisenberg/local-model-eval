from typing import List, Tuple
import pytest

Token = Tuple[str, str]

class ExpressionEvaluator:
    """A mathematical expression evaluator using a recursive descent parser.
    
    Supports +, -, *, / with correct operator precedence, parentheses for grouping,
    unary minus, and floating point numbers.
    """

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The evaluated result as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        parser = self._Parser(tokens)
        result = parser.parse_expression()

        if parser.pos < len(tokens):
            raise ValueError("Invalid token: unexpected characters after expression")

        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Convert expression string into a list of tokens."""
        tokens: List[Token] = []
        i = 0
        n = len(expr)
        
        while i < n:
            if expr[i].isspace():
                i += 1
                continue

            if expr[i].isdigit() or expr[i] == '.':
                j = i
                has_dot = False
                has_digit = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid token: multiple decimal points in number")
                        has_dot = True
                    else:
                        has_digit = True
                    j += 1
                if not has_digit:
                    raise ValueError("Invalid token: number must contain at least one digit")
                tokens.append(('NUMBER', expr[i:j]))
                i = j
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], expr[i]))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")

        return tokens

    class _Parser:
        """Internal recursive descent parser."""

        def __init__(self, tokens: List[Token]):
            self.tokens = tokens
            self.pos = 0

        def current(self) -> Token:
            if self.pos < len(self.tokens):
                return self.tokens[self.pos]
            return ('EOF', '')

        def consume(self, expected_type: str = None) -> Token:
            token = self.current()
            if expected_type and token[0] != expected_type:
                raise ValueError(f"Expected {expected_type}, got {token[0]}")
            self.pos += 1
            return token

        def parse_expression(self) -> float:
            """Parse addition and subtraction (lowest precedence)."""
            result = self.parse_term()
            while self.current()[0] in ('+', '-'):
                op = self.consume()[0]
                right = self.parse_term()
                if op == '+':
                    result += right
                else:
                    result -= right
            return result

        def parse_term(self) -> float:
            """Parse multiplication and division (higher precedence)."""
            result = self.parse_factor()
            while self.current()[0] in ('*', '/'):
                op = self.consume()[0]
                right = self.parse_factor()
                if op == '*':
                    result *= right
                else:
                    if right == 0:
                        raise ValueError("Division by zero")
                    result /= right
            return result

        def parse_factor(self) -> float:
            """Parse unary operators, numbers, and parenthesized expressions."""
            token = self.current()
            if token[0] == '-':
                self.consume()
                return -self.parse_factor()
            elif token[0] == '+':
                self.consume()
                return self.parse_factor()
            elif token[0] == 'NUMBER':
                self.consume()
                return float(token[1])
            elif token[0] == '(':
                self.consume()
                result = self.parse_expression()
                if self.current()[0] != ')':
                    raise ValueError("Mismatched parentheses")
                self.consume()
                return result
            else:
                raise ValueError(f"Invalid token: unexpected {token}")


# ==================== PYTEST TESTS ====================

def test_operator_precedence():
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 / 2 + 3") == 8.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping():
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert ev.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

def test_unary_minus():
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3 + 2") == -1.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("--5") == 5.0
    assert ev.evaluate("3 * -2") == -6.0

def test_floating_point_numbers():
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate(".5 + .5") == 1.0
    assert ev.evaluate("10.0 / 4.0") == 2.5

def test_error_handling():
    ev = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
        
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("5 / (2 - 2)")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("2 + 3)")
        
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 + a")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("3..14")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 +")