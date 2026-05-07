from typing import List, Tuple, Optional

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Correct operator precedence (*, / before +, -)
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers (e.g., '3.14')
    """
    
    def __init__(self) -> None:
        self.tokens: List[Tuple[str, Optional[float]]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

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

        self._tokenize(expr)
        self.pos = 0
        result = self._parse_expr()

        if self.pos < len(self.tokens) and self.tokens[self.pos][0] != 'EOF':
            raise ValueError("Invalid expression: unexpected tokens")

        return result

    def _tokenize(self, expr: str) -> None:
        """Convert expression string into a list of tokens."""
        self.tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            c = expr[i]
            if c.isspace():
                i += 1
                continue
                
            if c.isdigit() or c == '.':
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format")
                        has_dot = True
                    j += 1
                    
                num_str = expr[i:j]
                if num_str == '.':
                    raise ValueError("Invalid number format")
                    
                self.tokens.append(('NUM', float(num_str)))
                i = j
                
            elif c in '+-*/()':
                self.tokens.append((c, None))
                i += 1
                
            else:
                raise ValueError(f"Invalid token: {c}")
                
        self.tokens.append(('EOF', None))

    def _parse_expr(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self.pos < len(self.tokens) and self.tokens[self.pos][0] in ('+', '-'):
            op = self.tokens[self.pos][0]
            self.pos += 1
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos][0] in ('*', '/'):
            op = self.tokens[self.pos][0]
            self.pos += 1
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary operators, parentheses, and numbers (highest precedence)."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")

        token_type, token_val = self.tokens[self.pos]

        if token_type == 'NUM':
            self.pos += 1
            return token_val
        elif token_type == '(':
            self.pos += 1
            result = self._parse_expr()
            if self.pos >= len(self.tokens) or self.tokens[self.pos][0] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result
        elif token_type == '-':
            self.pos += 1
            return -self._parse_factor()
        elif token_type == '+':
            self.pos += 1
            return self._parse_factor()
        else:
            raise ValueError(f"Invalid token in expression: {token_type}")

import pytest

def test_basic_precedence():
    """Test correct operator precedence for +, -, *, /"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 / 2 - 3") == 2.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping():
    """Test parentheses override default precedence"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert ev.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

def test_unary_minus():
    """Test unary minus in various contexts"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("2 * -3") == -6.0
    assert ev.evaluate("--5") == 5.0

def test_floating_point_numbers():
    """Test support for decimal/floating point values"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 + 2.86") == pytest.approx(6.0)
    assert ev.evaluate("10.5 / 2.1") == pytest.approx(5.0)
    assert ev.evaluate("-0.5 * 4") == pytest.approx(-2.0)

def test_error_handling():
    """Test ValueError raising for invalid inputs"""
    ev = ExpressionEvaluator()
    
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
        
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 & 3")
        
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
        
    with pytest.raises(ValueError, match="Invalid expression"):
        ev.evaluate("2 +")