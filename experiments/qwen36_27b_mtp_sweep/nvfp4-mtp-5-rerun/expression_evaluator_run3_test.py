from typing import List, Tuple, Any


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Binary operators: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating-point numbers (e.g., '3.14', '.5')
    
    Raises ValueError for invalid input, mismatched parentheses, 
    division by zero, or empty expressions.
    """

    def __init__(self) -> None:
        self.tokens: List[Tuple[str, Any]] = []
        self.pos: int = 0

    def _tokenize(self, expr: str) -> List[Tuple[str, Any]]:
        """Converts an expression string into a list of (type, value) tokens."""
        tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            if expr[i].isspace():
                i += 1
                continue
                
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                num_str = expr[i:j]
                if not any(c.isdigit() for c in num_str):
                    raise ValueError(f"Invalid number format: '{num_str}'")
                tokens.append(('NUM', float(num_str)))
                i = j
            elif expr[i] == '+':
                tokens.append(('PLUS', '+'))
                i += 1
            elif expr[i] == '-':
                tokens.append(('MINUS', '-'))
                i += 1
            elif expr[i] == '*':
                tokens.append(('MUL', '*'))
                i += 1
            elif expr[i] == '/':
                tokens.append(('DIV', '/'))
                i += 1
            elif expr[i] == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif expr[i] == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")
                
        tokens.append(('EOF', None))
        return tokens

    def _current_token(self) -> Tuple[str, Any]:
        return self.tokens[self.pos]

    def _eat(self, token_type: str) -> None:
        if self._current_token()[0] == token_type:
            self.pos += 1
        else:
            raise ValueError(f"Expected {token_type}, got {self._current_token()[0]}")

    def _parse_expression(self) -> float:
        """Handles addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        while self._current_token()[0] in ('PLUS', 'MINUS'):
            op = self._current_token()[0]
            self._eat(op)
            right = self._parse_term()
            left = left + right if op == 'PLUS' else left - right
        return left

    def _parse_term(self) -> float:
        """Handles multiplication and division (higher precedence)."""
        left = self._parse_factor()
        while self._current_token()[0] in ('MUL', 'DIV'):
            op = self._current_token()[0]
            self._eat(op)
            right = self._parse_factor()
            if op == 'MUL':
                left *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                left /= right
        return left

    def _parse_factor(self) -> float:
        """Handles unary operators and delegates to primary expressions."""
        if self._current_token()[0] == 'PLUS':
            self._eat('PLUS')
            return self._parse_factor()
        if self._current_token()[0] == 'MINUS':
            self._eat('MINUS')
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Handles numbers and parenthesized expressions."""
        token = self._current_token()
        if token[0] == 'NUM':
            self._eat('NUM')
            return token[1]
        if token[0] == 'LPAREN':
            self._eat('LPAREN')
            result = self._parse_expression()
            if self._current_token()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self._eat('RPAREN')
            return result
        raise ValueError(f"Unexpected token: {token[0]}")

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: A string containing a valid mathematical expression.
            
        Returns:
            The evaluated result as a float.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        self.tokens = self._tokenize(expr)
        self.pos = 0
        result = self._parse_expression()
        
        if self._current_token()[0] != 'EOF':
            raise ValueError("Invalid expression or mismatched parentheses")
            
        return result

import pytest

def test_operator_precedence():
    """Tests correct precedence of +, -, *, /"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 / 2 - 1") == 4.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus():
    """Tests grouping and unary minus handling"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-(2 + 3)") == -5.0
    assert ev.evaluate("(-3) * 2") == -6.0
    assert ev.evaluate("--5") == 5.0
    assert ev.evaluate("-(-2)") == 2.0

def test_floating_point_numbers():
    """Tests support for decimal numbers"""
    ev = ExpressionEvaluator()
    assert ev.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert ev.evaluate(".5 + .5") == 1.0
    assert ev.evaluate("10.0 / 4.0") == pytest.approx(2.5)

def test_error_handling():
    """Tests ValueError raising for invalid inputs"""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("1 + a")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Invalid expression"):
        ev.evaluate("1 + 2 +")

def test_complex_nested_expression():
    """Tests a deeply nested expression combining all features"""
    ev = ExpressionEvaluator()
    expr = "((10 - 2) / 4) * 3 + 1.5"
    assert ev.evaluate(expr) == pytest.approx(7.5)
    assert ev.evaluate("-((2 + 3) * 4) / 2") == pytest.approx(-10.0)