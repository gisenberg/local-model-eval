from typing import List, Union

class Token:
    """Represents a lexical token in the expression."""
    def __init__(self, type: str, value: Union[float, None]):
        self.type = type
        self.value = value

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    
    Supports:
    - Operators: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary minus (and plus)
    - Floating point numbers
    """
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.
        
        Args:
            expr: A string containing a mathematical expression.
            
        Returns:
            The numerical result of the expression.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        self.tokens = self._tokenize(expr)
        self.pos = 0
        self.current = self.tokens[0]
        
        result = self._parse_expr()
        
        if self.current.type != 'EOF':
            raise ValueError("Invalid expression: unexpected tokens after valid expression")
            
        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """Convert expression string into a list of tokens."""
        tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            if expr[i].isspace():
                i += 1
                continue
                
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid token: multiple decimal points")
                        has_dot = True
                    j += 1
                tokens.append(Token('NUMBER', float(expr[i:j])))
                i = j
                continue
                
            if expr[i] == '+':
                tokens.append(Token('PLUS', '+'))
            elif expr[i] == '-':
                tokens.append(Token('MINUS', '-'))
            elif expr[i] == '*':
                tokens.append(Token('MULT', '*'))
            elif expr[i] == '/':
                tokens.append(Token('DIV', '/'))
            elif expr[i] == '(':
                tokens.append(Token('LPAREN', '('))
            elif expr[i] == ')':
                tokens.append(Token('RPAREN', ')'))
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")
                
            i += 1
            
        tokens.append(Token('EOF', None))
        return tokens

    def _advance(self) -> None:
        """Move to the next token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]
        else:
            self.current = Token('EOF', None)

    def _parse_expr(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_term()
        while self.current.type in ('PLUS', 'MINUS'):
            op = self.current.type
            self._advance()
            right = self._parse_term()
            if op == 'PLUS':
                left += right
            else:
                left -= right
        return left

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        left = self._parse_factor()
        while self.current.type in ('MULT', 'DIV'):
            op = self.current.type
            self._advance()
            right = self._parse_factor()
            if op == 'DIV':
                if right == 0.0:
                    raise ValueError("Division by zero")
                left /= right
            else:
                left *= right
        return left

    def _parse_factor(self) -> float:
        """Parse unary operators and primary expressions."""
        if self.current.type == 'MINUS':
            self._advance()
            return -self._parse_factor()
        if self.current.type == 'PLUS':
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions."""
        if self.current.type == 'NUMBER':
            val = self.current.value
            self._advance()
            return val
            
        if self.current.type == 'LPAREN':
            self._advance()
            val = self._parse_expr()
            if self.current.type != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return val
            
        raise ValueError(f"Invalid token: '{self.current.value}'")

import pytest

def test_basic_arithmetic_and_precedence():
    """Test basic operators and correct precedence (* / before + -)."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("2 + 3 * 4") == 14.0
    assert ev.evaluate("10 / 2 - 3") == 2.0
    assert ev.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping():
    """Test that parentheses correctly override default precedence."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("(2 + 3) * 4") == 20.0
    assert ev.evaluate("((2 + 3) * (4 - 1))") == 15.0
    assert ev.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus_and_floats():
    """Test unary operators and floating point number support."""
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3.14") == -3.14
    assert ev.evaluate("-(2 + 1)") == -3.0
    assert ev.evaluate("--5") == 5.0
    assert ev.evaluate("3.5 * -2") == -7.0
    assert ev.evaluate("+4.0") == 4.0

def test_division_by_zero():
    """Test that division by zero raises ValueError."""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("10 / (2 - 2)")

def test_invalid_expressions():
    """Test error handling for invalid inputs."""
    ev = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("2 & 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Invalid token"):
        ev.evaluate("3.1.4")