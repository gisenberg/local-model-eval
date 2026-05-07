from typing import List, Tuple

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1), --5)
    - Floating point numbers (e.g., 3.14, .5, 5.)
    
    Raises ValueError for invalid syntax, mismatched parentheses, 
    division by zero, or empty expressions.
    """

    def __init__(self) -> None:
        self.tokens: List[Tuple[str, str]] = []
        self.pos: int = 0
        self.current_token: Tuple[str, str] = ('EOF', '')

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result.
        
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

        self.tokens = self._tokenize(expr)
        self.pos = 0
        self.current_token = self.tokens[0]

        result = self._parse_expression()

        if self.current_token[0] != 'EOF':
            raise ValueError(f"Unexpected token: {self.current_token[1]}")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, str]]:
        """Convert expression string into a list of (type, value) tokens."""
        tokens: List[Tuple[str, str]] = []
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
                tokens.append(('NUMBER', expr[i:j]))
                i = j
            elif c == '+':
                tokens.append(('PLUS', '+'))
                i += 1
            elif c == '-':
                tokens.append(('MINUS', '-'))
                i += 1
            elif c == '*':
                tokens.append(('MULTIPLY', '*'))
                i += 1
            elif c == '/':
                tokens.append(('DIVIDE', '/'))
                i += 1
            elif c == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif c == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{c}'")
                
        tokens.append(('EOF', ''))
        return tokens

    def _advance(self) -> None:
        """Move to the next token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = ('EOF', '')

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self.current_token[0] in ('PLUS', 'MINUS'):
            op = self.current_token[0]
            self._advance()
            right = self._parse_term()
            result = result + right if op == 'PLUS' else result - right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self.current_token[0] in ('MULTIPLY', 'DIVIDE'):
            op = self.current_token[0]
            self._advance()
            right = self._parse_factor()
            if op == 'MULTIPLY':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary minus operators."""
        if self.current_token[0] == 'MINUS':
            self._advance()
            return -self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions (highest precedence)."""
        if self.current_token[0] == 'NUMBER':
            try:
                val = float(self.current_token[1])
            except ValueError:
                raise ValueError(f"Invalid number: {self.current_token[1]}")
            self._advance()
            return val
        elif self.current_token[0] == 'LPAREN':
            self._advance()
            result = self._parse_expression()
            if self.current_token[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        else:
            raise ValueError(f"Unexpected token: {self.current_token[1]}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / bind tighter than + and -"""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping and unary minus in various contexts"""
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("2 * -(3 + 4)") == -14.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("3 - -2") == 5.0

def test_floating_point_numbers(evaluator):
    """Test decimal number parsing and arithmetic"""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + 1.5") == 2.0
    assert evaluator.evaluate("10.0 / 4.0") == pytest.approx(2.5)

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError"""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("(2 + 3) / 0.0")

def test_invalid_inputs(evaluator):
    """Test error handling for malformed expressions"""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 & 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3)")