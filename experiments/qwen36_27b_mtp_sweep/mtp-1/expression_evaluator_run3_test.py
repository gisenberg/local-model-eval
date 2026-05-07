from typing import List, Tuple, Optional

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Correct operator precedence (*, / before +, -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14, .5, 2.)
    """

    def __init__(self) -> None:
        self.tokens: List[Tuple[str, Optional[str]]] = []
        self.pos: int = 0
        self.current_token: Tuple[str, Optional[str]] = ('EOF', None)

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.
        
        Args:
            expr: A string containing a mathematical expression.
            
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
        self.current_token = self.tokens[0]

        result = self.parse_expression()

        if self.current_token[0] != 'EOF':
            if self.current_token[0] == 'RPAREN':
                raise ValueError("Mismatched parentheses")
            raise ValueError(f"Unexpected token: {self.current_token[1]}")

        return result

    def _tokenize(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """Convert an expression string into a list of (type, value) tokens."""
        tokens: List[Tuple[str, Optional[str]]] = []
        i = 0
        n = len(text)
        
        while i < n:
            if text[i].isspace():
                i += 1
                continue
                
            if text[i].isdigit() or text[i] == '.':
                j = i
                while j < n and (text[j].isdigit() or text[j] == '.'):
                    j += 1
                num_str = text[i:j]
                if num_str.count('.') > 1:
                    raise ValueError(f"Invalid number: {num_str}")
                if not any(c.isdigit() for c in num_str):
                    raise ValueError(f"Invalid number: {num_str}")
                tokens.append(('NUMBER', num_str))
                i = j
            elif text[i] == '+':
                tokens.append(('PLUS', '+'))
                i += 1
            elif text[i] == '-':
                tokens.append(('MINUS', '-'))
                i += 1
            elif text[i] == '*':
                tokens.append(('MULT', '*'))
                i += 1
            elif text[i] == '/':
                tokens.append(('DIV', '/'))
                i += 1
            elif text[i] == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif text[i] == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{text[i]}'")
                
        tokens.append(('EOF', None))
        return tokens

    def _advance(self) -> None:
        """Move the parser to the next token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = ('EOF', None)

    def parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self.parse_term()
        while self.current_token[0] in ('PLUS', 'MINUS'):
            op = self.current_token[0]
            self._advance()
            right = self.parse_term()
            if op == 'PLUS':
                result += right
            else:
                result -= right
        return result

    def parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self.parse_factor()
        while self.current_token[0] in ('MULT', 'DIV'):
            op = self.current_token[0]
            self._advance()
            right = self.parse_factor()
            if op == 'MULT':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def parse_factor(self) -> float:
        """Parse unary plus and minus operators."""
        if self.current_token[0] == 'PLUS':
            self._advance()
            return self.parse_factor()
        if self.current_token[0] == 'MINUS':
            self._advance()
            return -self.parse_factor()
        return self.parse_primary()

    def parse_primary(self) -> float:
        """Parse numbers and parenthesized expressions (highest precedence)."""
        token_type, token_value = self.current_token
        
        if token_type == 'NUMBER':
            self._advance()
            return float(token_value)
            
        if token_type == 'LPAREN':
            self._advance()
            result = self.parse_expression()
            if self.current_token[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
            
        if token_type == 'RPAREN':
            raise ValueError("Mismatched parentheses")
            
        raise ValueError(f"Invalid token: {token_value}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence_and_parentheses(evaluator):
    """Test correct precedence of * / over + - and grouping with parentheses."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("((2 + 3) * (4 - 1)) / 5") == pytest.approx(3.0)

def test_unary_minus(evaluator):
    """Test unary minus in various positions."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("2 * -3.5") == -7.0

def test_floating_point_numbers(evaluator):
    """Test support for decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("1.5 + 2.5") == 4.0
    assert evaluator.evaluate("-.5") == -0.5
    assert evaluator.evaluate("2. / 4.") == 0.5

def test_error_handling(evaluator):
    """Test ValueError raising for invalid inputs."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")

def test_whitespace_and_complex_expressions(evaluator):
    """Test robustness with whitespace and nested operations."""
    assert evaluator.evaluate("  2   +   3  ") == 5.0
    assert evaluator.evaluate("-(  2.5 * (  3 - 1  )  )") == pytest.approx(-5.0)
    assert evaluator.evaluate("1 + 2 * 3 - 4 / 2") == pytest.approx(5.0)