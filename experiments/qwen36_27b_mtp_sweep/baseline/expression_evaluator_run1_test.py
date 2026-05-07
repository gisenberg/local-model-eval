from typing import List, Optional

class Token:
    """Represents a lexical token in the expression."""
    def __init__(self, type: str, value: Optional[float] = None):
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value})"


class ExpressionEvaluator:
    """
    A recursive descent parser and evaluator for mathematical expressions.
    
    Supports:
    - Binary operators: +, -, *, / with standard precedence
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers
    """

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

        tokens = self._tokenize(expr)
        return self._parse(tokens)

    def _tokenize(self, expr: str) -> List[Token]:
        """Convert expression string into a list of tokens."""
        tokens: List[Token] = []
        i = 0
        n = len(expr)

        while i < n:
            if expr[i].isspace():
                i += 1
                continue

            # Number (integer or float)
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format")
                        has_dot = True
                    j += 1
                num_str = expr[i:j]
                if num_str == '.' or num_str == '':
                    raise ValueError("Invalid number format")
                tokens.append(Token('NUMBER', float(num_str)))
                i = j
            elif expr[i] == '+':
                tokens.append(Token('PLUS'))
                i += 1
            elif expr[i] == '-':
                tokens.append(Token('MINUS'))
                i += 1
            elif expr[i] == '*':
                tokens.append(Token('MULTIPLY'))
                i += 1
            elif expr[i] == '/':
                tokens.append(Token('DIVIDE'))
                i += 1
            elif expr[i] == '(':
                tokens.append(Token('LPAREN'))
                i += 1
            elif expr[i] == ')':
                tokens.append(Token('RPAREN'))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{expr[i]}'")

        tokens.append(Token('EOF'))
        return tokens

    def _parse(self, tokens: List[Token]) -> float:
        """Initialize parser state and start recursive descent parsing."""
        self.tokens = tokens
        self.pos = 0
        self.current_token = tokens[0]

        result = self._parse_expression()

        if self.current_token.type != 'EOF':
            raise ValueError("Unexpected token after expression")

        return result

    def _advance(self) -> None:
        """Move to the next token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = Token('EOF')

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self.current_token.type in ('PLUS', 'MINUS'):
            op = self.current_token.type
            self._advance()
            right = self._parse_term()
            if op == 'PLUS':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self.current_token.type in ('MULTIPLY', 'DIVIDE'):
            op = self.current_token.type
            self._advance()
            right = self._parse_factor()
            if op == 'MULTIPLY':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary operators, numbers, and parenthesized expressions."""
        token = self.current_token
        if token.type in ('PLUS', 'MINUS'):
            self._advance()
            factor = self._parse_factor()
            return factor if token.type == 'PLUS' else -factor
        elif token.type == 'NUMBER':
            self._advance()
            return token.value
        elif token.type == 'LPAREN':
            self._advance()
            result = self._parse_expression()
            if self.current_token.type != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        else:
            raise ValueError(f"Unexpected token: {token.type}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic_and_precedence(evaluator):
    """Test correct operator precedence for +, -, *, /"""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping with parentheses and unary minus operator"""
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("-(-3)") == 3.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("-(2 * 3) + 10") == 4.0

def test_floating_point_support(evaluator):
    """Test evaluation of expressions containing decimal numbers"""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("1.5 + 2.5") == 4.0
    assert evaluator.evaluate("0.1 + 0.2") == pytest.approx(0.3)

def test_error_handling(evaluator):
    """Test that appropriate ValueErrors are raised for invalid inputs"""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")

def test_complex_nested_expressions(evaluator):
    """Test deeply nested and mixed expressions"""
    assert evaluator.evaluate("((2 + 3) * 4 - 1) / 5") == pytest.approx(3.8)
    assert evaluator.evaluate("-2.5 * (3.0 - 1.5)") == pytest.approx(-3.75)
    assert evaluator.evaluate("10 / (2 + 3) * 4") == pytest.approx(8.0)