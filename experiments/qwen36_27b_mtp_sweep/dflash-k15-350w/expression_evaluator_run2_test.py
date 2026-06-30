from typing import List, Optional

class Token:
    """Represents a lexical token in the expression."""
    def __init__(self, token_type: str, value: Optional[float] = None) -> None:
        self.type = token_type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14)
    """
    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The numerical result of the expression.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        result = self._parse_expression()

        if self._current_token().type != 'EOF':
            raise ValueError(f"Unexpected token: {self._current_token().type}")

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
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError("Invalid number format")
                        has_dot = True
                    j += 1
                num_str = expr[i:j]
                try:
                    tokens.append(Token('NUMBER', float(num_str)))
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")
                i = j
            elif expr[i] == '+':
                tokens.append(Token('PLUS'))
                i += 1
            elif expr[i] == '-':
                tokens.append(Token('MINUS'))
                i += 1
            elif expr[i] == '*':
                tokens.append(Token('MULT'))
                i += 1
            elif expr[i] == '/':
                tokens.append(Token('DIV'))
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

    def _current_token(self) -> Token:
        """Return the token at the current parsing position."""
        return self._tokens[self._pos]

    def _eat(self, token_type: str) -> None:
        """Consume the current token if it matches the expected type."""
        if self._current_token().type == token_type:
            self._pos += 1
        else:
            raise ValueError(f"Expected {token_type}, got {self._current_token().type}")

    def _parse_expression(self) -> float:
        """Parse addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token().type in ('PLUS', 'MINUS'):
            op = self._current_token().type
            self._pos += 1
            right = self._parse_term()
            if op == 'PLUS':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parse multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token().type in ('MULT', 'DIV'):
            op = self._current_token().type
            self._pos += 1
            right = self._parse_factor()
            if op == 'MULT':
                result *= right
            else:
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parse unary operators, numbers, and parentheses (highest precedence)."""
        token = self._current_token()

        if token.type == 'MINUS':
            self._pos += 1
            return -self._parse_factor()
        if token.type == 'PLUS':
            self._pos += 1
            return self._parse_factor()
        if token.type == 'NUMBER':
            self._pos += 1
            return token.value
        if token.type == 'LPAREN':
            self._pos += 1
            result = self._parse_expression()
            self._eat('RPAREN')
            return result

        raise ValueError(f"Unexpected token: {token.type}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / are evaluated before + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping with parentheses and unary minus operator."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-( -3.5 )") == 3.5

def test_floating_point_numbers(evaluator):
    """Test evaluation of expressions containing decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate("1.5 + 2.5") == 4.0
    assert evaluator.evaluate("10.0 / 4.0") == pytest.approx(2.5)

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (2 - 2)")

def test_error_handling(evaluator):
    """Test ValueError for empty, invalid, and mismatched parentheses."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError, match="Expected RPAREN"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3)")