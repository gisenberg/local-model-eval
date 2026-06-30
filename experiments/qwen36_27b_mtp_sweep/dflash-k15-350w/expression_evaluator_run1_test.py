from typing import List, Tuple

Token = Tuple[str, str]  # (type, value)


class ExpressionEvaluator:
    """Evaluates mathematical expressions using a recursive descent parser.
    
    Supports +, -, *, / with standard precedence, parentheses, unary minus,
    and floating-point numbers. Raises ValueError for invalid inputs.
    """

    class _Tokenizer:
        """Converts a raw expression string into a list of tokens."""

        def __init__(self, text: str) -> None:
            self.text = text
            self.pos = 0
            self.tokens: List[Token] = []
            self._tokenize()

        def _tokenize(self) -> None:
            while self.pos < len(self.text):
                ch = self.text[self.pos]
                if ch.isspace():
                    self.pos += 1
                    continue
                if ch.isdigit() or ch == '.':
                    self._read_number()
                elif ch == '+':
                    self.tokens.append(('PLUS', '+'))
                    self.pos += 1
                elif ch == '-':
                    self.tokens.append(('MINUS', '-'))
                    self.pos += 1
                elif ch == '*':
                    self.tokens.append(('MULT', '*'))
                    self.pos += 1
                elif ch == '/':
                    self.tokens.append(('DIV', '/'))
                    self.pos += 1
                elif ch == '(':
                    self.tokens.append(('LPAREN', '('))
                    self.pos += 1
                elif ch == ')':
                    self.tokens.append(('RPAREN', ')'))
                    self.pos += 1
                else:
                    raise ValueError(f"Invalid token: '{ch}'")
            self.tokens.append(('EOF', ''))

        def _read_number(self) -> None:
            start = self.pos
            has_dot = False
            while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
                if self.text[self.pos] == '.':
                    if has_dot:
                        raise ValueError("Invalid number format: multiple decimal points")
                    has_dot = True
                self.pos += 1
            num_str = self.text[start:self.pos]
            if num_str == '.':
                raise ValueError("Invalid number format: lone decimal point")
            self.tokens.append(('NUMBER', num_str))

    class _Parser:
        """Recursive descent parser implementing the grammar:
        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := ('-')* primary
        primary    := NUMBER | '(' expression ')'
        """

        def __init__(self, tokens: List[Token]) -> None:
            self.tokens = tokens
            self.pos = 0

        def _peek(self) -> Token:
            return self.tokens[self.pos]

        def _consume(self, expected_type: str = None) -> Token:
            token = self._peek()
            if expected_type and token[0] != expected_type:
                raise ValueError(f"Expected {expected_type}, got {token[0]}")
            self.pos += 1
            return token

        def parse_expression(self) -> float:
            result = self._parse_term()
            while self._peek()[0] in ('PLUS', 'MINUS'):
                op = self._consume()
                right = self._parse_term()
                if op[0] == 'PLUS':
                    result += right
                else:
                    result -= right
            return result

        def _parse_term(self) -> float:
            result = self._parse_factor()
            while self._peek()[0] in ('MULT', 'DIV'):
                op = self._consume()
                right = self._parse_factor()
                if op[0] == 'MULT':
                    result *= right
                else:
                    if right == 0.0:
                        raise ValueError("Division by zero")
                    result /= right
            return result

        def _parse_factor(self) -> float:
            if self._peek()[0] == 'MINUS':
                self._consume()
                return -self._parse_factor()
            return self._parse_primary()

        def _parse_primary(self) -> float:
            token = self._peek()
            if token[0] == 'NUMBER':
                self._consume()
                return float(token[1])
            elif token[0] == 'LPAREN':
                self._consume()
                result = self.parse_expression()
                if self._peek()[0] != 'RPAREN':
                    raise ValueError("Mismatched parentheses: missing closing parenthesis")
                self._consume()
                return result
            else:
                raise ValueError(f"Unexpected token: {token[1] if token[1] else 'EOF'}")

    def evaluate(self, expr: str) -> float:
        """Evaluate a mathematical expression string and return the result as a float.

        Args:
            expr: A string containing a mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or involves division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokenizer = self._Tokenizer(expr)
        parser = self._Parser(tokenizer.tokens)
        result = parser.parse_expression()

        if parser._peek()[0] != 'EOF':
            raise ValueError(f"Unexpected token after expression: {parser._peek()[1]}")

        return result

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Test that * and / bind tighter than + and -"""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_and_unary_minus(evaluator):
    """Test grouping and unary minus operator"""
    assert evaluator.evaluate("-(2 + 3) * 2") == -10.0
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("--4") == 4.0
    assert evaluator.evaluate("-( -(2 + 1) )") == 3.0

def test_floating_point_numbers(evaluator):
    """Test support for decimal numbers"""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + 1.5") == pytest.approx(2.0)
    assert evaluator.evaluate("10 / 4.0") == pytest.approx(2.5)

def test_division_by_zero(evaluator):
    """Test that division by zero raises ValueError"""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("(2 + 3) / 0.0")

def test_error_handling(evaluator):
    """Test mismatched parentheses, invalid tokens, and empty expressions"""
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 & 3")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")