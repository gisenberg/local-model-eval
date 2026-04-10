from typing import List, Tuple


class ExpressionEvaluator:
    """A mathematical expression evaluator using recursive descent parsing."""

    def __init__(self):
        """Initialize the expression evaluator."""
        pass

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.

        Args:
            expr: A string containing a mathematical expression

        Returns:
            The numerical result of evaluating the expression

        Raises:
            ValueError: For invalid expressions, division by zero, etc.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Empty expression")

        parser = self._Parser(tokens)
        result = parser.parse()

        if parser.current_token is not None:
            raise ValueError(f"Unexpected token: {parser.current_token[1]}")

        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, str]]:
        """
        Convert expression string into list of (token_type, value) tuples.

        Args:
            expr: The expression string to tokenize

        Returns:
            List of token tuples
        """
        tokens = []
        i = 0
        expr = expr.replace(' ', '')  # Remove spaces

        while i < len(expr):
            char = expr[i]

            if char.isdigit() or char == '.':
                # Parse number (integer or float)
                num_str = ''
                has_dot = False
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format at position {i}")
                        has_dot = True
                    num_str += expr[i]
                    i += 1
                tokens.append(('NUMBER', num_str))
                continue

            elif char in '+-*/':
                tokens.append((char, char))
                i += 1

            elif char == '(':
                tokens.append(('(', '('))
                i += 1

            elif char == ')':
                tokens.append((')', ')'))
                i += 1

            else:
                raise ValueError(f"Invalid character: '{char}' at position {i}")

        return tokens

    class _Parser:
        """Internal parser class for recursive descent parsing."""

        def __init__(self, tokens: List[Tuple[str, str]]):
            """Initialize parser with token list."""
            self.tokens = tokens
            self.pos = 0
            self.current_token = None
            self._advance()

        def _advance(self):
            """Move to next token."""
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
                self.pos += 1
            else:
                self.current_token = None

        def parse(self) -> float:
            """Parse the expression and return result."""
            result = self._parse_expression()
            return result

        def _parse_expression(self) -> float:
            """
            Parse expression: handles + and - operators.
            expression → term (('+'|'-') term)*
            """
            result = self._parse_term()

            while self.current_token and self.current_token[0] in ('+', '-'):
                operator = self.current_token[0]
                self._advance()
                right = self._parse_term()

                if operator == '+':
                    result = result + right
                else:  # operator == '-'
                    result = result - right

            return result

        def _parse_term(self) -> float:
            """
            Parse term: handles * and / operators.
            term → factor (('*'|'/') factor)*
            """
            result = self._parse_factor()

            while self.current_token and self.current_token[0] in ('*', '/'):
                operator = self.current_token[0]
                self._advance()
                right = self._parse_factor()

                if operator == '*':
                    result = result * right
                else:  # operator == '/'
                    if right == 0:
                        raise ValueError("Division by zero")
                    result = result / right

            return result

        def _parse_factor(self) -> float:
            """
            Parse factor: handles numbers, parentheses, and unary minus.
            factor → number | '(' expression ')' | '-' factor
            """
            if self.current_token is None:
                raise ValueError("Unexpected end of expression")

            token_type, token_value = self.current_token

            # Handle unary minus
            if token_type == '-':
                self._advance()
                # Unary minus has higher precedence, so we parse the next factor
                return -self._parse_factor()

            # Handle number
            if token_type == 'NUMBER':
                self._advance()
                return float(token_value)

            # Handle parentheses
            if token_type == '(':
                self._advance()
                result = self._parse_expression()

                if self.current_token is None or self.current_token[0] != ')':
                    raise ValueError("Missing closing parenthesis")

                self._advance()
                return result

            raise ValueError(f"Unexpected token: {token_value}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("15 / 4") == 3.75

def test_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0

def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_errors(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3")