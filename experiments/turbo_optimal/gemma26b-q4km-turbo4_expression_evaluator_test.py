import re
from typing import List, Optional

class Token:
    """Represents a single unit of the expression (number, operator, or parenthesis)."""
    def __init__(self, type: str, value: any):
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value})"

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a Recursive Descent Parser.
    Supports +, -, *, /, unary minus, parentheses, and floating point numbers.
    """

    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates the given mathematical expression.

        Args:
            expr: A string representing the mathematical expression.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or results in division by zero.
        """
        if not expr.strip():
            raise ValueError("Expression is empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError("No valid tokens found in expression")

        result = self._parse_expression()

        # If we haven't reached the end of the tokens, there's trailing garbage
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at end of expression: {self._tokens[self._pos].value}")

        return float(result)

    def _tokenize(self, expr: str) -> List[Token]:
        """Converts the input string into a list of Tokens."""
        # Regex pattern: Numbers (including decimals), Operators, Parentheses
        token_pattern = re.compile(r'\d*\.\d+|\d+|[+\-*/()]')
        tokens = []
        
        # Check for invalid characters by comparing reconstructed string length
        # This is a simple way to detect characters that don't match our pattern
        clean_expr = re.sub(r'\s+', '', expr)
        found_tokens_str = ""

        for match in token_pattern.finditer(expr):
            val = match.group()
            found_tokens_str += val
            if re.match(r'^\d*\.\d+|\d+$', val):
                tokens.append(Token("NUMBER", float(val)))
            elif val in "+-*/()":
                tokens.append(Token(val, val))
            else:
                raise ValueError(f"Invalid token: {val}")

        if len(clean_expr) != len(found_tokens_str):
            # Find the first character that wasn't matched by the regex
            # This is a simplified error detection for invalid characters
            raise ValueError("Expression contains invalid characters or tokens")

        return tokens

    def _peek(self) -> Optional[Token]:
        """Returns the current token without advancing the pointer."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _eat(self, expected_type: str) -> Token:
        """Consumes the current token if it matches the expected type."""
        token = self._peek()
        if token and token.type == expected_type:
            self._pos += 1
            return token
        actual = token.value if token else "EOF"
        raise ValueError(f"Expected {expected_type} but found {actual}")

    def _parse_expression(self) -> float:
        """
        Grammar: expression -> term { ('+' | '-') term }
        Handles addition and subtraction.
        """
        node = self._parse_term()
        while self._peek() and self._peek().type in ('+', '-'):
            op = self._eat(self::class_type_from_token(self._peek())).type # Logic below
            # Simplified for implementation:
            op_token = self._tokens[self._pos - 1]
            if op_token.type == '+':
                node += self._parse_term()
            elif op_token.type == '-':
                node -= self._parse_term()
        return node

    # Refined Grammar Implementation for clarity and correctness
    def _parse_expression(self) -> float:
        """expression -> term { ('+' | '-') term }"""
        result = self._parse_term()
        while self._peek() and self._peek().type in ('+', '-'):
            op = self._peek().type
            self._pos += 1
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """term -> factor { ('*' | '/') factor }"""
        result = self._parse_factor()
        while self._peek() and self._peek().type in ('*', '/'):
            op = self._peek().type
            self._pos += 1
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """factor -> '-' factor | '+' factor | primary"""
        token = self._peek()
        if token and token.type in ('-', '+'):
            self._pos += 1
            sign = -1 if token.type == '-' else 1
            return sign * self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """primary -> NUMBER | '(' expression ')'"""
        token = self._peek()
        if not token:
            raise ValueError("Unexpected end of expression")

        if token.type == 'NUMBER':
            self._pos += 1
            return token.value
        elif token.type == '(':
            self._pos += 1  # consume '('
            result = self._parse_expression()
            if not self._peek() or self._peek().type != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._pos += 1  # consume ')'
            return result
        elif token.type == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")
        else:
            raise ValueError(f"Unexpected token: {token.value}")

# --- Pytest Tests ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    assert evaluator.evaluate("10 - 4 / 2") == 8.0
    assert evaluator.evaluate("10 / 2 + 3") == 8.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication/Division before Addition/Subtraction
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3 + 4") == 6.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(1 + 2) * 3") == 9.0
    assert evaluator.evaluate("((1 + 2) * (3 - 1))") == 6.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5 + 3") == -2.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-3.14 * 2") == -6.28

def test_error_cases():
    evaluator = ExpressionEvaluator()
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    # Invalid tokens
    with pytest.raises(ValueError):
        evaluator.evaluate("1 + @")
    
    # Empty expression
    with pytest.raises(ValueError, match="Expression is empty"):
        evaluator.evaluate("   ")

if __name__ == "__main__":
    # This allows running the file directly to see the tests
    pytest.main([__file__])