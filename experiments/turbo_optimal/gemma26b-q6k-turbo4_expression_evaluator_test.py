import re
from typing import List, NamedTuple, Optional

class Token(NamedTuple):
    """Represents a single lexical unit."""
    type: str
    value: str

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser.
    Supports +, -, *, /, unary minus, parentheses, and floating point numbers.
    """

    def __init__(self) -> None:
        self.tokens: List[Token] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates the given mathematical expression.

        Args:
            expr: The string expression to evaluate.

        Returns:
            The result of the evaluation as a float.

        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or results in division by zero.
        """
        if not expr.strip():
            raise ValueError("Expression is empty")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        if not self.tokens:
            raise ValueError("Expression contains no valid tokens")

        result = self._expression()

        # If we haven't consumed all tokens, there is trailing garbage
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at end of expression: {self.tokens[self.pos].value}")

        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """
        Converts the input string into a list of Token objects using regex.

        Args:
            expr: The raw input string.

        Returns:
            A list of Token objects.

        Raises:
            ValueError: If an invalid character is encountered.
        """
        token_specification = [
            ('NUMBER',   r'\d*\.\d+|\d+'),  # Integer or decimal number
            ('PLUS',     r'\+'),            # Addition
            ('MINUS',    r'-'),              # Subtraction
            ('MUL',      r'\*'),            # Multiplication
            ('DIV',      r'/'),             # Division
            ('LPAREN',   r'\('),            # Left parenthesis
            ('RPAREN',   r'\)'),            # Right parenthesis
            ('WS',       r'\s+'),           # Whitespace
        ]
        
        # Combine all patterns into one regex
        regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
        tokens: List[Token] = []
        last_index = 0

        for match in re.finditer(regex, expr):
            # Check if there was any text between the last match and this one (invalid tokens)
            if match.start() > last_index:
                invalid_part = expr[last_index:match.start()].strip()
                if invalid_part:
                    raise ValueError(f"Invalid token: '{invalid_part}'")

            kind = match.lastgroup
            value = match.group()

            if kind == 'WS':
                pass  # Skip whitespace
            else:
                tokens.append(Token(kind, value))
            
            last_index = match.end()

        # Check if there is trailing invalid text after the last match
        if last_index < len(expr) and expr[last_index:].strip():
            raise ValueError(f"Invalid token: '{expr[last_index:].strip()}'")

        return tokens

    def _peek(self) -> Optional[Token]:
        """Returns the current token without consuming it."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _eat(self, expected_type: str) -> Token:
        """
        Consumes the current token if it matches the expected type.

        Args:
            expected_type: The type of token expected.

        Returns:
            The consumed Token.

        Raises:
            ValueError: If the current token does not match the expected type.
        """
        token = self._peek()
        if token and token.type == expected_type:
            self.pos += 1
            return token
        raise ValueError(f"Expected {expected_type}, but found {token.type if token else 'EOF'}")

    def _expression(self) -> float:
        """
        Handles addition and subtraction (lowest precedence).
        Grammar: expression -> term { ('+' | '-') term }
        """
        node = self._term()
        while True:
            token = self._peek()
            if token and token.type in ('PLUS', 'MINUS'):
                op = self._eat(token.type)
                right = self._term()
                if op.type == 'PLUS':
                    node += right
                else:
                    node -= right
            else:
                break
        return node

    def _term(self) -> float:
        """
        Handles multiplication and division.
        Grammar: term -> factor { ('*' | '/') factor }
        """
        node = self._factor()
        while True:
            token = self._peek()
            if token and token.type in ('MUL', 'DIV'):
                op = self._eat(token.type)
                right = self._factor()
                if op.type == 'MUL':
                    node *= right
                else:
                    if right == 0:
                        raise ValueError("Division by zero")
                    node /= right
            else:
                break
        return node

    def _factor(self) -> float:
        """
        Handles unary minus.
        Grammar: factor -> '-' factor | primary
        """
        token = self._peek()
        if token and token.type == 'MINUS':
            self._eat('MINUS')
            return -self._factor()
        return self._primary()

    def _primary(self) -> float:
        """
        Handles numbers and parenthesized expressions (highest precedence).
        Grammar: primary -> NUMBER | '(' expression ')'
        """
        token = self._peek()
        if not token:
            raise ValueError("Unexpected end of expression")

        if token.type == 'NUMBER':
            self._eat('NUMBER')
            return float(token.value)
        
        elif token.type == 'LPAREN':
            self._eat('LPAREN')
            result = self._expression()
            self._eat('RPAREN')
            return result
        
        else:
            raise ValueError(f"Unexpected token: {token.value}")

# --- Pytest Tests ---

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("10 - 5") == 5.0
    assert evaluator.evaluate("4 * 2.5") == 10.0
    assert evaluator.evaluate("20 / 4") == 5.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Subtraction and addition left-to-right
    assert evaluator.evaluate("10 - 2 - 1") == 7.0
    # Division and multiplication left-to-right
    assert evaluator.evaluate("12 / 3 * 2") == 8.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + (4 / 2))") == 10.0
    assert evaluator.evaluate("((1 + 1) * (1 + 1))") == 4.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 + -2") == 3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-5 * -2") == 10.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Expected RPAREN"):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError, match="Unexpected token at end of expression"):
        evaluator.evaluate("1 + 2)")
    
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("1 + a")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("5 @ 2")
        
    # Empty expression
    with pytest.raises(ValueError, match="Expression is empty"):
        evaluator.evaluate("   ")