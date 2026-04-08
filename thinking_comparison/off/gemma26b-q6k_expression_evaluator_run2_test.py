import re
from typing import List, Optional

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that uses a recursive descent parser
    to handle arithmetic operations, precedence, and parentheses.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates a mathematical expression string.
        
        Args:
            expr: The string expression to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: For division by zero, mismatched parentheses, or invalid tokens.
        """
        if not expr.strip():
            raise ValueError("Expression is empty")

        self._tokens = self._tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError("Expression contains no valid tokens")

        result = self._parse_expression()

        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token at end: {self._tokens[self._pos]}")

        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of tokens (numbers, operators, parens).
        """
        # Regex matches: floats/integers, operators (+, -, *, /), or parentheses
        token_pattern = r'\d*\.\d+|\d+|[+\-*/()]'
        tokens = re.findall(token_pattern, expr)
        
        # Validate that the reconstructed string from tokens matches the non-whitespace input
        # This ensures we catch invalid characters like 'abc' or '$'
        cleaned_expr = re.sub(r'\s+', '', expr)
        reconstructed = "".join(tokens)
        
        # Check if there are characters in the original string that weren't captured by the regex
        # This is a simple way to detect "invalid tokens"
        if len(re.sub(r'\s+', '', "".join(re.findall(r'\S', expr))), 0) != 0:
            # We check if the sum of lengths of tokens (ignoring whitespace) matches the original
            # This is a bit complex due to regex, so we'll check if any non-token chars exist
            all_chars_pattern = r'\d*\.\d+|\d+|[+\-*/()]|\s+'
            if len(re.sub(all_chars_pattern, '', expr)) > 0:
                raise ValueError("Expression contains invalid tokens")

        return tokens

    def _peek(self) -> Optional[str]:
        """Returns the current token without consuming it."""
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        """Consumes and returns the current token."""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Handles addition and subtraction (lowest precedence).
        """
        node = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                node += right
            else:
                node -= right
        return node

    def _parse_term(self) -> float:
        """
        Handles multiplication and division.
        """
        node = self._parse_unary()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_unary()
            if op == '*':
                node *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                node /= right
        return node

    def _parse_unary(self) -> float:
        """
        Handles unary minus.
        """
        if self._peek() == '-':
            self._consume()
            return -self._parse_unary()
        return self._parse_factor()

    def _parse_factor(self) -> float:
        """
        Handles numbers and parentheses (highest precedence).
        """
        token = self._peek()

        if token == '(':
            self._consume()  # consume '('
            result = self._parse_expression()
            if self._peek() != ')':
                raise ValueError("Mismatched parentheses: missing ')'")
            self._consume()  # consume ')'
            return result
        
        if token is not None and (token[0].isdigit() or token[0] == '.'):
            try:
                return float(self._consume())
            except ValueError:
                raise ValueError(f"Invalid number format: {token}")
        
        if token == ')':
            raise ValueError("Mismatched parentheses: unexpected ')'")
            
        raise ValueError(f"Unexpected token: {token if token else 'EOF'}")


# --- Pytest Tests ---
import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("10 + 5 - 2") == 13.0
    assert evaluator.evaluate("10 * 2 / 5") == 4.0
    assert evaluator.evaluate("3.14 * 2") == 6.28

def test_precedence():
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # Division before subtraction
    assert evaluator.evaluate("10 - 6 / 2") == 7.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    # Overriding precedence
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("5 * -2") == -10.0
    assert evaluator.evaluate("--5") == 5.0  # Double unary

def test_error_cases():
    evaluator = ExpressionEvaluator()
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10 / 0")
    
    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(1 + 2")
    
    # Invalid tokens
    with pytest.raises(ValueError, match="invalid tokens"):
        evaluator.evaluate("10 + abc")
        
    # Empty expression
    with pytest.raises(ValueError, match="Expression is empty"):
        evaluator.evaluate("   ")

if __name__ == "__main__":
    # Manual test run
    ev = ExpressionEvaluator()
    print(f"Result: {ev.evaluate('-(3 + 5) * 2 / 0.5')}") # Expected -32.0