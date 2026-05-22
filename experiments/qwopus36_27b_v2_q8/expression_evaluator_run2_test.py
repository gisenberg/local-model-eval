from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports addition, subtraction, multiplication, division with correct 
    operator precedence, parentheses for grouping, unary minus, and 
    floating point numbers.
    """

    def __init__(self) -> None:
        """Initializes the evaluator with an empty token list and position pointer."""
        self.tokens: List[Tuple[str, Union[float, str]]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string.
        
        Args:
            expr: The expression string to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or contains division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        result = self._parse_expression()
        
        # Check for mismatched closing parentheses or leftover tokens
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if token[0] == ')':
                raise ValueError("Mismatched parentheses")
            raise ValueError(f"Unexpected token: {token}")
            
        return result

    def _tokenize(self, expr: str) -> List[Tuple[str, Union[float, str]]]:
        """Converts an expression string into a list of tokens."""
        tokens: List[Tuple[str, Union[float, str]]] = []
        i = 0
        n = len(expr)
        
        while i < n:
            c = expr[i]
            
            if c.isspace():
                i += 1
                continue
                
            if c.isdigit() or c == '.':
                start = i
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    i += 1
                try:
                    num = float(expr[start:i])
                except ValueError:
                    raise ValueError(f"Invalid number: {expr[start:i]}")
                tokens.append(('NUM', num))
                
            elif c in '+-*/()':
                tokens.append((c, c))
                
            else:
                raise ValueError(f"Invalid token: {c}")
                
        return tokens

    def _current_token(self) -> Tuple[str, Union[float, str]]:
        """Returns the current token or an EOF marker."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ('EOF', None)

    def _consume(self, expected_type: str = None) -> Tuple[str, Union[float, str]]:
        """Consumes and returns the current token, optionally checking its type."""
        token = self._current_token()
        if expected_type and token[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[0]}")
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parses addition and subtraction expressions.
        Grammar: Expression -> Term ((+ | -) Term)*
        """
        result = self._parse_term()
        while self._current_token()[0] in ('+', '-'):
            op = self._consume()[0]
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """
        Parses multiplication and division expressions.
        Grammar: Term -> Factor ((* | /) Factor)*
        """
        result = self._parse_factor()
        while self._current_token()[0] in ('*', '/'):
            op = self._consume()[0]
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """
        Parses factors (numbers, parenthesized expressions, unary minus).
        Grammar: Factor -> ['-'] Number | '(' Expression ')'
        """
        token = self._current_token()

        if token[0] == '(':
            self._consume()
            result = self._parse_expression()
            if self._current_token()[0] != ')':
                raise ValueError("Mismatched parentheses")
            self._consume()
            return result

        if token[0] == '-':
            self._consume()
            # Recursively handle multiple unary minuses, e.g., - - 5
            return -self._parse_factor()

        if token[0] == 'NUM':
            return self._consume()[1]

        if token[0] == 'EOF':
            raise ValueError("Unexpected end of expression")

        raise ValueError(f"Unexpected token: {token[0]}")

import pytest

@pytest.fixture
def evaluator():
    """Provides a fresh ExpressionEvaluator instance for each test."""
    return ExpressionEvaluator()

def test_precedence_and_basic_operations(evaluator):
    """Tests correct operator precedence and basic arithmetic."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("2 - 3") == -1.0
    assert evaluator.evaluate("2 * 3") == 6.0
    assert evaluator.evaluate("6 / 2") == 3.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # Multiplication before addition
    assert evaluator.evaluate("2 * 3 + 4") == 10.0

def test_parentheses_grouping(evaluator):
    """Tests correct evaluation of parenthesized expressions."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 * (5 - 2)") == 30.0
    assert evaluator.evaluate("((2 + 3) * 4) - 5") == 15.0

def test_unary_minus(evaluator):
    """Tests support for unary minus operator."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0
    assert evaluator.evaluate("2 + -3") == -1.0
    assert evaluator.evaluate("- - 5") == 5.0

def test_floating_point_numbers(evaluator):
    """Tests support for floating point numbers."""
    assert evaluator.evaluate("3.14 + 2") == 5.14
    assert evaluator.evaluate("1.5 * 4") == 6.0
    assert evaluator.evaluate("7.5 / 2.5") == 3.0
    assert evaluator.evaluate("-2.5") == -2.5

def test_value_errors(evaluator):
    """Tests that appropriate ValueErrors are raised for invalid inputs."""
    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
        
    # Mismatched parentheses (missing closing)
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
        
    # Mismatched parentheses (extra closing)
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
        
    # Invalid token
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")
        
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
        
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")