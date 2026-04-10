from typing import List, Tuple


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    Supports +, -, *, / with correct precedence, parentheses, unary minus,
    and floating point numbers.
    """

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result as a float.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The result of the evaluation as a float
            
        Raises:
            ValueError: If the expression is empty, invalid, has mismatched 
                       parentheses, or contains division by zero
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Empty expression")
            
        result, pos = self._parse_expression(tokens, 0)
        if pos < len(tokens):
            raise ValueError(f"Unexpected token '{tokens[pos]}' at position {pos}")
        return result

    def _tokenize(self, expr: str) -> List[str]:
        """
        Tokenize the expression into numbers, operators, and parentheses.
        
        Args:
            expr: The input expression string
            
        Returns:
            A list of tokens (numbers as strings, operators, and parentheses)
            
        Raises:
            ValueError: If an invalid character is encountered
        """
        tokens: List[str] = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
            
            if char.isdigit() or char == '.':
                # Parse number (integer or float)
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format at position {i}: multiple decimal points")
                        has_dot = True
                    j += 1
                if j == i:
                    raise ValueError(f"Invalid character '{char}' at position {i}")
                tokens.append(expr[i:j])
                i = j
            elif char in '+-*/()':
                tokens.append(char)
                i += 1
            else:
                raise ValueError(f"Invalid character '{char}' at position {i}")
        
        return tokens

    def _parse_expression(self, tokens: List[str], pos: int) -> Tuple[float, int]:
        """
        Parse addition and subtraction (lowest precedence).
        
        Args:
            tokens: List of tokens
            pos: Current position in tokens
            
        Returns:
            Tuple of (result, new_position)
        """
        value, pos = self._parse_term(tokens, pos)
        
        while pos < len(tokens) and tokens[pos] in ('+', '-'):
            op = tokens[pos]
            pos += 1
            right, pos = self._parse_term(tokens, pos)
            if op == '+':
                value += right
            else:
                value -= right
        
        return value, pos

    def _parse_term(self, tokens: List[str], pos: int) -> Tuple[float, int]:
        """
        Parse multiplication and division.
        
        Args:
            tokens: List of tokens
            pos: Current position in tokens
            
        Returns:
            Tuple of (result, new_position)
        """
        value, pos = self._parse_factor(tokens, pos)
        
        while pos < len(tokens) and tokens[pos] in ('*', '/'):
            op = tokens[pos]
            pos += 1
            right, pos = self._parse_factor(tokens, pos)
            if op == '*':
                value *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                value /= right
        
        return value, pos

    def _parse_factor(self, tokens: List[str], pos: int) -> Tuple[float, int]:
        """
        Parse numbers, parentheses, and unary minus.
        
        Args:
            tokens: List of tokens
            pos: Current position in tokens
            
        Returns:
            Tuple of (result, new_position)
        """
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")
        
        token = tokens[pos]
        
        if token == '-':
            # Unary minus
            pos += 1
            value, pos = self._parse_factor(tokens, pos)
            return -value, pos
        
        if token == '(':
            pos += 1
            value, pos = self._parse_expression(tokens, pos)
            if pos >= len(tokens) or tokens[pos] != ')':
                raise ValueError("Mismatched parentheses: expected ')'")
            pos += 1
            return value, pos
        
        try:
            value = float(token)
            return value, pos + 1
        except ValueError:
            raise ValueError(f"Invalid token '{token}' at position {pos}")

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