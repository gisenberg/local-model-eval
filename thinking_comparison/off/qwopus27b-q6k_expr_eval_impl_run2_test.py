from typing import List, Tuple


class ExpressionEvaluator:
    """
    A recursive descent parser for mathematical expressions.
    Supports +, -, *, / with correct precedence, parentheses, unary minus, and floating point numbers.
    """
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result as a float.
        
        Args:
            expr: A string containing the mathematical expression to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is empty, malformed, or contains invalid tokens.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        tokens = self._tokenize(expr)
        if not tokens:
            raise ValueError("Empty expression")
        
        parser = self._Parser(tokens)
        result = parser.parse()
        
        if parser.pos < len(tokens):
            raise ValueError(f"Unexpected token: {tokens[parser.pos]}")
            
        return result
    
    def _tokenize(self, expr: str) -> List[str]:
        """
        Tokenize the input expression into a list of tokens.
        
        Args:
            expr: The input string to tokenize.
            
        Returns:
            A list of string tokens.
            
        Raises:
            ValueError: If the expression contains invalid characters or malformed numbers.
        """
        tokens: List[str] = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            if char.isspace():
                i += 1
                continue
            
            elif char in '+-*/()':
                tokens.append(char)
                i += 1
            
            elif char.isdigit() or char == '.':
                # Parse number (integer or float)
                j = i
                has_dot = False
                
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    if expr[j] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format at position {j}")
                        has_dot = True
                    j += 1
                
                if i == j:
                    raise ValueError(f"Invalid character at position {i}")
                
                tokens.append(expr[i:j])
                i = j
            
            else:
                raise ValueError(f"Invalid character: '{char}'")
        
        return tokens
    
    class _Parser:
        """
        Internal parser class for recursive descent parsing.
        """
        
        def __init__(self, tokens: List[str]):
            self.tokens = tokens
            self.pos = 0
        
        def parse(self) -> float:
            """
            Parse the expression and return the result.
            
            Returns:
                The evaluated result as a float.
                
            Raises:
                ValueError: If the expression is malformed.
            """
            result, new_pos = self._parse_expression()
            if new_pos < len(self.tokens):
                raise ValueError(f"Unexpected token: {self.tokens[new_pos]}")
            return result
        
        def _parse_expression(self) -> Tuple[float, int]:
            """
            Parse addition and subtraction (lowest precedence).
            Expression := Term (('+' | '-') Term)*
            
            Returns:
                A tuple of (result, next_position).
            """
            value, pos = self._parse_term()
            
            while pos < len(self.tokens) and self.tokens[pos] in '+-':
                op = self.tokens[pos]
                pos += 1
                right, pos = self._parse_term()
                if op == '+':
                    value += right
                else:
                    value -= right
            
            return value, pos
        
        def _parse_term(self) -> Tuple[float, int]:
            """
            Parse multiplication and division (higher precedence).
            Term := Factor (('*' | '/') Factor)*
            
            Returns:
                A tuple of (result, next_position).
            """
            value, pos = self._parse_factor()
            
            while pos < len(self.tokens) and self.tokens[pos] in '*/':
                op = self.tokens[pos]
                pos += 1
                right, pos = self._parse_factor()
                if op == '*':
                    value *= right
                else:
                    if right == 0:
                        raise ValueError("Division by zero")
                    value /= right
            
            return value, pos
        
        def _parse_factor(self) -> Tuple[float, int]:
            """
            Parse atomic values: numbers, parenthesized expressions, or unary minus.
            Factor := Number | '(' Expression ')' | '-' Factor
            
            Returns:
                A tuple of (result, next_position).
            """
            if self.pos >= len(self.tokens):
                raise ValueError("Unexpected end of expression")
            
            token = self.tokens[self.pos]
            
            if token == '(':
                self.pos += 1
                value, pos = self._parse_expression()
                if pos >= len(self.tokens) or self.tokens[pos] != ')':
                    raise ValueError("Mismatched parentheses")
                self.pos = pos + 1
                return value, self.pos
            
            elif token == '-':
                # Unary minus
                self.pos += 1
                value, pos = self._parse_factor()
                return -value, pos
            
            elif token == ')':
                raise ValueError("Unexpected closing parenthesis")
            
            elif token in '+-*/':
                raise ValueError(f"Unexpected operator: {token}")
            
            else:
                # Number
                try:
                    value = float(token)
                except ValueError:
                    raise ValueError(f"Invalid number: {token}")
                self.pos += 1
                return value, self.pos

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