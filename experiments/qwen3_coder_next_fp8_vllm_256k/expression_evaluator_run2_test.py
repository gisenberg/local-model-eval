from typing import List, Union, Tuple
import re


class ExpressionEvaluator:
    """
    A mathematical expression evaluator using a recursive descent parser.
    
    Supports:
        - Basic arithmetic: +, -, *, /
        - Parentheses for grouping
        - Unary minus
        - Floating point numbers
    
    Raises:
        ValueError: For invalid expressions, mismatched parentheses, division by zero, etc.
    """
    
    def __init__(self):
        # Token types: NUMBER, PLUS, MINUS, MUL, DIV, LPAREN, RPAREN, EOF
        self._tokens: List[Tuple[str, Union[str, float]]] = []
        self._pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result.
        
        Args:
            expr: A string containing a mathematical expression
            
        Returns:
            The numerical result as a float
            
        Raises:
            ValueError: For invalid expressions, mismatched parentheses, division by zero, etc.
        """
        if not expr or expr.strip() == "":
            raise ValueError("Empty expression")
        
        self._tokens = self._tokenize(expr)
        self._pos = 0
        
        result = self._parse_expr()
        
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token: {self._tokens[self._pos][0]}")
        
        return result
    
    def _tokenize(self, expr: str) -> List[Tuple[str, Union[str, float]]]:
        """
        Convert expression string into a list of tokens.
        
        Args:
            expr: The expression string to tokenize
            
        Returns:
            List of (token_type, value) tuples
            
        Raises:
            ValueError: For invalid tokens
        """
        tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            # Skip whitespace
            if expr[i].isspace():
                i += 1
                continue
            
            # Number (including decimals)
            if expr[i].isdigit() or expr[i] == '.':
                # Check that we're not starting with a decimal without a digit (e.g., ".5" is invalid)
                # Actually, we'll allow ".5" -> 0.5 for convenience
                start = i
                if expr[i] == '.':
                    i += 1
                    if i >= n or not (expr[i].isdigit()):
                        raise ValueError(f"Invalid token: '.' not followed by digit")
                
                # Scan the number
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    # Ensure only one decimal point
                    if expr[i] == '.':
                        if i == start:  # .5 case handled above, but double-check
                            i += 1
                            continue
                        # Another dot found -> invalid
                        if i + 1 < n and expr[i+1] == '.':
                            raise ValueError(f"Invalid token: consecutive decimal points")
                    i += 1
                
                num_str = expr[start:i]
                try:
                    num_val = float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number: '{num_str}'")
                
                tokens.append(('NUMBER', num_val))
                continue
            
            # Operators and parentheses
            if expr[i] == '+':
                tokens.append(('PLUS', '+'))
                i += 1
            elif expr[i] == '-':
                tokens.append(('MINUS', '-'))
                i += 1
            elif expr[i] == '*':
                tokens.append(('MUL', '*'))
                i += 1
            elif expr[i] == '/':
                tokens.append(('DIV', '/'))
                i += 1
            elif expr[i] == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif expr[i] == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
            else:
                raise ValueError(f"Invalid character: '{expr[i]}'")
        
        return tokens
    
    def _current_token(self) -> Tuple[str, Union[str, float]]:
        """Get the current token."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return ('EOF', None)
    
    def _advance(self) -> Tuple[str, Union[str, float]]:
        """Advance to the next token and return the current one."""
        token = self._current_token()
        self._pos += 1
        return token
    
    def _parse_expr(self) -> float:
        """Parse expression: term { (+|-) term }"""
        result = self._parse_term()
        
        while self._current_token()[0] in ('PLUS', 'MINUS'):
            op = self._advance()[0]
            right = self._parse_term()
            if op == 'PLUS':
                result += right
            else:  # MINUS
                result -= right
        
        return result
    
    def _parse_term(self) -> float:
        """Parse term: factor { (*|/) factor }"""
        result = self._parse_factor()
        
        while self._current_token()[0] in ('MUL', 'DIV'):
            op = self._advance()[0]
            right = self._parse_factor()
            if op == 'MUL':
                result *= right
            else:  # DIV
                if right == 0.0:
                    raise ValueError("Division by zero")
                result /= right
        
        return result
    
    def _parse_factor(self) -> float:
        """Parse factor: (+|-) factor | NUMBER | LPAREN expr RPAREN"""
        token = self._current_token()
        
        # Handle unary plus/minus
        if token[0] == 'PLUS':
            self._advance()
            return self._parse_factor()
        elif token[0] == 'MINUS':
            self._advance()
            return -self._parse_factor()
        
        # Number
        if token[0] == 'NUMBER':
            self._advance()
            return token[1]
        
        # Parentheses
        if token[0] == 'LPAREN':
            self._advance()  # consume '('
            result = self._parse_expr()
            
            # Check for closing parenthesis
            if self._current_token()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses: expected ')'")
            self._advance()  # consume ')'
            return result
        
        # If we get here, token is unexpected
        if token[0] == 'EOF':
            raise ValueError("Unexpected end of expression")
        raise ValueError(f"Unexpected token: {token[0]}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 5 + 2") == 7.0

def test_precedence_and_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 + (3 * 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * 4) - 1") == 19.0

def test_unary_minus_and_floats(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-5.5 + 2.5") == -3.0
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)

def test_division_by_zero(evaluator):
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (2 - 2)")

def test_invalid_expressions(evaluator):
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + * 3")
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 @ 3")