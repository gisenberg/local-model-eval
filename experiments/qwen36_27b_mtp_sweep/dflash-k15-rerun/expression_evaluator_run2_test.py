from typing import List, Tuple, Any

class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports:
    - Addition (+), Subtraction (-), Multiplication (*), Division (/)
    - Correct operator precedence (* and / before + and -)
    - Parentheses for grouping
    - Unary minus (e.g., -3, -(2+1))
    - Floating point numbers (e.g., 3.14, .5, 3.)
    
    Raises ValueError for:
    - Empty expressions
    - Invalid tokens
    - Mismatched parentheses
    - Division by zero
    """
    
    def __init__(self) -> None:
        self.tokens: List[Tuple[str, Any]] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluates a mathematical expression string and returns the result.
        
        Args:
            expr: A string containing a mathematical expression.
            
        Returns:
            The evaluated result as a float.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        self._tokenize(expr)
        self.pos = 0
        
        if self._current_token() == 'EOF':
            raise ValueError("Empty expression")
            
        result = self._parse_expression()
        
        if self._current_token() != 'EOF':
            raise ValueError("Invalid token or mismatched parentheses")
            
        return result

    def _tokenize(self, expr: str) -> None:
        """Converts the input string into a list of tokens."""
        tokens = []
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
                    tokens.append(('NUM', float(num_str)))
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")
                i = j
                
            elif expr[i] in '+-*/()':
                tokens.append((expr[i], None))
                i += 1
            else:
                raise ValueError(f"Invalid token: {expr[i]}")
                
        tokens.append(('EOF', None))
        self.tokens = tokens

    def _current_token(self) -> str:
        """Returns the type of the current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos][0]
        return 'EOF'

    def _advance(self) -> None:
        """Moves to the next token."""
        self.pos += 1

    def _parse_expression(self) -> float:
        """Parses addition and subtraction (lowest precedence)."""
        result = self._parse_term()
        while self._current_token() in ('+', '-'):
            op = self._current_token()
            self._advance()
            right = self._parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        """Parses multiplication and division (higher precedence)."""
        result = self._parse_factor()
        while self._current_token() in ('*', '/'):
            op = self._current_token()
            self._advance()
            right = self._parse_factor()
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        """Parses unary plus and minus."""
        if self._current_token() in ('+', '-'):
            op = self._current_token()
            self._advance()
            val = self._parse_factor()
            return val if op == '+' else -val
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parses numbers and parenthesized expressions."""
        token_type = self._current_token()
        
        if token_type == '(':
            self._advance()
            result = self._parse_expression()
            if self._current_token() != ')':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return result
        elif token_type == 'NUM':
            self._advance()
            return self.tokens[self.pos - 1][1]
        elif token_type == ')':
            raise ValueError("Mismatched parentheses")
        else:
            raise ValueError(f"Invalid token: {token_type}")

import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_operator_precedence(evaluator):
    """Tests that * and / are evaluated before + and -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 / 2") == 9.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses_grouping(evaluator):
    """Tests that parentheses correctly override default precedence."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0
    assert evaluator.evaluate("10 / (2 + 3)") == pytest.approx(2.0)

def test_unary_minus(evaluator):
    """Tests unary minus at the start and after operators."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-3 + 5") == 2.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0
    assert evaluator.evaluate("- -5") == 5.0

def test_floating_point_numbers(evaluator):
    """Tests support for decimal numbers."""
    assert evaluator.evaluate("3.14 * 2") == pytest.approx(6.28)
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("10.0 / 4.0") == 2.5

def test_error_handling(evaluator):
    """Tests that appropriate ValueErrors are raised for invalid inputs."""
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("1 / 0")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
        
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
        
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + abc")
        
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
        
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")