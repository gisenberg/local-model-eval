"""
Mathematical Expression Evaluator using Recursive Descent Parsing.
Supports +, -, *, /, parentheses, unary minus, and floating-point numbers.
"""

import re
from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    Evaluates mathematical expressions safely without using eval().
    
    Supports:
    - Binary operators: +, -, *, /
    - Unary minus: -
    - Parentheses: ()
    - Floating point numbers
    
    Raises:
        ValueError: For syntax errors, mismatched parentheses, 
                    division by zero, or invalid tokens.
    """

    def __init__(self):
        self.tokens: List[Tuple[str, Union[float, None]]] = []
        self.pos: int = 0

    def _lex(self, expression: str) -> List[Tuple[str, Union[float, None]]]:
        """
        Converts the input string into a list of tokens.
        
        Args:
            expression: The string representation of the mathematical expression.
            
        Returns:
            List of tuples representing (type, value).
            
        Raises:
            ValueError: If an invalid character is found.
        """
        tokens = []
        i = 0
        n = len(expression)
        
        while i < n:
            char = expression[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
            
            # Numbers (including floats)
            if char.isdigit() or char == '.':
                start = i
                has_dot = False
                while i < n and (expression[i].isdigit() or expression[i] == '.'):
                    if expression[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format near index {i}")
                        has_dot = True
                    i += 1
                
                num_str = expression[start:i]
                # Validate number
                if num_str in ['.', '..', '-']:
                    raise ValueError(f"Invalid number: '{num_str}'")
                try:
                    num_val = float(num_str)
                    tokens.append(('NUMBER', num_val))
                except ValueError:
                    raise ValueError(f"Invalid number: '{num_str}'")
                continue
            
            # Operators and Parens
            if char == '+':
                tokens.append(('PLUS', None))
                i += 1
            elif char == '-':
                tokens.append(('MINUS', None))
                i += 1
            elif char == '*':
                tokens.append(('MUL', None))
                i += 1
            elif char == '/':
                tokens.append(('DIV', None))
                i += 1
            elif char == '(':
                tokens.append(('LPAREN', None))
                i += 1
            elif char == ')':
                tokens.append(('RPAREN', None))
                i += 1
            else:
                raise ValueError(f"Invalid token: '{char}' at position {i}")
        
        return tokens

    def evaluate(self, expr: str) -> float:
        """
        Parses and evaluates a mathematical expression string.
        
        Args:
            expr: A string containing the mathematical expression.
            
        Returns:
            The calculated result as a float.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or division by zero occurs.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        self.tokens = self._lex(expr)
        self.pos = 0
        
        if not self.tokens:
            raise ValueError("Empty expression")
            
        result = self._parse_expression()
        
        # Ensure all tokens were consumed
        if self.pos < len(self.tokens):
            remaining = ''.join(t[0] for t in self.tokens[self.pos:])
            raise ValueError(f"Unexpected characters after valid expression: {remaining}")
            
        return result

    def _current_token(self) -> Tuple[str, Union[float, None]]:
        """Returns the current token, or EOF marker if at end."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ('EOF', None)

    def _consume(self, expected_type: str) -> Tuple[str, Union[float, None]]:
        """Consumes the current token if it matches the expected type."""
        tok = self._current_token()
        if tok[0] != expected_type:
            if expected_type == 'EOF':
                raise ValueError(f"Expected end of expression, got '{tok[0]}'")
            raise ValueError(f"Expected {expected_type}, got '{tok[0]}'")
        
        self.pos += 1
        return tok

    def _parse_expression(self) -> float:
        """
        Parses an expression handling addition and subtraction.
        Grammar: Expression -> Term { (+|-) Term }
        """
        left = self._parse_term()
        
        while True:
            tok = self._current_token()
            if tok[0] == 'PLUS':
                self._consume('PLUS')
                right = self._parse_term()
                left += right
            elif tok[0] == 'MINUS':
                self._consume('MINUS')
                right = self._parse_term()
                left -= right
            else:
                break
        
        return left

    def _parse_term(self) -> float:
        """
        Parses a term handling multiplication and division.
        Grammar: Term -> Factor { (*|/) Factor }
        """
        left = self._parse_factor()
        
        while True:
            tok = self._current_token()
            if tok[0] == 'MUL':
                self._consume('MUL')
                right = self._parse_factor()
                left *= right
            elif tok[0] == 'DIV':
                self._consume('DIV')
                right = self._parse_factor()
                if right == 0:
                    raise ValueError("Division by zero")
                left /= right
            else:
                break
        
        return left

    def _parse_factor(self) -> float:
        """
        Parses a factor handling unary minus and primary values.
        Grammar: Factor -> [-] Primary
        """
        if self._current_token()[0] == 'MINUS':
            self._consume('MINUS')
            return -self._parse_factor()
        
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """
        Parses primary values: numbers or parenthesized expressions.
        Grammar: Primary -> NUMBER | ( Expression )
        """
        tok = self._current_token()
        
        if tok[0] == 'NUMBER':
            self._consume('NUMBER')
            return float(tok[1])
        elif tok[0] == 'LPAREN':
            self._consume('LPAREN')
            val = self._parse_expression()
            self._consume('RPAREN')
            return val
        elif tok[0] == 'EOF':
            raise ValueError("Unexpected end of expression")
        elif tok[0] == 'RPAREN':
            # Should not be reached normally due to _parse_expression logic, 
            # but covers mismatched closing parenthesis cases earlier in chain
            raise ValueError("Unexpected ')'")
        else:
            raise ValueError(f"Unexpected token '{tok[0]}'")


# Pytest Tests
if __name__ == '__main__':
    import pytest

    def test_basic_arithmetic():
        ev = ExpressionEvaluator()
        assert ev.evaluate("2 + 2") == 4.0
        assert ev.evaluate("10 - 3") == 7.0
        assert ev.evaluate("4 * 5") == 20.0
        assert ev.evaluate("20 / 4") == 5.0

    def test_operator_precedence_and_parentheses():
        ev = ExpressionEvaluator()
        # Multiplication before addition
        assert ev.evaluate("2 + 3 * 4") == 14.0
        # Division before subtraction
        assert ev.evaluate("10 - 20 / 4") == 5.0
        # Parentheses override precedence
        assert ev.evaluate("(2 + 3) * 4") == 20.0
        # Nested parentheses
        assert ev.evaluate("((2))") == 2.0

    def test_unary_minus_and_negatives():
        ev = ExpressionEvaluator()
        assert ev.evaluate("-5") == -5.0
        assert ev.evaluate("--5") == 5.0
        assert ev.evaluate("3 * -2") == -6.0
        assert ev.evaluate("- (2 + 1)") == -3.0
        assert ev.evaluate("5 + - 3") == 2.0

    def test_floating_point():
        ev = ExpressionEvaluator()
        assert abs(ev.evaluate("3.14 * 2") - 6.28) < 0.0001
        assert abs(ev.evaluate(".5 + .5") - 1.0) < 0.0001
        assert ev.evaluate("10 / 3") == 10.0 / 3.0

    def test_error_handling():
        ev = ExpressionEvaluator()
        # Division by zero
        try:
            ev.evaluate("5 / 0")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        
        # Mismatched parentheses
        try:
            ev.evaluate("(2 + 2")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
            
        try:
            ev.evaluate("2 + 2)")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        # Invalid tokens
        try:
            ev.evaluate("2 @ 3")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
            
        # Empty expression
        try:
            ev.evaluate("")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
            
        # Trailing garbage
        try:
            ev.evaluate("2 + 2 +")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    # Run tests manually if executed directly
    print("Running tests...")
    test_basic_arithmetic()
    test_operator_precedence_and_parentheses()
    test_unary_minus_and_negatives()
    test_floating_point()
    test_error_handling()
    print("All tests passed.")