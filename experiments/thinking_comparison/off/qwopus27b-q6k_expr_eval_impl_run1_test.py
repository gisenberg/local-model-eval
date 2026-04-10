from typing import List, Union, Optional

class Token:
    """Represents a token in the expression."""
    def __init__(self, type_: str, value: Union[str, float, None]):
        self.type = type_
        self.value = value
    
    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value})"

class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    
    Supports +, -, *, / with correct precedence, parentheses, unary minus,
    and floating point numbers.
    """
    
    def __init__(self) -> None:
        self.tokens: List[Token] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: A string containing a mathematical expression.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is empty, has mismatched parentheses,
                       contains invalid tokens, or attempts division by zero.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
        
        self.tokenize(expr)
        self.pos = 0
        return self.parse()
    
    def tokenize(self, expr: str) -> None:
        """
        Convert the expression string into a list of tokens.
        
        Args:
            expr: The expression string to tokenize.
            
        Raises:
            ValueError: If an invalid character is encountered.
        """
        self.tokens = []
        i = 0
        n = len(expr)
        
        while i < n:
            char = expr[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
            
            # Parse numbers (integers and floats)
            if char.isdigit() or char == '.':
                j = i
                has_dot = False
                while j < n and (expr[j].isdigit() or (expr[j] == '.' and not has_dot)):
                    if expr[j] == '.':
                        has_dot = True
                    j += 1
                
                if i == j:
                    raise ValueError(f"Invalid character at position {i}")
                
                num_str = expr[i:j]
                if num_str == '.':
                    raise ValueError(f"Invalid number at position {i}")
                
                self.tokens.append(Token('NUMBER', float(num_str)))
                i = j
                continue
            
            # Parse operators and parentheses
            if char == '+':
                self.tokens.append(Token('PLUS', '+'))
                i += 1
                continue
            
            if char == '-':
                self.tokens.append(Token('MINUS', '-'))
                i += 1
                continue
            
            if char == '*':
                self.tokens.append(Token('MULTIPLY', '*'))
                i += 1
                continue
            
            if char == '/':
                self.tokens.append(Token('DIVIDE', '/'))
                i += 1
                continue
            
            if char == '(':
                self.tokens.append(Token('LPAREN', '('))
                i += 1
                continue
            
            if char == ')':
                self.tokens.append(Token('RPAREN', ')'))
                i += 1
                continue
            
            raise ValueError(f"Invalid character '{char}' at position {i}")
        
        self.tokens.append(Token('EOF', None))
    
    def parse(self) -> float:
        """
        Parse the tokenized expression and return the result.
        
        Returns:
            The result of the parsed expression.
            
        Raises:
            ValueError: If there are unexpected tokens after the expression.
        """
        result = self.parse_expression()
        if self.current_token().type != 'EOF':
            token = self.current_token()
            raise ValueError(f"Unexpected token '{token.value}' after expression")
        return result
    
    def current_token(self) -> Token:
        """Return the current token."""
        return self.tokens[self.pos]
    
    def advance(self) -> None:
        """Advance to the next token."""
        self.pos += 1
    
    def parse_expression(self) -> float:
        """
        Parse an expression (handles + and -).
        
        Grammar: Expression -> Term (('+' | '-') Term)*
        
        Returns:
            The result of the expression.
        """
        left = self.parse_term()
        
        while self.current_token().type in ('PLUS', 'MINUS'):
            op = self.current_token()
            self.advance()
            right = self.parse_term()
            
            if op.type == 'PLUS':
                left = left + right
            else:
                left = left - right
        
        return left
    
    def parse_term(self) -> float:
        """
        Parse a term (handles * and /).
        
        Grammar: Term -> Factor (('*' | '/') Factor)*
        
        Returns:
            The result of the term.
            
        Raises:
            ValueError: If division by zero is attempted.
        """
        left = self.parse_factor()
        
        while self.current_token().type in ('MULTIPLY', 'DIVIDE'):
            op = self.current_token()
            self.advance()
            right = self.parse_factor()
            
            if op.type == 'MULTIPLY':
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
        
        return left
    
    def parse_factor(self) -> float:
        """
        Parse a factor (handles numbers, parentheses, and unary operators).
        
        Grammar: Factor -> Number | '(' Expression ')' | '-' Factor | '+' Factor
        
        Returns:
            The result of the factor.
            
        Raises:
            ValueError: If an unexpected token is encountered or parentheses are mismatched.
        """
        token = self.current_token()
        
        if token.type == 'EOF':
            raise ValueError("Expected operand but found end of expression")
        
        if token.type == 'MINUS':
            self.advance()
            return -self.parse_factor()
        
        if token.type == 'PLUS':
            self.advance()
            return self.parse_factor()
        
        if token.type == 'NUMBER':
            value = token.value
            self.advance()
            return value
        
        if token.type == 'LPAREN':
            self.advance()
            result = self.parse_expression()
            if self.current_token().type != 'RPAREN':
                raise ValueError("Mismatched parentheses: expected ')'")
            self.advance()
            return result
        
        raise ValueError(f"Unexpected token '{token.value}'")

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