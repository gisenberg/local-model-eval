

Here's a complete implementation of a mathematical expression evaluator in Python, following all your requirements:

```python
import re
from typing import List, Union, Tuple

class Token:
    """Token class representing a lexical token."""
    NUMBER = 'NUMBER'
    PLUS = 'PLUS'
    MINUS = 'MINUS'
    MULTIPLY = 'MULTIPLY'
    DIVIDE = 'DIVIDE'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    EOF = 'EOF'
    
    def __init__(self, type_: str, value: Union[float, str, None] = None):
        self.type = type_
        self.value = value
        
    def __repr__(self):
        return f"Token({self.type}, {self.value})"


class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    
    Supports:
    - Basic arithmetic operators: +, -, *, /
    - Parentheses for grouping
    - Unary minus
    - Floating point numbers
    
    Raises:
        ValueError: For invalid expressions, mismatched parentheses, 
                    division by zero, or invalid tokens.
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.pos: int = 0
    
    def tokenize(self, expr: str) -> List[Token]:
        """
        Convert input string into a list of tokens.
        
        Args:
            expr: Input expression string
            
        Returns:
            List of Token objects
            
        Raises:
            ValueError: If expression is empty or contains invalid tokens
        """
        if not expr or expr.strip() == "":
            raise ValueError("Empty expression")
        
        # Remove whitespace
        expr = expr.strip()
        
        tokens = []
        i = 0
        
        while i < len(expr):
            char = expr[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
            
            # Handle numbers (including floating point)
            if char.isdigit() or (char == '.' and i + 1 < len(expr) and expr[i+1].isdigit()):
                # Check for number that starts with a decimal point like .5 (convert to 0.5)
                if char == '.':
                    j = i
                    while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                        j += 1
                    num_str = expr[i:j]
                    try:
                        value = float(num_str)
                    except ValueError:
                        raise ValueError(f"Invalid number: '{num_str}'")
                    tokens.append(Token(Token.NUMBER, value))
                    i = j
                    continue
                
                # Regular number starting with digit
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                num_str = expr[i:j]
                try:
                    value = float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number: '{num_str}'")
                tokens.append(Token(Token.NUMBER, value))
                i = j
                continue
            
            # Handle operators and parentheses
            if char == '+':
                tokens.append(Token(Token.PLUS, '+'))
                i += 1
            elif char == '-':
                tokens.append(Token(Token.MINUS, '-'))
                i += 1
            elif char == '*':
                tokens.append(Token(Token.MULTIPLY, '*'))
                i += 1
            elif char == '/':
                tokens.append(Token(Token.DIVIDE, '/'))
                i += 1
            elif char == '(':
                tokens.append(Token(Token.LPAREN, '('))
                i += 1
            elif char == ')':
                tokens.append(Token(Token.RPAREN, ')'))
                i += 1
            else:
                raise ValueError(f"Invalid character: '{char}'")
        
        if not tokens:
            raise ValueError("Empty expression")
            
        return tokens
    
    def parse(self, tokens: List[Token]) -> float:
        """
        Parse and evaluate token list using recursive descent.
        
        Grammar:
            expr     -> term ((PLUS | MINUS) term)*
            term     -> factor ((MULTIPLY | DIVIDE) factor)*
            factor   -> (PLUS | MINUS)* unary
            unary    -> NUMBER | LPAREN expr RPAREN
        
        Args:
            tokens: List of Token objects
            
        Returns:
            Evaluated result as float
            
        Raises:
            ValueError: For mismatched parentheses, division by zero, etc.
        """
        self.tokens = tokens
        self.pos = 0
        result = self.expr()
        
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
        
        return result
    
    def expr(self) -> float:
        """Parse expression: term ((PLUS | MINUS) term)*"""
        left = self.term()
        
        while (self.pos < len(self.tokens) and 
               self.tokens[self.pos].type in (Token.PLUS, Token.MINUS)):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.term()
            
            if op.type == Token.PLUS:
                left += right
            else:  # MINUS
                left -= right
        
        return left
    
    def term(self) -> float:
        """Parse term: factor ((MULTIPLY | DIVIDE) factor)*"""
        left = self.factor()
        
        while (self.pos < len(self.tokens) and 
               self.tokens[self.pos].type in (Token.MULTIPLY, Token.DIVIDE)):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.factor()
            
            if op.type == Token.MULTIPLY:
                left *= right
            else:  # DIVIDE
                if right == 0.0:
                    raise ValueError("Division by zero")
                left /= right
        
        return left
    
    def factor(self) -> float:
        """Parse factor: (PLUS | MINUS)* unary"""
        # Handle unary plus/minus (multiple allowed: --3, ++++++5, etc.)
        sign = 1
        while (self.pos < len(self.tokens) and 
               self.tokens[self.pos].type in (Token.PLUS, Token.MINUS)):
            if self.tokens[self.pos].type == Token.MINUS:
                sign *= -1
            self.pos += 1
        
        return sign * self.unary()
    
    def unary(self) -> float:
        """Parse unary: NUMBER | LPAREN expr RPAREN"""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        
        token = self.tokens[self.pos]
        
        if token.type == Token.NUMBER:
            self.pos += 1
            return token.value
        
        if token.type == Token.LPAREN:
            self.pos += 1  # consume '('
            result = self.expr()
            
            if self.pos >= len(self.tokens) or self.tokens[self.pos].type != Token.RPAREN:
                raise ValueError("Mismatched parentheses")
            
            self.pos += 1  # consume ')'
            return result
        
        if token.type == Token.RPAREN:
            raise ValueError("Mismatched parentheses: unexpected ')'")
        
        raise ValueError(f"Unexpected token: {token}")


    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Args:
            expr: Mathematical expression string
            
        Returns:
            Result of evaluation as float
            
        Raises:
            ValueError: For invalid expressions, mismatched parentheses,
                       division by zero, or invalid tokens
        """
        tokens = self.tokenize(expr)
        return self.parse(tokens)
```

Now here are 5 comprehensive pytest tests:

```python
# test_evaluator.py
import pytest
from your_module import ExpressionEvaluator  # Replace 'your_module' with actual module name

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_operations(evaluator):
    """Test basic arithmetic operations with precedence."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("2 * 3") == 6.0
    assert evaluator.evaluate("6 / 2") == 3.0
    assert evaluator.evaluate("10 - 3") == 7.0

def test_precedence_and_associativity(evaluator):
    """Test operator precedence and associativity."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 3 - 2") == 5.0  # left associative
    assert evaluator.evaluate("2 * 3 + 4") == 10.0
    assert evaluator.evaluate("8 / 4 / 2") == 1.0  # left associative

def test_parentheses_and_unary_minus(evaluator):
    """Test parentheses and unary minus."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--3") == 3.0
    assert evaluator.evaluate("-(2 + (3 * 4))") == -14.0
    assert evaluator.evaluate("-( -(5) )") == 5.0

def test_floating_point_and_edge_cases(evaluator):
    """Test floating point numbers and edge cases."""
    assert evaluator.evaluate("3.14") == 3.14
    assert evaluator.evaluate("1.5 * 2") == 3.0
    assert evaluator.evaluate("0.1 + 0.2") == pytest.approx(0.3, rel=1e-9)
    assert evaluator.evaluate("  2 + 3  ") == 5.0
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("2.") == 2.0

def test_error_cases(evaluator):
    """Test error handling for invalid expressions."""
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")
    
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")
    
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + a")
    
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("5 *")
```

Key features of this implementation:

1. **Recursive Descent Parser**: Implements proper grammar rules with `expr`, `term`, `factor`, `unary` methods in the correct precedence order.

2. **Unary Minus Support**: The `factor` method handles multiple unary operators in sequence (like `--3` or `----5`).

3. **Proper Precedence**: `*` and `/` have higher precedence than `+` and `-`.

4. **Floating Point Support**: Handles decimals, leading decimals (like `.5`), and trailing decimals (like `2.`).

5. **Comprehensive Error Handling**:
   - Empty expressions
   - Mismatched parentheses
   - Division by zero
   - Invalid characters/tokens

6. **Type Hints**: All methods and classes include proper type annotations.

7. **Documentation**: Full docstrings explaining behavior, parameters, and exceptions.

To use, save the code as `evaluator.py` and run:
```bash
pytest test_evaluator.py
```

Make sure to update the import statement in the test file to match where you've saved the evaluator class.