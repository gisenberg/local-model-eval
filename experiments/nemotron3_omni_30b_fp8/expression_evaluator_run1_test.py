import re
from typing import List, Tuple, Union

class ExpressionEvaluator:
    """
    A mathematical expression evaluator that supports:
    - Binary operators: +, -, *, /
    - Parentheses for grouping
    - Unary minus (e.g., '-3', '-(2+1)')
    - Floating point numbers
    
    Uses recursive descent parsing with proper operator precedence.
    
    Raises:
        ValueError: For mismatched parentheses, division by zero, invalid tokens, or empty expressions
    """
    
    def __init__(self):
        self.tokens: List[Union[str, float]] = []
        self.pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: The expression to evaluate (e.g., "3 + 4 * (2 - 1)")
            
        Returns:
            The result of the expression as a float
            
        Raises:
            ValueError: If the expression is invalid
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        self.tokens = self._tokenize(expr)
        self.pos = 0
        
        try:
            result = self._expr()
            if self.pos < len(self.tokens):
                raise ValueError(f"Invalid token at position {self.pos}")
            return result
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")
    
    def _tokenize(self, expr: str) -> List[Union[str, float]]:
        """Convert expression string into token list."""
        # Remove spaces and handle negative signs appropriately
        expr = expr.replace(' ', '')
        tokens = []
        i = 0
        
        while i < len(expr):
            char = expr[i]
            
            if char in '+-*/()':
                # Check for unary minus
                if char == '-' and (i == 0 or expr[i-1] in '+-*/('):
                    # Handle unary minus by looking ahead for number
                    j = i + 1
                    while j < len(expr) and (expr[j].isdigit() or expr[j) == '.'):
                        j += 1
                    if j > i + 1:  # Found a number
                        tokens.append('-' + expr[i+1:j])
                        i = j
                        continue
                    else:
                        tokens.append(char)
                else:
                    tokens.append(char)
                i += 1
            elif char.isdigit() or char == '.':
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                tokens.append(float(expr[i:j]))
                i = j
            else:
                raise ValueError(f"Invalid character: {char}")
                
        return tokens
    
    def _expr(self) -> float:
        """Parse and evaluate addition and subtraction expressions."""
        result = self._term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            if op == '+':
                result += self._term()
            else:
                result -= self._term()
        return result
    
    def _term(self) -> float:
        """Parse and evaluate multiplication and division expressions."""
        result = self._factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            if op == '*':
                result *= self._factor()
            else:
                divisor = self._factor()
                if divisor == 0:
                    raise ZeroDivisionError("Division by zero")
                result /= divisor
        return result
    
    def _factor(self) -> float:
        """Parse and evaluate factors (numbers, parentheses, unary operations)."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
            
        token = self.tokens[self.pos]
        
        if token == '(':
            self.pos += 1
            result = self._expr()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return result
        elif token == '-':
            self.pos += 1
            return -self._factor()
        else:
            self.pos += 1
            return token

# Pytest tests
def test_basic_addition():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3 + 4") == 7.0
    assert evaluator.evaluate("3.5 + 2.1") == 5.6

def test_operator_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # 3*4 first
    assert evaluator.evaluate("2 * 3 + 4") == 10.0  # 2*3 first
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 * -4") == -12.0
    assert evaluator.evaluate("-( -2 )") == 2.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + (4 * 5))") == 46.0
    try:
        evaluator.evaluate("(2 + 3")  # Missing closing parenthesis
        assert False, "Should raise ValueError"
    except ValueError:
        pass

def test_division_by_zero():
    evaluator = ExpressionEvaluator()
    try:
        evaluator.evaluate("5 / 0")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Division by zero" in str(e)

def test_invalid_expression():
    evaluator = ExpressionEvaluator()
    try:
        evaluator.evaluate("3 + 4 *")
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    try:
        evaluator.evaluate("3 + 4 & 5")
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    try:
        evaluator.evaluate("")
        assert False, "Should raise ValueError"
    except ValueError:
        pass