from typing import List

class ExpressionEvaluator:
    def __init__(self):
        pass

    def evaluate(self, expr: str) -> float:
        """
        Evaluates the given mathematical expression string.
        
        Args:
            expr (str): The mathematical expression to evaluate.
        
        Returns:
            float: The result of the evaluated expression.
        
        Raises:
            ValueError: If the expression is invalid, has mismatched parentheses,
                        division by zero, or is empty.
        """
        tokens = self.tokenize(expr)
        if not tokens:
            raise ValueError("Empty expression")
        self.pos = 0
        self.tokens = tokens
        result = self.parse_expr()
        if self.pos != len(self.tokens):
            raise ValueError("Invalid tokens")
        return result

    def tokenize(self, expr: str) -> List[str]:
        """
        Converts the input string into a list of tokens.
        
        Args:
            expr (str): The expression string to tokenize.
        
        Returns:
            List[str]: A list of tokens (numbers, operators, parentheses).
        
        Raises:
            ValueError: If the input contains invalid characters.
        """
        tokens = []
        i = 0
        while i < len(expr):
            c = expr[i]
            if c.isspace():
                i += 1
            elif c in '+-*/()':
                tokens.append(c)
                i += 1
            elif c.isdigit() or c == '.':
                num = ''
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    num += expr[i]
                    i += 1
                tokens.append(num)
            else:
                raise ValueError("Invalid token")
        return tokens

    def parse_expr(self) -> float:
        """
        Parses an expression with addition and subtraction.
        """
        val = self.parse_term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ['+', '-']:
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.parse_term()
            if op == '+':
                val += right
            else:
                val -= right
        return val

    def parse_term(self) -> float:
        """
        Parses a term with multiplication and division.
        """
        val = self.parse_factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ['*', '/']:
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.parse_factor()
            if op == '*':
                val *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                val /= right
        return val

    def parse_factor(self) -> float:
        """
        Parses a factor, including unary minus.
        """
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '-':
            self.pos += 1
            val = self.parse_factor()
            return -val
        else:
            return self.parse_primary()

    def parse_primary(self) -> float:
        """
        Parses a primary expression (number or parenthesized expression).
        """
        if self.tokens[self.pos] == '(':
            self.pos += 1
            val = self.parse_expr()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Mismatched parentheses")
            self.pos += 1
            return val
        else:
            num_token = self.tokens[self.pos]
            self.pos += 1
            try:
                return float(num_token)
            except ValueError:
                raise ValueError("Invalid number")

import pytest

def test_basic_arithmetic():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3+4*2") == 11.0

def test_precedence():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3+4*2") == 11.0

def test_parentheses():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(3+4)*2") == 14.0

def test_unary_minus():
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(3+4)") == -7.0

def test_error_cases():
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("3+abc")
    with pytest.raises(ValueError, match="Invalid number"):
        evaluator.evaluate("3+*")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("3/0")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("3+4)")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")

evaluator = ExpressionEvaluator()
print(evaluator.evaluate("3 + 4 * (2 - 1)"))  # Output: 7.0
print(evaluator.evaluate("-3 + 4 * 2"))       # Output: 5.0