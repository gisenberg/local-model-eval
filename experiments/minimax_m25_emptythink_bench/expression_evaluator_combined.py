"""
Expression Evaluator Module

A recursive descent parser for evaluating mathematical expressions
supporting +, -, *, /, parentheses, unary minus, and floating point numbers.
"""

import re
from typing import List


class ExpressionEvaluator:
    """
    A mathematical expression evaluator using recursive descent parsing.
    
    Supports:
        - Basic arithmetic: +, -, *, /
        - Operator precedence: * and / before + and -
        - Parentheses for grouping
        - Unary minus (e.g., "-3", "-(2+1)")
        - Floating point numbers (e.g., "3.14")
    
    Does NOT use eval() or ast.literal_eval().
    """
    
    def __init__(self) -> None:
        """Initialize the evaluator with empty state."""
        self._tokens: List[str] = []
        self._pos: int = 0
    
    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression and return the result.
        
        Args:
            expr: A string containing a mathematical expression.
            
        Returns:
            The result of the expression as a float.
            
        Raises:
            ValueError: For mismatched parentheses, division by zero,
                       invalid tokens, or empty expressions.
        """
        # Validate and preprocess
        if expr is None:
            raise ValueError("Expression cannot be None")
        
        expr = expr.strip()
        
        if not expr:
            raise ValueError("Empty expression")
        
        # Check for mismatched parentheses
        open_count = expr.count('(')
        close_count = expr.count(')')
        if open_count != close_count:
            raise ValueError("Mismatched parentheses")
        
        # Tokenize the expression
        self._tokens = self._tokenize(expr)
        
        if not self._tokens:
            raise ValueError("Empty expression")
        
        # Validate tokens
        self._validate_tokens()
        
        # Parse and evaluate starting from the lowest precedence
        self._pos = 0
        result = self._parse_add_sub()
        
        # Ensure all tokens were consumed
        if self._pos < len(self._tokens):
            raise ValueError(f"Invalid expression: unexpected token '{self._tokens[self._pos]}'")
        
        return result
    
    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert expression string into a list of tokens.
        
        Args:
            expr: The expression string to tokenize.
            
        Returns:
            List of tokens (numbers, operators, parentheses).
        """
        # Match numbers (including decimals) and operators/parentheses
        token_pattern = re.compile(r'\d+\.?\d*|\+|\-|\*|\/|\(|\)')
        tokens = token_pattern.findall(expr)
        
        # Handle implicit multiplication (e.g., "2(3)" or "(2)(3)")
        # This is handled by the parser logic for unary operators
        
        return tokens
    
    def _validate_tokens(self) -> None:
        """
        Validate that tokens are valid mathematical tokens.
        
        Raises:
            ValueError: If any invalid tokens are found.
        """
        valid_tokens = set('+-*/()')
        
        for token in self._tokens:
            # Check if it's a number
            try:
                float(token)
            except ValueError:
                # Not a number, check if it's a valid operator/symbol
                if token not in valid_tokens:
                    raise ValueError(f"Invalid token: '{token}'")
    
    def _parse_add_sub(self) -> float:
        """
        Parse addition and subtraction (lowest precedence).
        
        Returns:
            The result of the addition/subtraction expression.
        """
        result = self._parse_mul_div()
        
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ('+', '-'):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_mul_div()
            
            if op == '+':
                result += right
            else:
                result -= right
        
        return result
    
    def _parse_mul_div(self) -> float:
        """
        Parse multiplication and division (higher precedence than add/sub).
        
        Returns:
            The result of the multiplication/division expression.
        """
        result = self._parse_unary()
        
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ('*', '/'):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_unary()
            
            if op == '*':
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        
        return result
    
    def _parse_unary(self) -> float:
        """
        Parse unary operators (unary minus).
        
        Returns:
            The result with unary minus applied if present.
        """
        if self._pos < len(self._tokens) and self._tokens[self._pos] == '-':
            self._pos += 1
            operand = self._parse_unary()
            return -operand
        
        return self._parse_primary()
    
    def _parse_primary(self) -> float:
        """
        Parse primary expressions: numbers and parenthesized expressions.
        
        Returns:
            The value of the primary expression.
        """
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of expression")
        
        token = self._tokens[self._pos]
        
        # Handle parentheses
        if token == '(':
            self._pos += 1
            result = self._parse_add_sub()
            
            if self._pos >= len(self._tokens) or self._tokens[self._pos] != ')':
                raise ValueError("Mismatched parentheses")
            
            self._pos += 1  # Consume the closing parenthesis
            return result
        
        # Handle numbers
        try:
            value = float(token)
            self._pos += 1
            return value
        except ValueError:
            raise ValueError(f"Invalid token: '{token}'")


# ============================================================================
# Pytest Tests
# ============================================================================

import pytest


class TestExpressionEvaluator:
    """Test suite for ExpressionEvaluator."""
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        evaluator = ExpressionEvaluator()
        
        assert evaluator.evaluate("2 + 3") == 5.0
        assert evaluator.evaluate("10 - 4") == 6.0
        assert evaluator.evaluate("3 * 4") == 12.0
        assert evaluator.evaluate("15 / 3") == 5.0
        assert evaluator.evaluate("7 / 2") == 3.5
    
    def test_operator_precedence(self):
        """Test correct operator precedence (*, / before +, -)."""
        evaluator = ExpressionEvaluator()
        
        assert evaluator.evaluate("2 + 3 * 4") == 14.0
        assert evaluator.evaluate("10 - 2 * 3") == 4.0
        assert evaluator.evaluate("20 / 4 + 3") == 8.0
        assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
        assert evaluator.evaluate("10 / 2 - 1") == 4.0
    
    def test_parentheses(self):
        """Test parentheses for grouping."""
        evaluator = ExpressionEvaluator()
        
        assert evaluator.evaluate("(2 + 3) * 4") == 20.0
        assert evaluator.evaluate("2 * (3 + 4)") == 14.0
        assert evaluator.evaluate("(2 + 3) * (4 + 5)") == 45.0
        assert evaluator.evaluate("((2 + 3))") == 5.0
        assert evaluator.evaluate("(2 + (3 * 4))") == 14.0
    
    def test_unary_minus(self):
        """Test unary minus support."""
        evaluator = ExpressionEvaluator()
        
        assert evaluator.evaluate("-3") == -3.0
        assert evaluator.evaluate("--3") == 3.0
        assert evaluator.evaluate("-3 + 4") == 1.0
        assert evaluator.evaluate("-(2 + 1)") == -3.0
        assert evaluator.evaluate("-(2 * 3)") == -6.0
        assert evaluator.evaluate("-2 * 3") == -6.0
        assert evaluator.evaluate("2 * -3") == -6.0
    
    def test_error_cases(self):
        """Test error handling for various error conditions."""
        evaluator = ExpressionEvaluator()
        
        # Empty expression
        with pytest.raises(ValueError):
            evaluator.evaluate("")
        
        with pytest.raises(ValueError):
            evaluator.evaluate("   ")
        
        # Mismatched parentheses
        with pytest.raises(ValueError):
            evaluator.evaluate("(2 + 3")
        
        with pytest.raises(ValueError):
            evaluator.evaluate("2 + 3)")
        
        with pytest.raises(ValueError):
            evaluator.evaluate("((2 + 3)")
        
        # Division by zero
        with pytest.raises(ValueError):
            evaluator.evaluate("5 / 0")
        
        with pytest.raises(ValueError):
            evaluator.evaluate("10 / (2 - 2)")
        
        # Invalid tokens
        with pytest.raises(ValueError):
            evaluator.evaluate("2 @ 3")
        
        with pytest.raises(ValueError):
            evaluator.evaluate("5 % 3")
        
        # None expression
        with pytest.raises(ValueError):
            evaluator.evaluate(None)  # type: ignore


if __name__ == "__main__":
    # Quick demonstration
    evaluator = ExpressionEvaluator()
    
    test_cases = [
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "-3 + 4",
        "-(2 + 1)",
        "3.14 * 2",
        "10 / 2 / 2",
    ]
    
    print("Expression Evaluator Demo")
    print("=" * 40)
    for expr in test_cases:
        result = evaluator.evaluate(expr)
        print(f"{expr:20} = {result}")
