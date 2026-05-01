class ExpressionEvaluator:
    def __init__(self, expr: str) -> None:
        """Initialize the evaluator with the given expression string."""
        self.tokens = self.tokenize(expr)
        self.pos = 0

    def tokenize(self, expr: str) -> list:
        """Tokenize the expression into numbers, operators, and parentheses."""
        tokens = []
        i = 0
        n = len(expr)
        while i < n:
            if expr[i] == ' ':
                i += 1
                continue
            if expr[i] in '0123456789.':
                start = i
                while i < n:
                    c = expr[i]
                    if c in '0123456789' or c == '.':
                        i += 1
                    else:
                        break
                token = expr[start:i]
                if token.count('.') > 1 or not any(c.isdigit() for c in token):
                    raise ValueError(f"Invalid number: {token}")
                tokens.append(token)
            elif expr[i] in '+-*/()':
                tokens.append(expr[i])
                i += 1
            else:
                raise ValueError(f"Invalid character: {expr[i]}")
        return tokens

    def evaluate(self) -> float:
        """Evaluate the mathematical expression and return the result as a float."""
        return self.parse_expression()

    def parse_expression(self) -> float:
        """Parse an expression, which consists of terms separated by '+' or '-' operators."""
        result = self.parse_term()
        while self.current_token() in ['+', '-']:
            op = self.current_token()
            self.consume()
            term = self.parse_term()
            if op == '+':
                result += term
            else:
                result -= term
        return result

    def parse_term(self) -> float:
        """Parse a term, which consists of factors separated by '*' or '/' operators."""
        result = self.parse_factor()
        while self.current_token() in ['*', '/']:
            op = self.current_token()
            self.consume()
            factor = self.parse_factor()
            if op == '*':
                result *= factor
            else:
                if factor == 0:
                    raise ValueError("Division by zero")
                result /= factor
        return result

    def parse_factor(self) -> float:
        """Parse a factor, which can be a unary minus, a parenthesized expression, or a number."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of input")
        if self.current_token() == '-':
            self.consume()
            return -self.parse_factor()
        elif self.current_token() == '(':
            self.consume()
            result = self.parse_expression()
            if self.current_token() != ')':
                raise ValueError("Mismatched parentheses")
            self.consume()
            return result
        else:
            num_str = self.current_token()
            self.consume()
            return float(num_str)

    def current_token(self) -> str:
        """Get the current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> None:
        """Consume the current token, advancing the position."""
        if self.pos < len(self.tokens):
            self.pos += 1
        else:
            raise ValueError("Unexpected end of input")


import pytest

def test_empty_expression():
    evaluator = ExpressionEvaluator("")
    with pytest.raises(ValueError):
        evaluator.evaluate()

def test_division_by_zero():
    evaluator = ExpressionEvaluator("5/0")
    with pytest.raises(ValueError) as exc_info:
        evaluator.evaluate()
    assert "Division by zero" in str(exc_info.value)

def test_mismatched_parentheses():
    evaluator = ExpressionEvaluator("(1+2")
    with pytest.raises(ValueError) as exc_info:
        evaluator.evaluate()
    assert "Mismatched parentheses" in str(exc_info.value)

def test_unary_minus():
    evaluator = ExpressionEvaluator("-3")
    assert evaluator.evaluate() == -3.0

def test_operator_precedence():
    evaluator = ExpressionEvaluator("2 + 3 * 4")
    assert evaluator.evaluate() == 14.0