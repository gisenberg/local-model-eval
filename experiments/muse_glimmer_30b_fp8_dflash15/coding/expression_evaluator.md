```python
# expression_evaluator.py
from typing import List, Tuple, Union

class ExpressionEvaluator:
    """Recursive descent evaluator for + - * / with parentheses and unary minus."""

    def evaluate(self, expr: str) -> float:
        """Evaluate expr and return a float. Raises ValueError on bad input."""
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens: List[Tuple[str, Union[str, None]]] = self._tokenize(expr)
        self.pos: int = 0
        result = self._parse_expr()

        cur_type = self._current()[0]
        if cur_type == 'RPAREN':
            raise ValueError("Mismatched parentheses")
        if cur_type != 'EOF':
            raise ValueError("Invalid expression: unexpected token")
        return result

    # ---------- lexer ----------
    def _tokenize(self, expr: str) -> List[Tuple[str, Union[str, None]]]:
        """Convert string to a list of tokens."""
        tokens: List[Tuple[str, Union[str, None]]] = []
        i = 0
        n = len(expr)
        while i < n:
            c = expr[i]
            if c.isspace():
                i += 1
                continue
            if c.isdigit() or c == '.':
                start = i
                dots = 0
                while i < n and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        dots += 1
                    i += 1
                num = expr[start:i]
                if dots > 1 or num == '.' or num == '':
                    raise ValueError(f"Invalid number '{num}'")
                try:
                    float(num)
                except ValueError:
                    raise ValueError(f"Invalid number '{num}'")
                tokens.append(('NUMBER', num))
                continue
            if c in '+-*/()':
                if c == '(':
                    tokens.append(('LPAREN', c))
                elif c == ')':
                    tokens.append(('RPAREN', c))
                else:
                    tokens.append(('OP', c))
                i += 1
                continue
            raise ValueError(f"Invalid character '{c}'")
        tokens.append(('EOF', None))
        return tokens

    # ---------- parser helpers ----------
    def _current(self) -> Tuple[str, Union[str, None]]:
        return self.tokens[self.pos]

    def _advance(self) -> None:
        self.pos += 1

    # grammar
    # expr   := term (( '+' | '-' ) term)*
    # term   := factor (( '*' | '/' ) factor)*
    # factor := ('+'|'-') factor | primary
    # primary:= number | '(' expr ')'

    def _parse_expr(self) -> float:
        """Parse addition/subtraction."""
        value = self._parse_term()
        while self._current()[0] == 'OP' and self._current()[1] in ('+', '-'):
            op = self._current()[1]
            self._advance()
            right = self._parse_term()
            value = value + right if op == '+' else value - right
        return value

    def _parse_term(self) -> float:
        """Parse multiplication/division."""
        value = self._parse_factor()
        while self._current()[0] == 'OP' and self._current()[1] in ('*', '/'):
            op = self._current()[1]
            self._advance()
            right = self._parse_factor()
            if op == '*':
                value *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                value /= right
        return value

    def _parse_factor(self) -> float:
        """Parse unary plus/minus."""
        typ, val = self._current()
        if typ == 'OP' and val == '-':
            self._advance()
            return -self._parse_factor()
        if typ == 'OP' and val == '+':
            self._advance()
            return +self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse number or parenthesized expression."""
        typ, val = self._current()
        if typ == 'NUMBER':
            self._advance()
            return float(val)
        if typ == 'LPAREN':
            self._advance()
            value = self._parse_expr()
            if self._current()[0] != 'RPAREN':
                raise ValueError("Mismatched parentheses")
            self._advance()
            return value
        if typ == 'RPAREN':
            raise ValueError("Mismatched parentheses")
        raise ValueError(f"Invalid token '{val}'")
```

```python
# test_expression_evaluator.py
import pytest
from expression_evaluator import ExpressionEvaluator

@pytest.fixture
def ev():
    return ExpressionEvaluator()

def test_basic_arithmetic(ev):
    assert ev.evaluate("2+3") == 5
    assert ev.evaluate("10-4") == 6
    assert ev.evaluate("3*4") == 12
    assert ev.evaluate("8/2") == 4

def test_precedence(ev):
    assert ev.evaluate("2+3*4") == 14
    assert ev.evaluate("2*3+4") == 10
    assert ev.evaluate("10-2*3") == 4

def test_parentheses(ev):
    assert ev.evaluate("(2+3)*4") == 20
    assert ev.evaluate("((1+2)*3)") == 9
    assert ev.evaluate("3 + (4 - 2) * 5") == 13

def test_unary_minus(ev):
    assert ev.evaluate("-3") == -3
    assert ev.evaluate("-(2+1)") == -3
    assert ev.evaluate("2 * -3") == -6
    assert ev.evaluate(" - ( -5 ) ") == 5

def test_error_cases(ev):
    with pytest.raises(ValueError, match="Empty expression"):
        ev.evaluate("   ")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        ev.evaluate("(1+2")
    with pytest.raises(ValueError, match="Division by zero"):
        ev.evaluate("1/0")
    with pytest.raises(ValueError):
        ev.evaluate("2 & 3")
    with pytest.raises(ValueError):
        ev.evaluate("2 + * 3")
```

The evaluator uses a recursive-descent parser, respects precedence, supports unary minus, floats and parentheses, and raises descriptive `ValueError`s for empty input, mismatched parentheses, division by zero and invalid tokens.