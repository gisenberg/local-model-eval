    class Token:
        type: str
        value: Any

    class ExpressionEvaluator:
        def __init__(self, expr):
            self.tokens = self._tokenize(expr)
            self.pos = 0

        def evaluate(self):
            if not self.tokens: raise ValueError("Empty")
            result = self._parse_expression()
            if self.pos < len(self.tokens): raise ValueError("Invalid syntax")
            return result

        def _parse_expression(self):
            val = self._parse_term()
            while self._peek() in ('+', '-'):
                op = self._consume()
                right = self._parse_term()
                if op == '+': val += right
                else: val -= right
            return val

        def _parse_term(self):
            val = self._parse_factor()
            while self._peek() in ('*', '/'):
                op = self._consume()
                right = self._parse_factor()
                if op == '*': val *= right
                else:
                    if right == 0: raise ValueError("Div by zero")
                    val /= right
            return val

        def _parse_factor(self):
            token = self._peek()
            if token in ('+', '-'):
                op = self._consume()
                val = self._parse_factor() # Recursive for unary
                return -val if op == '-' else val
            elif token == '(':
                self._consume()
                val = self._parse_expression()
                if self._peek() != ')': raise ValueError("Mismatched parens")
                self._consume()
                return val
            elif token == 'NUMBER':
                return self._consume()
            else:
                raise ValueError("Invalid token")
    