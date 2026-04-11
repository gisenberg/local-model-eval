from __future__ import annotations
from typing import List, Optional


class ExpressionEvaluator:
    """
    Recursive‑descent parser for arithmetic expressions.

    Supports:
        * binary operators +, -, *, / with correct precedence
        * parentheses for grouping
        * unary plus/minus (e.g. "-3", "-(2+1)")
        * floating point numbers (e.g. "3.14")
    Raises ValueError with a descriptive message for:
        * mismatched parentheses
        * division by zero
        * invalid tokens / characters
        * empty expression
    """

    def evaluate(self, expr: str) -> float:
        """
        Parse and evaluate *expr*.

        Parameters
        ----------
        expr: str
            The arithmetic expression to evaluate.

        Returns
        -------
        float
            The result of the expression.

        Raises
        ------
        ValueError
            If the expression is empty, contains invalid tokens,
            has mismatched parentheses, or attempts division by zero.
        """
        self._tokens: List[str] = self._tokenize(expr)
        if not self._tokens:
            raise ValueError("Empty expression")

        self._pos: int = 0
        result = self._parse_expr()
        if self._pos != len(self._tokens):
            # Something left over → unexpected token
            raise ValueError(f"Unexpected token: {self._tokens[self._pos]}")
        return result

    # -----------------------------------------------------------------
    # Tokeniser
    # -----------------------------------------------------------------
    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert the input string into a list of tokens.

        Tokens are numbers (as strings), '+', '-', '*', '/', '(', ')'.

        Raises
        ------
        ValueError
            If an invalid character is encountered or a number contains
            more than one decimal point.
        """
        tokens: List[str] = []
        i = 0
        n = len(expr)

        while i < n:
            ch = expr[i]

            if ch.isspace():
                i += 1
                continue

            if ch in "+-*/()":
                tokens.append(ch)
                i += 1
                continue

            if ch.isdigit() or ch == ".":
                start = i
                dot_seen = False
                while i < n and (expr[i].isdigit() or expr[i] == "."):
                    if expr[i] == ".":
                        if dot_seen:
                            raise ValueError("Invalid number format")
                        dot_seen = True
                    i += 1
                tokens.append(expr[start:i])
                continue

            # Anything else is illegal
            raise ValueError(f"Invalid character: {ch}")

        return tokens

    # -----------------------------------------------------------------
    # Parser helpers
    # -----------------------------------------------------------------
    def _peek(self) -> Optional[str]:
        """Return the current token without consuming it, or None at EOF."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self) -> str:
        """Consume and return the current token; raise if at EOF."""
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        self._pos += 1
        return tok

    # -----------------------------------------------------------------
    # Grammar:
    #   expr   ::= term (('+' | '-') term)*
    #   term   ::= factor (('*' | '/') factor)*
    #   factor ::= ('+' | '-') factor | number | '(' expr ')'
    # -----------------------------------------------------------------
    def _parse_expr(self) -> float:
        """Parse addition and subtraction."""
        value = self._parse_term()
        while self._peek() in ("+", "-"):
            op = self._consume()
            right = self._parse_term()
            if op == "+":
                value += right
            else:  # op == "-"
                value -= right
        return value

    def _parse_term(self) -> float:
        """Parse multiplication and division."""
        value = self._parse_factor()
        while self._peek() in ("*", "/"):
            op = self._consume()
            right = self._parse_factor()
            if op == "*":
                value *= right
            else:  # op == "/"
                if right == 0.0:
                    raise ValueError("Division by zero")
                value /= right
        return value

    def _parse_factor(self) -> float:
        """Parse a factor: unary +/- , number, or parenthesised expression."""
        token = self._peek()

        # Unary plus/minus
        if token == "+":
            self._consume()
            return self._parse_factor()
        if token == "-":
            self._consume()
            return -self._parse_factor()

        # Parenthesised sub‑expression
        if token == "(":
            self._consume()  # consume '('
            inner = self._parse_expr()
            if self._peek() != ")":
                raise ValueError("Mismatched parentheses")
            self._consume()  # consume ')'
            return inner

        # Number literal
        if token is None:
            raise ValueError("Unexpected end of expression")
        try:
            num = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid token: {token}") from exc
        self._consume()
        return num
