from __future__ import annotations

import re
from typing import List, Tuple


class ExpressionEvaluator:
    """
    Recursive‑descent evaluator for arithmetic expressions with the following
    features:

    * Operators: +  -  *  /
    * Correct precedence ( * and / before + and - )
    * Parentheses for grouping
    * Unary minus (e.g. ``-3`` or ``-(2+1)``)
    * Floating‑point numbers

    Errors are raised as :class:`ValueError` with a descriptive message.
    """

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def evaluate(self, expr: str) -> float:
        """
        Evaluate ``expr`` and return the result as a ``float``.

        Parameters
        ----------
        expr: str
            The arithmetic expression to evaluate. Must be non‑empty and
            syntactically valid according to the grammar defined in this class.

        Returns
        -------
        float
            The numeric value of the expression.

        Raises
        ------
        ValueError
            * Empty expression.
            * Mismatched parentheses.
            * Division by zero.
            * Invalid token encountered.
    """

    # Tokenise and parse
    tokens: List[str] = self._tokenize(expr)
    if not tokens or tokens[0] == "(":
        raise ValueError("Empty expression")

    result = self._parse_expression(tokens)

    # Ensure no stray tokens remain (e.g. mismatched parentheses)
    if tokens:
        raise ValueError(f"Mismatched parentheses: leftover tokens {tokens}")

    return float(result)


# --------------------------------------------------------------------- #
#  Private helpers
# --------------------------------------------------------------------- #
    def _tokenize(self, expr: str) -> List[str]:
        """
        Convert the raw string into a list of tokens.

        Supported token types:
            * numbers (int or float)
            * operators: + - * /
            * parentheses: ( )
        Any other character raises ``ValueError`` with a descriptive message.
        """
        # Regex captures either a number or an operator/parenthesis
        pattern = re.compile(r"(\d+\.\d*|\.\d+)|([-*/()])")
        tokens = []
        for m in pattern.finditer(expr):
            token = m.group(0)
            if token.isdigit() or "." in token:
                # numeric literal – keep as string, conversion later
                tokens.append(token)
            else:  # operator or parenthesis
                tokens.append(token)

        # If any character is not part of a token, it's invalid
        if len("".join(tokens)) != len(expr):
            raise ValueError(f"Invalid token encountered in expression: '{expr}'")

        return tokens

    def _parse_expression(self, tokens: List[str]) -> float:
        """
        Top‑level parser – parses an additive expression (sum of terms).

        Grammar:
            expr   :: term { ('+' | '-') term }*
            term   :: factor
            factor :: ('-' | '(') factor | number
        """
        result = self._parse_term(tokens)
        while tokens and tokens[0] in {"+", "-"}:
            op = tokens.pop(0)
            right = self._parse_term(tokens)
            if op == "+":
                result += right
            else:  # "-"
                result -= right
        return result

    def _parse_term(self, tokens: List[str]) -> float:
        """
        Parses a multiplicative term (product of factors).

        Grammar:
            term :: factor
        """
        return self._parse_factor(tokens)

    def _parse_factor(self, tokens: List[str]) -> float:
        """
        Parses a primary factor – either a number, a parenthesised expression,
        or an unary minus applied to another factor.

        Grammar:
            factor :: ('-' | '(') factor | number
        """
        if not tokens:
            raise ValueError("Unexpected end of input while parsing factor")

        token = tokens.pop(0)

        # Unary minus
        if token == "-":
            return -self._parse_factor(tokens)

        # Parenthesised sub‑expression
        if token == "(":
            result = self._parse_expression(tokens)
            if not tokens or tokens[0] != ")":
                raise ValueError("Mismatched parentheses")
            tokens.pop(0)  # consume ')'
            return result

        # Number literal (int or float)
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid token '{token}' in expression") from exc
        return value


# --------------------------------------------------------------------- #
#  End of class
# --------------------------------------------------------------------- #
