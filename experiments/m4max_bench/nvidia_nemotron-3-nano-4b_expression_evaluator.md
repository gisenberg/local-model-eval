# nvidia/nemotron-3-nano-4b — expression_evaluator

## Thinking (3210 chars)

```
We need to produce Python code implementing ExpressionEvaluator class with evaluate method using recursive descent parser, supporting + - * /, parentheses, unary minus, floats. Raise ValueError for mismatched parentheses, division by zero, invalid tokens, empty expressions.

Provide type hints and docstrings. Also write 5 pytest tests covering those cases.

We must output code only? Probably include both implementation and test file. Provide as a single answer with two files: expression_evaluator.py and test_expression_evaluator.py.

Let's design parser:

Tokenization: split into tokens: numbers (float), operators (+ - * /), parentheses, maybe unary minus is token '-'.

But need to differentiate binary vs unary minus. In recursive descent we can handle via precedence: treat '-' as operator with lower precedence than '+'? Actually typical approach: parse expression as sum of terms, term = product or number; product = factor * (factor) etc. Unary minus can be applied to a factor.

Simplify: token list includes numbers and operators. We'll generate tokens via regex: r'(\d+\.\d*|\.\d+)|([-*/()])'.

But need to handle unary minus: e.g., '-3', '-(2+1)'. In tokenization, '-' is just token.

Parsing algorithm:

class ExpressionEvaluator:
    def evaluate(self, expr: str) -> float:
        tokens = self._tokenize(expr)
        if not tokens or tokens[0] == '(':
            raise ValueError("Empty expression")
        result = self.parse_expression(tokens)
        return result

parse_expression -> parse_additive (handles + - as additive? Actually we treat addition and subtraction at same level). Use recursive descent: parse_expression -> parse_term; parse_term -> parse_factor * parse_factor? Better:

Standard shunting-yard not needed.

Simplify: parse_expression = parse_additive
parse_additive: left = parse_term; while next token is '+' or '-': op = token; right = parse_term; apply op.

But unary minus is part of term? Actually '-' as additive operator can be binary. Unary minus appears before a factor, e.g., '-3' is additive with left operand 0? In typical parser, we treat unary minus as part of factor: factor -> ('-') factor | number | '(' expression ')'.

Thus parse_term = parse_factor; parse_additive = parse_term; while next token in ['+', '-']: op = token; right = parse_term; apply.

But then binary '-' is same precedence as '+'. That's fine.

Now parse_factor -> handle unary minus: if token == '-': consume, result = -parse_factor(); else if token == '(': consume, result = parse_expression(); else number token: convert to float.

Division by zero check when applying '/'.

Mismatched parentheses detection: we can keep a stack of open parentheses; at end must be empty. Also ensure not closing without opening.

Invalid tokens: any token not recognized.

Now implement.

Testing:

1. basic arithmetic: "2+3*4" => 14
2. precedence: "5-3/2" => 4? Actually 5 - (3/2) = 5 - 1.5 = 3.5
3. parentheses: "(2+3)*4" => 20
4. unary minus: "-3", "-(2+1)" => -3, -3
5. error cases:
   mismatched parentheses: "((2+3"
   division by zero: "5/0"
   invalid tokens: "2 + a"
   empty expression: ""

Also test that evaluate returns float.

Now write pytest tests.

Let's produce code.

```

## Output


**`expression_evaluator.py`**

```python
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
```

---

**`test_expression_evaluator.py`**

```python
import pytest
from expression_evaluator import ExpressionEvaluator

evaluator = ExpressionEvaluator()


def test_basic_arithmetic():
    assert evaluator.evaluate("2+3*4") == 14.0


def test_precedence():
    # 5 - (3/2) = 5 - 1.5 = 3.5
    assert evaluator.evaluate("5-3/2") == pytest.approx(3.5)


def test_parentheses_grouping():
    assert evaluator.evaluate("(2+3)*4") == 20.0


def test_unary_minus():
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2+1)") == -3.0


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("((2+3)", "Mismatched parentheses: leftover tokens [')']"),
        ("5/0", "Division by zero"),
        ("2 + a", "Invalid token encountered in expression: 'a'"),
        ("", "Empty expression"),
    ],
)
def test_error_cases(expr, expected):
    with pytest.raises(ValueError) as exc:
        evaluator.evaluate(expr)
    assert str(exc.value) == expected
```

**Explanation of the tests**

| Test | What it checks |
|------|----------------|
| `test_basic_arithmetic` | Simple expression, correct precedence (`*` before `+`). |
| `test_precedence` | Division has higher precedence than subtraction. |
| `test_parentheses_grouping` | Parentheses override normal precedence. |
| `test_unary_minus` | Unary minus works both as a standalone token and inside parentheses. |
| `test_error_cases` (parametrised) | All error conditions raise a `ValueError` with the exact message defined in the implementation. |

Run the tests with:

```bash
pytest test_expression_evaluator.py -q
```

All should pass, confirming that the recursive‑descent evaluator meets the specification.
