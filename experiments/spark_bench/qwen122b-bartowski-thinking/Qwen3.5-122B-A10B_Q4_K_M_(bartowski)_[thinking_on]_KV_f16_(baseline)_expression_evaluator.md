# Qwen3.5-122B-A10B Q4_K_M (bartowski) [thinking on] — KV f16 (baseline) — Expression Evaluator

## Thinking (14445 chars)

Here's a thinking process that leads to the suggested solution:

1.  **Analyze the Request:**
    *   **Goal:** Build a mathematical expression evaluator in Python.
    *   **Class Name:** `ExpressionEvaluator`.
    *   **Method:** `evaluate(expr: str) -> float`.
    *   **Features:**
        *   Operators: `+`, `-`, `*`, `/`.
        *   Precedence: Standard (`*`, `/` > `+`, `-`).
        *   Grouping: Parentheses `()`.
        *   Unary Operators: Unary minus (e.g., `-3`, `-(2+1)`).
        *   Numbers: Floating point (e.g., `3.14`).
    *   **Error Handling:** Raise `ValueError` for mismatched parentheses, division by zero, invalid tokens, empty expressions.
    *   **Constraint:** Use a *recursive descent parser*. Do *not* use `eval()` or `ast.literal_eval()`.
    *   **Code Quality:** Type hints, docstrings.
    *   **Testing:** 5 `pytest` tests covering specific scenarios.

2.  **Design the Parser (Recursive Descent):**
    *   Recursive descent parsers work by defining functions for each level of grammar precedence.
    *   **Grammar:**
        *   `Expression` -> `Term` { (`+` | `-`) `Term` }
        *   `Term` -> `Factor` { (`*` | `/`) `Factor` }
        *   `Factor` -> `Number` | `(` `Expression` `)` | `UnaryOp` `Factor`
        *   `UnaryOp` -> `+` | `-`
        *   `Number` -> `Digit` { `Digit` | `.` `Digit` }
    *   **Tokenizer:** Before parsing, it's usually easier to tokenize the string into a list of tokens (numbers, operators, parentheses). This simplifies the parser logic significantly compared to character-by-character parsing.
    *   **State:** The parser needs to keep track of the current token index.

3.  **Drafting the Code Structure:**

    *   **Tokenizer:**
        *   Iterate through the string.
        *   Skip whitespace.
        *   Identify numbers (including decimals).
        *   Identify operators/parentheses.
        *   Raise `ValueError` on invalid characters.
    *   **Parser Class (`ExpressionEvaluator`):**
        *   `__init__`: Initialize tokens.
        *   `evaluate`: Entry point.
        *   `parse_expression`: Handles `+`, `-`.
        *   `parse_term`: Handles `*`, `/`.
        *   `parse_factor`: Handles numbers, parentheses, unary operators.
        *   Helper methods: `peek`, `consume`, `is_at_end`.

4.  **Handling Specific Requirements:**

    *   **Unary Minus:** This is tricky in standard precedence.
        *   Option A: Treat `-` as a binary operator with lower precedence than unary.
        *   Option B: Handle unary minus inside `parse_factor`.
        *   *Decision:* Option B is standard for recursive descent. If we see a `-` or `+` at the start of a factor, consume it and parse the next factor, applying the negation.
        *   *Edge Case:* `3 * -4`. The tokenizer needs to distinguish binary minus from unary minus, OR the parser needs to handle it.
        *   *Refinement:* It's often easier to let the tokenizer produce a single `-` token and let the parser decide if it's unary or binary based on context.
        *   *Context:* If the previous token was an operator or opening parenthesis, the current `-` is unary. If the previous token was a number or closing parenthesis, it's binary.
        *   *Alternative:* Tokenizer can distinguish `UNARY_MINUS` vs `BINARY_MINUS`.
        *   *Simpler Approach:* In `parse_factor`, check if the current token is `+` or `-`. If so, consume it, parse the next factor, and apply the sign. This handles `-(2+1)` and `-3`. But what about `3 * -4`?
        *   *Correction:* In `parse_term`, after consuming a factor, if the next token is `*` or `/`, continue. If the next token is `+` or `-`, stop.
        *   *Wait, standard grammar:*
            `Expression` -> `Term` { (`+` | `-`) `Term` }
            `Term` -> `Factor` { (`*` | `/`) `Factor` }
            `Factor` -> `Number` | `(` `Expression` `)` | `(` `+` | `-` `)` `Factor`
        *   Actually, usually unary operators bind tighter than multiplication. `-3 * 4` is `(-3) * 4`.
        *   So `Factor` should handle unary operators.
        *   Logic in `parse_factor`:
            1. Check for `+` or `-`. If found, consume, parse `Factor`, apply sign.
            2. Check for `(`. If found, consume, parse `Expression`, expect `)`.
            3. Check for Number.
            4. Else error.
        *   *Wait, ambiguity:* `3 - -4`.
            *   `parse_expression` calls `parse_term`.
            *   `parse_term` calls `parse_factor` (gets 3).
            *   `parse_term` sees `-`. It's binary. Consumes `-`. Calls `parse_term`.
            *   `parse_term` calls `parse_factor`.
            *   `parse_factor` sees `-`. It's unary. Consumes `-`. Calls `parse_factor` (gets 4). Returns -4.
            *   `parse_term` returns -4.
            *   `parse_expression` returns 3 - (-4) = 7.
            *   This works perfectly.

    *   **Division by Zero:** Check before dividing.
    *   **Mismatched Parentheses:** Check during tokenization or parsing. Parsing is safer for structure.
    *   **Empty Expression:** Check at start.
    *   **Invalid Tokens:** Check during tokenization.

5.  **Refining the Tokenizer:**
    *   Need to handle floats like `3.`, `.5`, `3.14`.
    *   Regex is handy for tokenizing numbers, but manual parsing is often clearer for educational/strict requirements. Let's use manual parsing for robustness.
    *   Tokens: `NUMBER`, `PLUS`, `MINUS`, `MUL`, `DIV`, `LPAREN`, `RPAREN`, `EOF`.

6.  **Writing the Implementation (Mental Draft):**

    ```python
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
    ```

    *Wait, one edge case:* `3 * - 4`.
    *   `parse_term` gets 3.
    *   Sees `*`. Consumes. Calls `parse_factor`.
    *   `parse_factor` sees `-`. Consumes. Calls `parse_factor`.
    *   `parse_factor` sees 4. Returns 4.
    *   `parse_factor` returns -4.
    *   `parse_term` returns 3 * -4 = -12.
    *   This logic holds.

    *Wait, another edge case:* `-(2+1)`.
    *   `parse_factor` sees `-`. Consumes. Calls `parse_factor`.
    *   `parse_factor` sees `(`. Consumes. Calls `parse_expression`.
    *   `parse_expression` returns 3.
    *   `parse_factor` expects `)`. Consumes. Returns 3.
    *   Outer `parse_factor` returns -3.
    *   Logic holds.

    *Wait, Tokenizer details:*
    *   Need to handle `.` correctly. `3.` is valid float? Usually yes. `.5` is valid? Usually yes.
    *   Need to handle `.` not followed by digits (e.g., `3. + 1`).
    *   Let's ensure the tokenizer captures full numbers.

7.  **Writing the Tests:**
    *   Test 1: Basic arithmetic (`2 + 3 * 4`).
    *   Test 2: Precedence (`10 - 2 * 3`).
    *   Test 3: Parentheses (`(2 + 3) * 4`).
    *   Test 4: Unary minus (`-5`, `-(2+3)`).
    *   Test 5: Errors (Div by zero, mismatched parens, invalid char).

8.  **Refining Code Quality:**
    *   Add Type Hints (`List`, `Optional`, `Union`, etc.).
    *   Add Docstrings.
    *   Ensure `ValueError` messages are descriptive.
    *   Ensure `evaluate` returns `float`.

9.  **Self-Correction/Review during drafting:**
    *   *Issue:* Tokenizer might produce `.` as a separate token if not careful.
    *   *Fix:* In tokenizer, if current char is digit or `.` (and next is digit), accumulate.
    *   *Issue:* `pos` management.
    *   *Fix:* Use a helper `_current()` and `_advance()`.
    *   *Issue:* Empty string.
    *   *Fix:* Check `len(tokens) == 0` in `evaluate`.
    *   *Issue:* Trailing garbage (e.g., `1 + 2 a`).
    *   *Fix:* After `parse_expression` returns, check if `pos == len(tokens)`.

10. **Final Code Assembly:** (Proceeding to generate the Python code based on these thoughts). I will structure it with the Tokenizer logic inside the class or as a helper to keep it self-contained. A nested class or helper function for tokens is cleaner. Let's use a simple list of tuples for tokens to avoid defining a full `Token` class overhead, or a `NamedTuple`. `NamedTuple` is cleaner.

    *Wait, one specific constraint:* "Raise ValueError... for mismatched parentheses".
    *   My logic checks for `)` after `(`. What about `(` without `)`? The parser will hit EOF expecting `)`. I need to handle `EOF` in `_parse_factor`.

    *Wait, Unary Minus Precedence:*
    *   Does `-3^2` mean `(-3)^2` or `-(3^2)`? Python does `-(3^2)`.
    *   My grammar: `Factor` handles unary. `Term` handles `* /`. `Expression` handles `+ -`.
    *   So `-3 * 4` -> `(-3) * 4`.
    *   Is this correct? In Python: `-3 * 4` is `-12`. `(-3) * 4` is `-12`.
    *   What about `-3 ** 2`? Python: `-9`. `(-3) ** 2` is `9`.
    *   Since I don't support `**`, this distinction is less critical, but standard math convention usually treats unary minus as having higher precedence than multiplication.
    *   My grammar: `Factor` (unary) -> `Term` (mul) -> `Expression` (add).
    *   So `-3 * 4` parses as `Factor(-3)` then `Term` multiplies by 4. Result `-12`.
    *   This matches Python behavior for `*`.

    *Wait, Tokenizer Logic:*
    *   Need to handle `.` carefully. `1.2.3` is invalid.
    *   Need to handle `.` at start `.5`.
    *   Need to handle `.` at end `5.`.
    *   Regex `r'\d+\.?\d*|\.\d+'` covers most, but manual is safer for error reporting.
    *   Let's stick to manual accumulation for numbers.

    *Wait, Error Messages:*
    *   "Mismatched parentheses": Check if `)` is missing or extra.
    *   "Division by zero": Check before `/`.
    *   "Invalid tokens": Check during tokenization.
    *   "Empty expressions": Check at start.

    *Wait, Type Hints:*
    *   `List[Token]`, `float`, `str`.

    *Wait, Tests:*
    *   Use `pytest`.
    *   `pytest.raises(ValueError)`.

    Let's write the code.

    *(Self-Correction on Tokenizer)*:
    If I use a simple list of `(type, value)` tuples, it's easy.
    Types: `NUM`, `OP`, `LPAREN`, `RPAREN`.
    Actually, distinguishing `+` and `-` is needed for the parser logic.
    Let's use specific types: `PLUS`, `MINUS`, `MUL`, `DIV`, `LPAREN`, `RPAREN`, `NUMBER`.

    *(Self-Correction on Unary Minus)*:
    If I have `3 - - 4`.
    Tokenizer: `3`, `-`, `-`, `4`.
    Parser:
    `parse_expr` -> `parse_term` (3).
    Sees `-`. Consumes. Calls `parse_term`.
    `parse_term` -> `parse_factor`.
    `parse_factor` sees `-`. Consumes. Calls `parse_factor`.
    `parse_factor` sees `4`. Returns 4.
    `parse_factor` returns -4.
    `parse_term` returns -4.
    `parse_expr` returns 3 - (-4) = 7.
    Correct.

    *(Self-Correction on Tokenizer)*:
    What about `3.14`?
    Tokenizer needs to read `3`, `.`, `1`, `4` as one number.
    Logic:
    If digit: read all digits. If next is `.` and next-next is digit, read `.` and digits.
    If `.`: read digits.
    This handles `.5` and `3.` and `3.5`.
    Wait, `3.` is valid float in Python (`3.0`).
    `3.14` is valid.
    `.5` is valid.
    `3..4` is invalid.
    I will implement a robust number parser.

    *(Final check on constraints)*:
    - Class `ExpressionEvaluator`.
    - Method `evaluate`.
    - No `eval`.
    - Recursive Descent.
    - Type hints.
    - Docstrings.
    - 5 Tests.

    Let's assemble.

    *(Wait, one detail on Tokenizer)*:
    If I encounter a character that isn't whitespace, digit, dot, or operator, raise ValueError immediately.

    *(Wait, one detail on Parser)*:
    If `parse_expression` finishes but there are tokens left (e.g. `1 + 2 3`), raise ValueError.

    *(Wait, one detail on Division)*:
    Floating point division. `5 / 2` -> `2.5`.

    *(Wait, one detail on Unary)*:
    `-(2+1)`.
    `parse_factor` sees `-`. Consumes. Calls `parse_factor`.
    `parse_factor` sees `(`. Consumes. Calls `parse_expression`.
    `parse_expression` returns 3.
    `parse_factor` expects `)`. Consumes. Returns 3.
    Outer `parse_factor` returns -3.
    Correct.

    *(Wait, one detail on Empty)*:
    `""` -> `tokens` is empty. Raise ValueError.
    `"   "` -> `tokens` is empty. Raise ValueError.

    Okay, ready to code.

    *(Wait, Tokenizer Implementation)*:
    I'll put the tokenizer inside `evaluate` or `__init__`. `__init__` is better for separation of concerns, but `evaluate` is the public API. I'll put tokenization in `evaluate` or a private `_tokenize` called by `evaluate`. Let's put it in `evaluate` to keep state clean per call, or `__init__` if the class is meant to be instantiated once per expression. The prompt says `evaluate(expr: str)`. So instantiation happens, then evaluate. I will tokenize in `evaluate` to ensure fresh state.

    Actually, standard pattern for this class structure:
    `evaluator = ExpressionEvaluator()`
    `evaluator.evaluate("1+1")`
    So `evaluate` should handle the setup.

    Let's refine the Tokenizer to be robust.
    `isdigit`, `isalpha` (for error), `isspace`.

    Okay, plan confirmed.


## Output

```python
import re
import pytest
from typing import List, Tuple, Union, Optional

# Define token types as constants for clarity
TOKEN_NUMBER = "NUMBER"
TOKEN_PLUS = "PLUS"
TOKEN_MINUS = "MINUS"
TOKEN_MUL = "MUL"
TOKEN_DIV = "DIV"
TOKEN_LPAREN = "LPAREN"
TOKEN_RPAREN = "RPAREN"
TOKEN_EOF = "EOF"

Token = Tuple[str, Union[float, str]]


class ExpressionEvaluator:
    """
    A recursive descent parser for evaluating mathematical expressions.
    
    Supports +, -, *, /, parentheses, unary minus, and floating point numbers.
    Does not use eval() or ast.literal_eval().
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.tokens: List[Token] = []
        self.pos: int = 0

    def evaluate(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string.
        
        Args:
            expr: The mathematical expression string to evaluate.
            
        Returns:
            The result of the evaluation as a float.
            
        Raises:
            ValueError: If the expression is empty, contains invalid tokens,
                        has mismatched parentheses, or division by zero occurs.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty expression")

        self.tokens = self._tokenize(expr)
        self.pos = 0

        if not self.tokens:
            raise ValueError("Empty expression")

        result = self._parse_expression()

        # Ensure all tokens were consumed
        if self.pos < len(self.tokens):
            raise ValueError(f"Invalid syntax: unexpected token '{self._peek()}'")

        return result

    def _tokenize(self, expr: str) -> List[Token]:
        """
        Convert the input string into a list of tokens.
        
        Args:
            expr: The raw expression string.
            
        Returns:
            A list of (type, value) tuples.
            
        Raises:
            ValueError: If an invalid character is encountered.
        """
        tokens: List[Token] = []
        i = 0
        length = len(expr)

        while i < length:
            char = expr[i]

            if char.isspace():
                i += 1
                continue

            if char.isdigit() or char == '.':
                # Parse number
                start = i
                has_dot = False
                while i < length and (expr[i].isdigit() or expr[i] == '.'):
                    if expr[i] == '.':
                        if has_dot:
                            raise ValueError(f"Invalid number format at index {i}")
                        has_dot = True
                    i += 1
                
                num_str = expr[start:i]
                # Validate number structure (e.g., prevent just "." or "..")
                if num_str == '.' or num_str.startswith('.') and len(num_str) == 1:
                     raise ValueError(f"Invalid number format at index {start}")
                
                try:
                    value = float(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number '{num_str}' at index {start}")
                
                tokens.append((TOKEN_NUMBER, value))
                continue

            if char == '+':
                tokens.append((TOKEN_PLUS, '+'))
            elif char == '-':
                tokens.append((TOKEN_MINUS, '-'))
            elif char == '*':
                tokens.append((TOKEN_MUL, '*'))
            elif char == '/':
                tokens.append((TOKEN_DIV, '/'))
            elif char == '(':
                tokens.append((TOKEN_LPAREN, '('))
            elif char == ')':
                tokens.append((TOKEN_RPAREN, ')'))
            else:
                raise ValueError(f"Invalid token '{char}' at index {i}")

            i += 1

        tokens.append((TOKEN_EOF, None))
        return tokens

    def _peek(self) -> Token:
        """
        Return the current token without advancing.
        
        Returns:
            The current token tuple.
        """
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (TOKEN_EOF, None)

    def _consume(self) -> Token:
        """
        Return the current token and advance the position.
        
        Returns:
            The current token tuple.
        """
        token = self._peek()
        self.pos += 1
        return token

    def _parse_expression(self) -> float:
        """
        Parse an expression handling addition and subtraction.
        Precedence: Lowest (+, -)
        
        Grammar: Expression -> Term { (+|-) Term }
        
        Returns:
            The calculated float value.
        """
        value = self._parse_term()

        while True:
            token_type, _ = self._peek()
            if token_type == TOKEN_PLUS:
                self._consume()
                value += self._parse_term()
            elif token_type == TOKEN_MINUS:
                self._consume()
                value -= self._parse_term()
            else:
                break
        return value

    def _parse_term(self) -> float:
        """
        Parse a term handling multiplication and division.
        Precedence: Medium (*, /)
        
        Grammar: Term -> Factor { (*|/) Factor }
        
        Returns:
            The calculated float value.
            
        Raises:
            ValueError: If division by zero occurs.
        """
        value = self._parse_factor()

        while True:
            token_type, _ = self._peek()
            if token_type == TOKEN_MUL:
                self._consume()
                value *= self._parse_factor()
            elif token_type == TOKEN_DIV:
                self._consume()
                divisor = self._parse_factor()
                if divisor == 0:
                    raise ValueError("Division by zero")
                value /= divisor
            else:
                break
        return value

    def _parse_factor(self) -> float:
        """
        Parse a factor handling numbers, parentheses, and unary operators.
        Precedence: Highest (unary -, parentheses)
        
        Grammar: Factor -> Number | ( Expression ) | (+|-) Factor
        
        Returns:
            The calculated float value.
            
        Raises:
            ValueError: If parentheses are mismatched or token is invalid.
        """
        token_type, token_value = self._peek()

        # Handle Unary Plus/Minus
        if token_type in (TOKEN_PLUS, TOKEN_MINUS):
            self._consume()
            value = self._parse_factor()
            return -value if token_type == TOKEN_MINUS else value

        # Handle Parentheses
        if token_type == TOKEN_LPAREN:
            self._consume()
            value = self._parse_expression()
            if self._peek()[0] != TOKEN_RPAREN:
                raise ValueError("Mismatched parentheses: expected ')'")
            self._consume()
            return value

        # Handle Numbers
        if token_type == TOKEN_NUMBER:
            self._consume()
            return float(token_value)

        # Handle End of Input or Unexpected Token
        if token_type == TOKEN_EOF:
            raise ValueError("Unexpected end of expression")
        
        raise ValueError(f"Invalid token '{token_value}'")


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------

def test_basic_arithmetic():
    """Test basic addition, subtraction, multiplication, and division."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 2") == 5.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # Precedence check

def test_precedence_and_grouping():
    """Test operator precedence and parentheses grouping."""
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    # Parentheses override precedence
    assert evaluator.evaluate("(1 + 2) * 3") == 9.0
    # Nested parentheses
    assert evaluator.evaluate("((2 + 3) * 4) / 5") == 4.0
    # Complex precedence
    assert evaluator.evaluate("10 - 2 * 3 + 4") == 8.0

def test_unary_minus():
    """Test unary minus operator support."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-3.14") == -3.14
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 * -4") == -12.0
    assert evaluator.evaluate("3 - -4") == 7.0
    assert evaluator.evaluate("--5") == 5.0

def test_floating_point():
    """Test floating point number parsing."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate(".5 + .5") == 1.0
    assert evaluator.evaluate("10. / 2.") == 5.0
    assert evaluator.evaluate("1.5 + 2.5") == 4.0

def test_error_cases():
    """Test error handling for invalid inputs."""
    evaluator = ExpressionEvaluator()
    
    # Empty expression
    with pytest.raises(ValueError, match="Empty"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty"):
        evaluator.evaluate("   ")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched"):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError, match="Mismatched"):
        evaluator.evaluate("2 + 3)")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / (2 - 2)")

    # Invalid tokens
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 + a")
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")
    
    # Syntax error (trailing garbage)
    with pytest.raises(ValueError, match="Invalid syntax"):
        evaluator.evaluate("2 + 2 3")
```

### Explanation of Implementation

1.  **Tokenizer (`_tokenize`)**:
    *   Iterates through the input string character by character.
    *   Skips whitespace.
    *   Accumulates digits and dots into `NUMBER` tokens, validating that dots don't appear consecutively or in invalid positions.
    *   Maps operators and parentheses to specific token types.
    *   Raises `ValueError` immediately for invalid characters.

2.  **Recursive Descent Parser**:
    *   **`_parse_expression`**: Handles the lowest precedence operators (`+`, `-`). It calls `_parse_term` and loops while it sees `+` or `-`.
    *   **`_parse_term`**: Handles medium precedence operators (`*`, `/`). It calls `_parse_factor` and loops while it sees `*` or `/`. It explicitly checks for division by zero.
    *   **`_parse_factor`**: Handles the highest precedence elements.
        *   Checks for unary `+` or `-`. If found, it consumes the operator and recursively calls `_parse_factor` (allowing for `--5` or `-(2+1)`).
        *   Checks for `(`. If found, it consumes, calls `_parse_expression`, and expects a closing `)`.
        *   Checks for `NUMBER`.
        *   Raises errors for unexpected tokens or mismatched parentheses.

3.  **Error Handling**:
    *   **Empty Expression**: Checked at the start of `evaluate`.
    *   **Mismatched Parentheses**: Checked in `_parse_factor` when expecting `)` and in `_parse_expression` if EOF is reached unexpectedly.
    *   **Division by Zero**: Checked in `_parse_term` before performing division.
    *   **Invalid Tokens**: Checked in `_tokenize` and `_parse_factor`.
    *   **Trailing Garbage**: Checked in `evaluate` after parsing completes to ensure all tokens were consumed.

4.  **Testing**:
    *   Uses `pytest` to verify functionality.
    *   Covers basic arithmetic, precedence rules, unary minus scenarios, floating point numbers, and specific error conditions as requested.

5.  **Constraints**:
    *   No `eval()` or `ast` used.
    *   Type hints included.
    *   Docstrings included for all methods.
    *   Class named `ExpressionEvaluator` with `evaluate` method.
