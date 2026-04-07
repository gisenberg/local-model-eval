# Benchmark: Expression Evaluator

**Difficulty:** Medium
**Expected tests:** 5
**Skills tested:** Recursive descent parsing, operator precedence, unary operators, error handling, test writing

## Prompt

```
Build a mathematical expression evaluator in Python. Requirements:
1. Support +, -, *, / with correct operator precedence
2. Support parentheses for grouping
3. Support unary minus (e.g., '-3', '-(2+1)')
4. Support floating point numbers (e.g., '3.14')
5. Raise ValueError with a descriptive message for: mismatched parentheses, division by zero, invalid tokens, empty expressions
6. Implement as a class called ExpressionEvaluator with an evaluate(expr: str) -> float method
7. Use a recursive descent parser — do NOT use eval() or ast.literal_eval()
8. Include type hints throughout and a brief docstring on each method
9. Write 5 pytest tests covering: basic arithmetic, precedence, parentheses, unary minus, and error cases
```

## What Makes This a Good Benchmark

- Requires understanding of formal grammars and recursive descent
- Tests proper operator precedence (not just left-to-right evaluation)
- Unary minus is a common source of parser bugs
- Error handling requires thinking through multiple failure modes
- Test writing requires the model to reason about its own implementation's error messages

## Common Failure Modes Observed

1. **Regex errors** (Nemotron 4B): Using `(?P<plus>+)` where `+` is a regex quantifier
2. **Test import bugs** (Nemotron 30B): `from module import ValueError` (builtin, not exported)
3. **Test regex mismatches** (Gemma Q4): Test `match=` string doesn't align with actual error message
4. **Thinking budget exhaustion** (Qwen models at low context): Chain-of-thought consumes all tokens before code output

## Evaluation Criteria

- Does the code import without errors?
- Do all 5 tests pass when run with `pytest -v`?
- No manual fixes allowed — code must work exactly as generated
