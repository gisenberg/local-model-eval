# Benchmark: Expression Evaluator (Implementation Only)

**Difficulty:** Medium
**Expected tests:** 5 (provided, not model-generated)
**Skills tested:** Recursive descent parsing, operator precedence — isolated from test-writing ability

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

Do NOT write tests — only the implementation.
```

## Provided Test Suite

These tests are appended by the harness after the model's implementation:

```python
import pytest

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("15 / 4") == 3.75

def test_precedence(evaluator):
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0

def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((1 + 2) * (3 + 4))") == 21.0

def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_errors(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3")
```

## Why This Variant Matters

The original benchmark requires models to write BOTH implementation and tests. When tests fail, it's often because the test assertions don't match the implementation (e.g., error message wording mismatch), not because the implementation is wrong.

This variant isolates implementation quality by providing known-good tests. A model that scores higher on this variant than the original has good code but weak test-writing.
