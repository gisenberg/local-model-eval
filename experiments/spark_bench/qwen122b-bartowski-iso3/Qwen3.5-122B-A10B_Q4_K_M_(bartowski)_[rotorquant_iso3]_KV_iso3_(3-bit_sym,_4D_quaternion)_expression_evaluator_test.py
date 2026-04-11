import pytest
from expression_evaluator import ExpressionEvaluator

evaluator = ExpressionEvaluator()

def test_basic_arithmetic():
    """Tests basic addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("8 / 2") == 4.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0 # Precedence check

def test_operator_precedence():
    """Tests that multiplication/division happens before addition/subtraction."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("1 + 2 * 3 - 4 / 2") == 5.0 # 1 + 6 - 2 = 5
    assert evaluator.evaluate("10 / 2 + 3 * 2") == 11.0 # 5 + 6 = 11

def test_parentheses():
    """Tests grouping with parentheses."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3) * (4 - 1))") == 15.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0

def test_unary_minus():
    """Tests unary minus support."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-2 + 5") == 3.0
    assert evaluator.evaluate("3 * -4") == -12.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("-(3 * 4)") == -12.0
    assert evaluator.evaluate("--5") == 5.0 # Double negative
    assert evaluator.evaluate("1 - -2") == 3.0

def test_error_cases():
    """Tests various error conditions."""
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

    # Division by zero
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / 0")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("1 / (2 - 2)")

    # Invalid tokens
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + a")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("2 @ 3")

    # Floating point support
    assert evaluator.evaluate("3.14 * 2") == 6.28
    assert evaluator.evaluate("10 / 3") == pytest.approx(3.3333333333333335)
