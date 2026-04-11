import pytest
from expression_evaluator import ExpressionEvaluator

@pytest.fixture
def evaluator():
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    """Tests basic addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("1 + 2") == 3.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("8 / 2") == 4.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # Precedence check
    assert evaluator.evaluate("10 / 2 + 3") == 8.0

def test_operator_precedence(evaluator):
    """Tests correct operator precedence (* and / before + and -)."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("1 + 2 * 3 - 4 / 2") == 1 + 6 - 2.0  # 5.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 6 + 20.0  # 26.0

def test_parentheses_grouping(evaluator):
    """Tests grouping with parentheses."""
    assert evaluator.evaluate("(1 + 2) * 3") == 9.0
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    assert evaluator.evaluate("((2 + 3) * 4) - 5") == 15.0
    assert evaluator.evaluate("(1 + (2 * 3))") == 7.0

def test_unary_minus(evaluator):
    """Tests unary minus support."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-2 + 5") == 3.0
    assert evaluator.evaluate("3 * -4") == -12.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("--5") == 5.0
    assert evaluator.evaluate("-(3 * -2)") == 6.0

def test_error_cases(evaluator):
    """Tests various error conditions."""
    # Empty expression
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("   ")

    # Mismatched parentheses
    with pytest.raises(ValueError):
        evaluator.evaluate("(1 + 2")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("1 + 2)")

    # Division by zero
    with pytest.raises(ValueError):
        evaluator.evaluate("1 / 0")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("5 / (2 - 2)")

    # Invalid tokens
    with pytest.raises(ValueError):
        evaluator.evaluate("1 + a")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("1 @ 2")

    # Unexpected tokens at end
    with pytest.raises(ValueError):
        evaluator.evaluate("1 + 2 +")
