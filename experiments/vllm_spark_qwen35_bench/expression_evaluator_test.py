# test_expression_evaluator.py
import pytest
from expression_evaluator import ExpressionEvaluator

evaluator = ExpressionEvaluator()

def test_basic_arithmetic():
    """Test basic addition, subtraction, multiplication, and division."""
    assert evaluator.evaluate("2 + 2") == 4.0
    assert evaluator.evaluate("10 - 5") == 5.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("10 / 2") == 5.0

def test_operator_precedence():
    """Test that * and / bind tighter than + and -."""
    # 2 + 3 * 4 should be 2 + 12 = 14
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    # 10 - 2 * 3 should be 10 - 6 = 4
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    # 10 / 2 + 1 should be 5 + 1 = 6
    assert evaluator.evaluate("10 / 2 + 1") == 6.0

def test_parentheses():
    """Test grouping with parentheses overrides precedence."""
    # (2 + 3) * 4 should be 5 * 4 = 20
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    # 10 / (2 + 3) should be 10 / 5 = 2
    assert evaluator.evaluate("10 / (2 + 3)") == 2.0
    # Nested parentheses
    assert evaluator.evaluate("((1 + 2) * 3)") == 9.0

def test_unary_minus():
    """Test unary minus support at start and after operators."""
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-3.14") == -3.14
    # -(2 + 1)
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    # 3 - -5 (binary minus followed by unary minus)
    assert evaluator.evaluate("3 - -5") == 8.0
    # - ( - 5 )
    assert evaluator.evaluate("-(-5)") == 5.0

def test_error_cases():
    """Test that appropriate ValueErrors are raised."""
    # Empty expression
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    
    # Mismatched parentheses
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + 3)")
        
    # Division by zero
    with pytest.raises(ValueError):
        evaluator.evaluate("1 / 0")
        
    # Invalid tokens
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + a")
