import pytest
from expression_evaluator import ExpressionEvaluator

def test_basic_arithmetic():
    """Test basic addition, subtraction, multiplication, and division."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("8 / 2") == 4.0
    assert evaluator.evaluate("2 + 3 * 4") == 14.0  # Precedence check

def test_precedence_and_grouping():
    """Test operator precedence and parentheses grouping."""
    evaluator = ExpressionEvaluator()
    # Multiplication before addition
    assert evaluator.evaluate("1 + 2 * 3") == 7.0
    # Parentheses override precedence
    assert evaluator.evaluate("(1 + 2) * 3") == 9.0
    # Nested parentheses
    assert evaluator.evaluate("((2 + 3) * 4) - 5") == 15.0
    # Floating point numbers
    assert abs(evaluator.evaluate("3.14 * 2") - 6.28) < 0.0001

def test_unary_minus():
    """Test unary minus operator support."""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("3 + -2") == 1.0
    assert evaluator.evaluate("-(2 + 3)") == -5.0
    assert evaluator.evaluate("--3") == 3.0
    assert evaluator.evaluate("2 * -3") == -6.0

def test_error_mismatched_parentheses():
    """Test ValueError for mismatched parentheses."""
    evaluator = ExpressionEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate("(1 + 2")
    with pytest.raises(ValueError):
        evaluator.evaluate("1 + 2)")
    with pytest.raises(ValueError):
        evaluator.evaluate("1 + 2 ) 3")

def test_error_invalid_and_division():
    """Test ValueError for invalid tokens and division by zero."""
    evaluator = ExpressionEvaluator()
    
    # Division by zero
    with pytest.raises(ValueError):
        evaluator.evaluate("1 / 0")
    
    # Invalid characters
    with pytest.raises(ValueError):
        evaluator.evaluate("1 + @ 2")
    
    # Empty expression
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    
    # Whitespace only
    with pytest.raises(ValueError):
        evaluator.evaluate("   ")
