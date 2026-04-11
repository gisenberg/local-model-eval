import pytest
from expression_evaluator import ExpressionEvaluator

def test_basic_arithmetic():
    """Test basic arithmetic operations with correct precedence"""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2+3") == 5.0
    assert evaluator.evaluate("5-2") == 3.0
    assert evaluator.evaluate("4*5") == 20.0
    assert evaluator.evaluate("10/2") == 5.0
    assert evaluator.evaluate("1+2*3") == 7.0  # precedence test
    assert evaluator.evaluate("(1+2)*3") == 9.0  # parentheses test

def test_operator_precedence():
    """Test that operator precedence is correctly handled"""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("2+3*4") == 14.0  # * before +
    assert evaluator.evaluate("2*3+4") == 10.0  # * before +
    assert evaluator.evaluate("10-2*3") == 4.0  # * before -
    assert evaluator.evaluate("10/2+3") == 8.0  # / before +
    assert evaluator.evaluate("2+3+4*5") == 25.0  # * before + with multiple +
    assert evaluator.evaluate("2*3*4+5") == 29.0  # multiple * before +

def test_parentheses():
    """Test that parentheses correctly affect evaluation order"""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("(2+3)*4") == 20.0
    assert evaluator.evaluate("2*(3+4)") == 14.0
    assert evaluator.evaluate("((1+2)*3)+4") == 13.0
    assert evaluator.evaluate("(10/(2+3))") == 2.0
    assert evaluator.evaluate("10/((2+3)*2)") == 1.0
    assert evaluator.evaluate("-(2+3)") == -5.0  # unary minus with parentheses

def test_unary_minus():
    """Test handling of unary minus operations"""
    evaluator = ExpressionEvaluator()
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-3.14") == -3.14
    assert evaluator.evaluate("2+-3") == -1.0
    assert evaluator.evaluate("2--3") == 5.0
    assert evaluator.evaluate("(-3)") == -3.0
    assert evaluator.evaluate("-(2*3)") == -6.0
    assert evaluator.evaluate("2*(-3)") == -6.0
    assert evaluator.evaluate("2/-3") == -0.6666666666666666

def test_error_cases():
    """Test that appropriate errors are raised for invalid expressions"""
    evaluator = ExpressionEvaluator()

    # Empty expression
    with pytest.raises(ValueError):
        evaluator.evaluate("")

    # Mismatched parentheses
    with pytest.raises(ValueError):
        evaluator.evaluate("(2+3")

    with pytest.raises(ValueError):
        evaluator.evaluate("2+3)")

    with pytest.raises(ValueError):
        evaluator.evaluate("((2+3)*4")

    # Division by zero
    with pytest.raises(ValueError):
        evaluator.evaluate("1/0")

    with pytest.raises(ValueError):
        evaluator.evaluate("1/(2-2)")

    # Invalid tokens
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + ")

    with pytest.raises(ValueError):
        evaluator.evaluate("2 + a")

    with pytest.raises(ValueError):
        evaluator.evaluate("2 + 3 *")

    # Multiple operators in sequence
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + + 3")
