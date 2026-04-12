# test_expression_evaluator.py
import pytest
from expression_evaluator import ExpressionEvaluator

ev = ExpressionEvaluator()

# ----------------------------------------------------------------------
# Basic arithmetic
# ----------------------------------------------------------------------
def test_basic_addition():
    assert ev.evaluate("2 + 3") == 5.0

def test_basic_subtraction():
    assert ev.evaluate("5 - 2") == 3.0

def test_basic_multiplication():
    assert ev.evaluate("3 * 4") == 12.0

def test_basic_division():
    assert ev.evaluate("8 / 2") == 4.0

# ----------------------------------------------------------------------
# Operator precedence
# ----------------------------------------------------------------------
def test_precedence_mul_over_add():
    # 2 + 3 * 4 should be 2 + (3 * 4) = 14
    assert ev.evaluate("2 + 3 * 4") == 14.0

def test_precedence_div_over_sub():
    # 10 - 2 * 3 should be 10 - (2 * 3) = 4
    assert ev.evaluate("10 - 2 * 3") == 4.0

def test_precedence_mixed():
    # 10 / 2 + 3 * 4 - 1 = 5 + 12 - 1 = 16
    assert ev.evaluate("10 / 2 + 3 * 4 - 1") == 16.0

# ----------------------------------------------------------------------
# Parentheses
# ----------------------------------------------------------------------
def test_parentheses_grouping():
    # (2 + 3) * 4 = 5 * 4 = 20
    assert ev.evaluate("(2 + 3) * 4") == 20.0

def test_nested_parentheses():
    # ((2 + 3) * (1 + 1)) = 5 * 2 = 10
    assert ev.evaluate("((2 + 3) * (1 + 1))") == 10.0

def test_parentheses_with_precedence():
    # (10 / 2 + 3) * 2 = (5 + 3) * 2 = 16
    assert ev.evaluate("(10 / 2 + 3) * 2") == 16.0

# ----------------------------------------------------------------------
# Unary minus
# ----------------------------------------------------------------------
def test_unary_minus_simple():
    assert ev.evaluate("-3") == -3.0

def test_unary_minus_in_expression():
    # -3 + 5 = 2
    assert ev.evaluate("-3 + 5") == 2.0

def test_unary_minus_with_parentheses():
    # -(2 + 1) = -3
    assert ev.evaluate("-(2 + 1)") == -3.0

def test_double_unary_minus():
    # --5 = 5
    assert ev.evaluate("--5") == 5.0

# ----------------------------------------------------------------------
# Floating point numbers
# ----------------------------------------------------------------------
def test_floating_point():
    assert ev.evaluate("3.14") == 3.14

def test_floating_point_arithmetic():
    # 1.5 + 2.5 = 4.0
    assert ev.evaluate("1.5 + 2.5") == 4.0

def test_floating_point_precedence():
    # 3.0 * 2.0 + 1.0 = 6.0 + 1.0 = 7.0
    assert ev.evaluate("3.0 * 2.0 + 1.0") == 7.0

# ----------------------------------------------------------------------
# Error cases
# ----------------------------------------------------------------------
def test_mismatched_parentheses_left():
    with pytest.raises(ValueError):
        ev.evaluate("(2 + 3")

def test_mismatched_parentheses_right():
    with pytest.raises(ValueError):
        ev.evaluate("2 + 3)")

def test_division_by_zero():
    with pytest.raises(ValueError):
        ev.evaluate("5 / 0")

def test_invalid_token():
    with pytest.raises(ValueError):
        ev.evaluate("2 $ 3")

def test_empty_expression():
    with pytest.raises(ValueError):
        ev.evaluate("   ")

def test_invalid_syntax():
    with pytest.raises(ValueError):
        ev.evaluate("2 + + 3")
