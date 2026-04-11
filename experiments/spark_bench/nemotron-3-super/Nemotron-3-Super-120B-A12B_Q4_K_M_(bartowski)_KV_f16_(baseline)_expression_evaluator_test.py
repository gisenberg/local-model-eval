import pytest
from expression_evaluator import ExpressionEvaluator  # adjust import if needed


def test_basic_arithmetic():
    ev = ExpressionEvaluator()
    assert ev.evaluate("1+2") == 3.0
    assert ev.evaluate("5-3") == 2.0
    assert ev.evaluate("2*3") == 6.0
    assert ev.evaluate("8/4") == 2.0


def test_precedence():
    ev = ExpressionEvaluator()
    assert ev.evaluate("1+2*3") == 7.0      # * before +
    assert ev.evaluate("1*2+3") == 5.0      # * before +
    assert ev.evaluate("1+2+3*4") == 15.0   # * before +
    assert ev.evaluate("10/2-3") == 2.0     # / before -


def test_parentheses():
    ev = ExpressionEvaluator()
    assert ev.evaluate("(1+2)*3") == 9.0
    assert ev.evaluate("2*(3+4)") == 14.0
    assert ev.evaluate("((1+2))") == 3.0
    assert ev.evaluate("-(2+3)") == -5.0   # unary minus with parentheses


def test_unary_minus():
    ev = ExpressionEvaluator()
    assert ev.evaluate("-3") == -3.0
    assert ev.evaluate("+5") == 5.0
    assert ev.evaluate("-(-2)") == 2.0
    assert ev.evaluate("-(2+1)*3") == -9.0  # unary minus applies to the parenthesised sum


def test_error_cases():
    ev = ExpressionEvaluator()

    # Division by zero
    with pytest.raises(ValueError):
        ev.evaluate("1/0")

    # Mismatched parentheses
    with pytest.raises(ValueError):
        ev.evaluate("(1+2")

    # Empty expression
    with pytest.raises(ValueError):
        ev.evaluate("")

    # Invalid token (two operators in a row)
    with pytest.raises(ValueError):
        ev.evaluate("1++2")

    # Invalid character
    with pytest.raises(ValueError):
        ev.evaluate("1$a")
