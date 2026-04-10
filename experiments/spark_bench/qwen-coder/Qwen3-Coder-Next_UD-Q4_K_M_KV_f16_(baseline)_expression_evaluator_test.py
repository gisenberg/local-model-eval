import pytest
from expression_evaluator import ExpressionEvaluator


@pytest.fixture
def evaluator():
    return ExpressionEvaluator()


def test_basic_arithmetic(evaluator):
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("6 * 7") == 42.0
    assert evaluator.evaluate("8 / 2") == 4.0


def test_operator_precedence(evaluator):
    # * and / before + and -
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 4 / 2") == 8.0
    assert evaluator.evaluate("3 * 5 + 2") == 17.0
    assert evaluator.evaluate("12 / 3 + 1") == 5.0


def test_parentheses(evaluator):
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("((2 + 3) * 4) - 1") == 19.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((1))") == 1.0


def test_unary_minus(evaluator):
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(3 + 2)") == -5.0
    assert evaluator.evaluate("--3") == 3.0  # double unary minus
    assert evaluator.evaluate("-2 + 5") == 3.0
    assert evaluator.evaluate("5 + -2") == 3.0
    assert evaluator.evaluate("-(2 * 3)") == -6.0


def test_error_cases(evaluator):
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2 + 3")

    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2 + 3)")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5 / 0")

    # Invalid token
    with pytest.raises(ValueError, match="Invalid character"):
        evaluator.evaluate("2 + a")

    # Unexpected token after expression
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + 3 +")

    # Missing operand after operator
    with pytest.raises(ValueError, match="Unexpected token"):
        evaluator.evaluate("2 + * 3")
