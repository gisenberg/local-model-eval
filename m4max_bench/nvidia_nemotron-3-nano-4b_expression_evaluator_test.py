import pytest
from expression_evaluator import ExpressionEvaluator

evaluator = ExpressionEvaluator()


def test_basic_arithmetic():
    assert evaluator.evaluate("2+3*4") == 14.0


def test_precedence():
    # 5 - (3/2) = 5 - 1.5 = 3.5
    assert evaluator.evaluate("5-3/2") == pytest.approx(3.5)


def test_parentheses_grouping():
    assert evaluator.evaluate("(2+3)*4") == 20.0


def test_unary_minus():
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2+1)") == -3.0


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("((2+3)", "Mismatched parentheses: leftover tokens [')']"),
        ("5/0", "Division by zero"),
        ("2 + a", "Invalid token encountered in expression: 'a'"),
        ("", "Empty expression"),
    ],
)
def test_error_cases(expr, expected):
    with pytest.raises(ValueError) as exc:
        evaluator.evaluate(expr)
    assert str(exc.value) == expected
