import pytest
from expression_evaluator import ExpressionEvaluator

@pytest.fixture
def evaluator():
    """Fixture to provide an ExpressionEvaluator instance."""
    return ExpressionEvaluator()

def test_basic_arithmetic(evaluator):
    """Test basic arithmetic operations."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 5 / 2") == 7.5
    assert evaluator.evaluate("3 * 4 + 5") == 17.0
    assert evaluator.evaluate("10 / 2 - 3") == 2.0

def test_operator_precedence(evaluator):
    """Test correct operator precedence."""
    assert evaluator.evaluate("2 + 3 * 4 - 5 / 2") == 11.5
    assert evaluator.evaluate("3 + 4 * 2 / (1 - 5) ** 2") == 3.5
    assert evaluator.evaluate("10 - 3 * 2 + 4 / 2") == 6.0

def test_parentheses(evaluator):
    """Test parentheses for grouping."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("10 - (5 / 2)") == 7.5
    assert evaluator.evaluate("3 * (4 + 5)") == 27.0
    assert evaluator.evaluate("(10 / 2) - 3") == 2.0
    assert evaluator.evaluate("((2 + 3) * 4) - 5") == 15.0

def test_unary_minus(evaluator):
    """Test unary minus operations."""
    assert evaluator.evaluate("-3") == -3.0
    assert evaluator.evaluate("-(2 + 1)") == -3.0
    assert evaluator.evaluate("3 * -2") == -6.0
    assert evaluator.evaluate("10 / -2") == -5.0
    assert evaluator.evaluate("-3.14") == -3.14

def test_error_cases(evaluator):
    """Test error cases."""
    with pytest.raises(ValueError):
        evaluator.evaluate("(2 + 3")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + 3)")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("1 / 0")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + abc")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + 3 4")
    
    with pytest.raises(ValueError):
        evaluator.evaluate("2 + (3 *")
