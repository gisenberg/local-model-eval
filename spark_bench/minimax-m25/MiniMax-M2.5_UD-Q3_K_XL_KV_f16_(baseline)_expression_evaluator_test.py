"""
Pytest test suite for ExpressionEvaluator.
Covers basic arithmetic, operator precedence, parentheses,
unary minus, and various error cases.
"""

import pytest
from expression_evaluator import ExpressionEvaluator


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def evaluator():
    """Create a fresh evaluator instance for each test."""
    return ExpressionEvaluator()


# ----------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------
def test_basic_arithmetic(evaluator):
    """Basic binary operations."""
    assert evaluator.evaluate("2 + 3") == 5.0
    assert evaluator.evaluate("10 - 4") == 6.0
    assert evaluator.evaluate("3 * 4") == 12.0
    assert evaluator.evaluate("8 / 2") == 4.0


def test_precedence(evaluator):
    """Operator precedence: * / before + -."""
    assert evaluator.evaluate("2 + 3 * 4") == 14.0
    assert evaluator.evaluate("10 - 2 * 3") == 4.0
    assert evaluator.evaluate("20 / 4 + 3") == 8.0
    assert evaluator.evaluate("2 * 3 + 4 * 5") == 26.0
    # left‑to‑right for same precedence
    assert evaluator.evaluate("8 / 4 / 2") == 1.0


def test_parentheses(evaluator):
    """Parentheses for grouping."""
    assert evaluator.evaluate("(2 + 3) * 4") == 20.0
    assert evaluator.evaluate("2 * (3 + 4)") == 14.0
    assert evaluator.evaluate("((2 + 3))") == 5.0
    assert evaluator.evaluate("((1+2)*(3+4))") == 21.0
    # nested with unary minus
    assert evaluator.evaluate("(-1 + 2) * 3") == 3.0


def test_unary_minus(evaluator):
    """Unary minus (and double minus)."""
    assert evaluator.evaluate("-5") == -5.0
    assert evaluator.evaluate("-3 + 2") == -1.0
    assert evaluator.evaluate("-(2+1)") == -3.0
    assert evaluator.evaluate("--3") == 3.0
    assert evaluator.evaluate("-(-2)") == 2.0
    assert evaluator.evaluate("5 * -2") == -10.0


def test_errors(evaluator):
    """Various error conditions."""
    # Empty expression
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("")
    with pytest.raises(ValueError, match="Empty expression"):
        evaluator.evaluate("   ")

    # Mismatched parentheses
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("(2+3")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        evaluator.evaluate("2+3)")

    # Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("5/0")
    with pytest.raises(ValueError, match="Division by zero"):
        evaluator.evaluate("10/(2-2)")

    # Invalid token
    with pytest.raises(ValueError, match="Invalid token"):
        evaluator.evaluate("2 @ 3")

    # Unexpected token at end (extra characters)
    with pytest.raises(ValueError, match="Unexpected token at end"):
        evaluator.evaluate("2+3 4")
