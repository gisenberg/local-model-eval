import pytest

class TestExpressionEvaluator:
    def test_basic_arithmetic(self):
        """Test basic addition and subtraction."""
        evaluator = ExpressionEvaluator()
        assert evaluator.evaluate("1 + 2") == 3.0
        assert evaluator.evaluate("5 - 2") == 3.0
    
    def test_precedence(self):
        """Test operator precedence (* before +)."""
        evaluator = ExpressionEvaluator()
        # 1 + 2 * 3 should be 1 + (2 * 3) = 7
        assert evaluator.evaluate("1 + 2 * 3") == 7.0
        # 10 / 2 + 5 should be (10 / 2) + 5 = 10
        assert evaluator.evaluate("10 / 2 + 5") == 10.0
    
    def test_parentheses(self):
        """Test parentheses for grouping."""
        evaluator = ExpressionEvaluator()
        # (1 + 2) * 3 should be 9
        assert evaluator.evaluate("(1 + 2) * 3") == 9.0
        # Nested parentheses
        assert evaluator.evaluate("((1 + 2) * 3)") == 9.0
    
    def test_unary_minus(self):
        """Test unary minus support."""
        evaluator = ExpressionEvaluator()
        # Simple unary minus
        assert evaluator.evaluate("-5") == -5.0
        # Unary minus with parentheses
        assert evaluator.evaluate("-(2 + 1)") == -3.0
        # Unary minus in expression
        assert evaluator.evaluate("3 + -4") == -1.0
    
    def test_error_cases(self):
        """Test error handling for invalid inputs."""
        evaluator = ExpressionEvaluator()
        
        # Empty expression
        with pytest.raises(ValueError, match="Empty expression"):
            evaluator.evaluate("")
        
        # Mismatched parentheses
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            evaluator.evaluate("(1 + 2")
        
        # Division by zero
        with pytest.raises(ValueError, match="Division by zero"):
            evaluator.evaluate("1 / 0")
        
        # Invalid token (e.g., 'abc')
        with pytest.raises(ValueError, match="Invalid character"):
            evaluator.evaluate("3 + abc")

if __name__ == "__main__":
    # Example usage
    ev = ExpressionEvaluator()
    print(f"1 + 2 * 3 = {ev.evaluate('1 + 2 * 3')}")
    print(f"-(2+1) = {ev.evaluate('-(2+1)')}")
