import pytest
from typing import List, Tuple
from astar import AStarGrid

# Helper to calculate total path cost
def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    if not path:
        return 0
    total = 0
    for r, c in path:
        total += grid[r][c]
    return total

class TestAStarGrid:
    
    def test_simple_path_uniform_grid(self):
        """Test 1: Simple path on a uniform grid (all costs = 1)."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        
        # Optimal cost for 3x3 grid from corner to corner is 5 steps (Manhattan distance)
        # Cost = 1 (start) + 1 + 1 + 1 + 1 + 1 (end) = 6? 
        # Wait, path includes start and end. 
        # Path length (nodes) = 5 (e.g., (0,0)->(0,1)->(0,2)->(1,2)->(2,2))
        # Cost = 1+1+1+1+1 = 5.
        # Let's verify length.
        assert len(path) == 5
        assert calculate_path_cost(grid, path) == 5

    def test_path_around_obstacles(self):
        """Test 2: Path finding around obstacles (0s)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        # The path must go around the middle block
        # One valid path: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)
        # Another: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3)
        # Both have length 6 nodes, cost 6.
        assert len(path) == 6
        assert calculate_path_cost(grid, path) == 6
        # Verify no walls are in path
        for r, c in path:
            assert grid[r][c] != 0

    def test_weighted_grid_optimality(self):
        """Test 3: Weighted grid where path prefers lower-cost cells."""
        # High cost in the middle row, low cost on top/bottom
        grid = [
            [1, 1, 1, 1],
            [1, 10, 10, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        
        # Direct path through middle would be: (0,0)->(1,0)->(1,1)->(1,2)->(1,3)->(2,3)
        # Cost: 1 + 1 + 10 + 10 + 1 + 1 = 24
        # Path around top/bottom: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)
        # Cost: 1 + 1 + 1 + 1 + 1 + 1 = 6
        
        assert calculate_path_cost(grid, path) == 6
        
        # Verify the path does not go through the high cost cells (10)
        for r, c in path:
            assert grid[r][c] != 10

    def test_no_path_exists(self):
        """Test 4: No path exists because the grid is fully blocked."""
        grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is None

    def test_start_equals_end(self):
        """Test 5: Start and end coordinates are the same."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        
        assert path is not None
        assert path == [(1, 1)]
        assert len(path) == 1

    def test_invalid_coordinates(self):
        """Test 6: Start or end coordinates are out of bounds."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Test start out of bounds
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (0, 0))
            
        # Test end out of bounds
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (5, 5))
            
        # Test negative column
        with pytest.raises(ValueError):
            astar.find_path((0, -1), (0, 0))

    def test_start_or_end_is_wall(self):
        """Additional edge case: Start or end is a wall (0)."""
        grid = [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Start is wall
        assert astar.find_path((0, 0), (2, 2)) is None
        
        # End is wall
        assert astar.find_path((0, 1), (0, 0)) is None
