import pytest
from typing import List, Tuple
from a_star_grid import AStarGrid

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
        # Cost = 1 * 5 = 5 (since we count the cost of the start cell too? 
        # Note: Usually path cost includes start and end. 
        # Path length is 5 cells: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) is 5 cells? 
        # Actually Manhattan distance is 4 steps, so 5 cells.
        # Let's verify: (0,0) to (2,2) requires 2 down, 2 right. Total 4 moves. 5 cells visited.
        # Cost = 5 * 1 = 5.
        assert calculate_path_cost(grid, path) == 5

    def test_path_around_obstacles(self):
        """Test 2: Path finding around obstacles (walls)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        # Start (0,0) to End (2,3). Must go around the wall in row 1.
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        # Verify no wall cells are in the path
        for r, c in path:
            assert grid[r][c] > 0
        
        # Verify connectivity
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1

    def test_weighted_grid_optimality(self):
        """Test 3: Weighted grid where path prefers lower-cost cells."""
        # Grid with a "high cost" barrier in the middle row
        # Path should go around the high cost cells if the alternative is cheaper
        grid = [
            [1, 1, 1, 1, 1],
            [1, 10, 10, 10, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        # Start (0,0) to End (2,4)
        # Direct path through middle: 1 + 10 + 10 + 10 + 1 = 32 (approx, depends on exact cells)
        # Path going around top/bottom: 
        # (0,0)->(0,1)->(0,2)->(0,3)->(0,4)->(1,4)->(2,4) 
        # Cost: 1+1+1+1+1+1+1 = 7
        path = astar.find_path((0, 0), (2, 4))
        
        assert path is not None
        total_cost = calculate_path_cost(grid, path)
        
        # The optimal path should avoid the 10s.
        # Minimum possible cost is 7 (7 cells of cost 1).
        assert total_cost == 7
        
        # Verify the path does not touch the 10s
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
        assert len(path) == 1
        assert path[0] == (1, 1)
        assert calculate_path_cost(grid, path) == 1

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
            [0, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Start is wall
        assert astar.find_path((0, 0), (1, 1)) is None
        
        # End is wall
        assert astar.find_path((1, 0), (0, 0)) is None
