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
        """Test 1: Simple path on a uniform grid (all costs are 1)."""
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
        
        # In a uniform grid, Manhattan distance is the optimal cost (excluding start cost if we count entry)
        # Here we count entry cost. Path length is 5 steps (0,0 -> 0,1 -> 0,2 -> 1,2 -> 2,2)
        # Cost = 1+1+1+1+1 = 5.
        # Any path of length 5 is optimal.
        assert len(path) == 5
        assert calculate_path_cost(grid, path) == 5

    def test_path_around_obstacles(self):
        """Test 2: Path finding around obstacles (walls)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        # Start (0,0) to End (2,3). Must go around the wall in the middle.
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
        # Option A (Straight through middle): 1 + 10 + 10 + 10 + 1 = 32 (approx, depends on exact path)
        # Option B (Go around top/bottom): 1+1+1+1+1+1+1+1+1 = 9 (approx)
        # The algorithm should choose the path around the 10s.
        path = astar.find_path((0, 0), (2, 4))
        
        assert path is not None
        total_cost = calculate_path_cost(grid, path)
        
        # The optimal path should avoid the 10s entirely.
        # A path going (0,0)->(0,4)->(2,4) or similar.
        # Minimum steps: 4 right + 2 down = 6 steps. 7 cells total.
        # Cost = 7 * 1 = 7.
        # If it went through 10s, cost would be significantly higher.
        assert total_cost < 20, f"Path cost {total_cost} is too high, likely went through 10s."
        
        # Verify no 10s in path
        for r, c in path:
            assert grid[r][c] != 10

    def test_no_path_exists(self):
        """Test 4: No path exists because the destination is fully blocked."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        # Start (0,0) to End (2,2). The column 1 is blocked, and row 2 col 1 is blocked.
        # Actually, let's make it fully blocked:
        grid_blocked = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]
        ]
        astar_blocked = AStarGrid(grid_blocked)
        path = astar_blocked.find_path((0, 0), (0, 2))
        
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
            [1, 0, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Start is a wall
        path_start_wall = astar.find_path((0, 1), (1, 1))
        assert path_start_wall is None
        
        # End is a wall
        path_end_wall = astar.find_path((0, 0), (0, 1))
        assert path_end_wall is None
