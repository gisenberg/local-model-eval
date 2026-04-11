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
        
        # In a 3x3 uniform grid, shortest path length (nodes) is 5 (e.g., R,R,D,D or D,D,R,R)
        # Cost = 5 * 1 = 5
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
        # Start (0,0) to End (2,3). Must go around the wall in row 1.
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        # Verify no wall cells are in the path
        for r, c in path:
            assert grid[r][c] != 0
        
        # Verify connectivity
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1

    def test_weighted_grid_optimality(self):
        """Test 3: Weighted grid where path prefers lower-cost cells over shorter distance."""
        # Grid:
        # S 1 1
        # 1 10 1
        # 1 1 1 E
        # Direct path via middle (cost 10) is shorter in steps but expensive.
        # Path around (cost 1s) is longer in steps but cheaper.
        grid = [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        
        # Path via middle: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) cost: 1+1+1+1+1 = 5? 
        # Wait, let's construct a clearer example.
        # S 1 1
        # 1 100 1
        # 1 1 1 E
        # Path 1 (Top/Right): (0,0)->(0,1)->(0,2)->(1,2)->(2,2) Cost: 1+1+1+1+1 = 5
        # Path 2 (Bottom/Left): (0,0)->(1,0)->(2,0)->(2,1)->(2,2) Cost: 1+1+1+1+1 = 5
        # Let's make the direct path expensive.
        
        grid = [
            [1, 1, 1],
            [1, 100, 1],
            [1, 1, 1]
        ]
        # Actually, let's force a choice:
        # S 1 1
        # 1 10 1
        # 1 1 1 E
        # If we go (0,0)->(1,0)->(2,0)->(2,1)->(2,2) cost = 1+1+1+1+1 = 5
        # If we go (0,0)->(0,1)->(0,2)->(1,2)->(2,2) cost = 1+1+1+1+1 = 5
        # Let's make the "short" path go through a high cost.
        
        grid = [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ]
        # Start (0,0), End (2,2).
        # Option A: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) Cost: 1+1+1+1+1 = 5
        # Option B: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) Cost: 1+1+1+1+1 = 5
        # Option C: (0,0)->(1,0)->(1,1)->(1,2)->(2,2) Cost: 1+10+1+1+1 = 14 (Bad)
        
        # Let's try a specific case where the heuristic might mislead if not weighted correctly.
        # S 1 1
        # 1 10 1
        # 1 1 1 E
        # Actually, let's just ensure it picks the low cost path if one exists.
        grid = [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ]
        # Let's change the grid to force a specific optimal path.
        # S 1 1
        # 1 10 1
        # 1 1 1 E
        # If we make the top path expensive?
        grid = [
            [1, 10, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        # Start (0,0) to (2,2).
        # Path 1 (Top): (0,0)->(0,1)->(0,2)->(1,2)->(2,2) Cost: 1+10+1+1+1 = 14
        # Path 2 (Bottom): (0,0)->(1,0)->(2,0)->(2,1)->(2,2) Cost: 1+1+1+1+1 = 5
        # A* should pick Path 2.
        
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        total_cost = calculate_path_cost(grid, path)
        assert total_cost == 5, f"Expected cost 5, got {total_cost}. Path: {path}"
        
        # Verify it didn't take the expensive route
        assert (0, 1) not in path

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
        assert calculate_path_cost(grid, path) == grid[1][1]

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
            
        # Test negative index
        with pytest.raises(ValueError):
            astar.find_path((0, -1), (0, 0))

    def test_start_or_end_is_wall(self):
        """Additional edge case: Start or End is a wall (0)."""
        grid = [
            [0, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Start is wall
        assert astar.find_path((0, 0), (1, 1)) is None
        
        # End is wall
        assert astar.find_path((1, 0), (0, 0)) is None
