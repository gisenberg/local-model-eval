import pytest
from typing import List, Tuple
from a_star_grid import AStarGrid

def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Helper to calculate the total cost of a path (sum of costs of entered cells)."""
    if not path:
        return 0
    # The cost of the path is the sum of the costs of the cells we step INTO.
    # Usually, the start cell cost is not included in the movement cost, 
    # but the problem says "cost to enter that cell". 
    # Standard interpretation: Sum of grid[r][c] for all cells in path EXCEPT start.
    # However, if the problem implies the cost of the path is the sum of all nodes visited including start,
    # we adjust. Based on standard A* on weighted grids, we sum the costs of the edges/nodes traversed.
    # Let's assume the cost is the sum of the values of the cells we move into.
    # If start == end, cost is 0 (no movement).
    
    total = 0
    for i, (r, c) in enumerate(path):
        if i == 0:
            continue # Don't count start cell cost as we are already there
        total += grid[r][c]
    return total

def is_path_valid(grid: List[List[int]], path: List[Tuple[int, int]], start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    """Checks if the path is contiguous, within bounds, and avoids walls."""
    if not path:
        return False
    if path[0] != start or path[-1] != end:
        return False
    
    for i, (r, c) in enumerate(path):
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            return False
        if grid[r][c] == 0:
            return False
        
        if i > 0:
            prev_r, prev_c = path[i-1]
            dist = abs(r - prev_r) + abs(c - prev_c)
            if dist != 1:
                return False
    return True

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
        assert is_path_valid(grid, path, (0, 0), (2, 2))
        # Optimal cost for 3x3 uniform grid (Manhattan distance 4 steps)
        # Path length is 5 nodes. Steps = 4. Cost = 4 * 1 = 4.
        cost = calculate_path_cost(grid, path)
        assert cost == 4

    def test_path_around_obstacles(self):
        """Test 2: Path must go around obstacles (0s)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        assert is_path_valid(grid, path, (0, 0), (2, 3))
        # Must go down to row 2 or up (if possible, but here only down)
        # Path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3)
        # Cost: 1+1+1+1+1 = 5
        cost = calculate_path_cost(grid, path)
        assert cost == 5

    def test_weighted_grid_prefers_lower_cost(self):
        """Test 3: Weighted grid where path prefers lower-cost cells over shorter distance."""
        # Grid:
        # S 1 1
        # 1 10 1
        # 1 1 1 E
        # Direct path via middle (1,1) costs 10.
        # Path around (down then right) costs 1+1+1+1 = 4.
        grid = [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        assert is_path_valid(grid, path, (0, 0), (2, 2))
        
        cost = calculate_path_cost(grid, path)
        # The optimal path should avoid the 10.
        # Path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) OR (0,0)->(0,1)->(0,2)->(1,2)->(2,2)
        # Both have cost 4.
        assert cost == 4
        # Ensure we didn't take the expensive route
        assert (1, 1) not in path

    def test_no_path_exists_fully_blocked(self):
        """Test 4: No path exists because the destination is surrounded by walls."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        # Start at (0,0), End at (2,2). (2,2) is reachable? 
        # Let's make (2,2) unreachable.
        grid_blocked = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0] # (2,2) is 0 (wall)
        ]
        astar_blocked = AStarGrid(grid_blocked)
        path = astar_blocked.find_path((0, 0), (2, 2))
        assert path is None

        # Another case: End is reachable but surrounded by walls from start
        grid_isolated = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]
        astar_iso = AStarGrid(grid_isolated)
        path_iso = astar_iso.find_path((0, 0), (2, 2))
        assert path_iso is None

    def test_start_equals_end(self):
        """Test 5: Start and end are the same coordinate."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        
        assert path is not None
        assert path == [(1, 1)]
        assert calculate_path_cost(grid, path) == 0

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
            
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (2, 0))
            
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (0, 5))

    def test_start_or_end_is_wall(self):
        """Additional edge case: Start or End is a wall (0)."""
        grid = [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0]
        ]
        astar = AStarGrid(grid)
        
        # Start is wall
        path_start_wall = astar.find_path((0, 0), (1, 1))
        assert path_start_wall is None
        
        # End is wall
        path_end_wall = astar.find_path((1, 1), (2, 2))
        assert path_end_wall is None
