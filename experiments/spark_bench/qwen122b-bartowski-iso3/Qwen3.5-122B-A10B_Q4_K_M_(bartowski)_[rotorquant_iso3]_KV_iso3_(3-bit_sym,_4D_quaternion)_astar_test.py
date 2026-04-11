import pytest
from a_star import AStarGrid

# Helper to calculate total path cost
def calculate_path_cost(grid, path):
    if not path:
        return 0
    total = 0
    for r, c in path:
        total += grid[r][c]
    return total

def test_simple_path_uniform_grid():
    """
    Test 1: Simple path on a uniform grid.
    Grid is 3x3 with all costs 1. Path should be straight or L-shaped with equal cost.
    """
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
    # Manhattan distance is 4 steps (2 down + 2 right), cost = 1 * 5 (including start)
    # Actually, cost is sum of cells entered. 
    # Path length 5 cells. Cost = 5.
    assert len(path) == 5
    assert calculate_path_cost(grid, path) == 5

def test_path_around_obstacles():
    """
    Test 2: Path around obstacles.
    A wall blocks the direct path, forcing a detour.
    """
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))
    
    assert path is not None
    # Verify no wall is in the path
    for r, c in path:
        assert grid[r][c] != 0
    
    # Verify optimality: Must go around the wall.
    # Path: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3) OR similar
    # Cost calculation: 1+1+1+1+1+1 = 6
    assert calculate_path_cost(grid, path) == 6

def test_weighted_grid_optimality():
    """
    Test 3: Weighted grid where path prefers lower-cost cells.
    A longer path with lower costs should be chosen over a shorter path with high costs.
    """
    grid = [
        [1, 1, 1, 1],
        [1, 10, 10, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Start (0,0) to End (2,3)
    # Option A (Direct through middle): (0,0)->(1,0)->(1,1)->(1,2)->(1,3)->(2,3)
    # Cost: 1 + 1 + 10 + 10 + 1 + 1 = 23
    # Option B (Around top/bottom): (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)
    # Cost: 1 + 1 + 1 + 1 + 1 + 1 = 6
    
    path = astar.find_path((0, 0), (2, 3))
    assert path is not None
    
    total_cost = calculate_path_cost(grid, path)
    assert total_cost == 6, f"Expected cost 6, got {total_cost}. Path did not avoid high cost cells."
    
    # Verify the path does not touch the high cost cells (10)
    for r, c in path:
        assert grid[r][c] != 10

def test_no_path_exists():
    """
    Test 4: No path exists (fully blocked).
    Start and end are separated by a wall of 0s.
    """
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is None

def test_start_equals_end():
    """
    Test 5: Start equals end.
    Should return a list containing only the start coordinate.
    """
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    
    assert path == [(1, 1)]
    assert calculate_path_cost(grid, path) == 1

def test_invalid_coordinates():
    """
    Test 6: Invalid coordinates (out of bounds).
    Should raise ValueError.
    """
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
    
    # Test start out of bounds (col)
    with pytest.raises(ValueError):
        astar.find_path((0, 10), (0, 0))

def test_start_or_end_is_wall():
    """
    Test 7: Start or end is a wall.
    Should return None.
    """
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Start is wall
    assert astar.find_path((0, 0), (1, 1)) is None
    
    # End is wall
    assert astar.find_path((1, 1), (0, 0)) is None
