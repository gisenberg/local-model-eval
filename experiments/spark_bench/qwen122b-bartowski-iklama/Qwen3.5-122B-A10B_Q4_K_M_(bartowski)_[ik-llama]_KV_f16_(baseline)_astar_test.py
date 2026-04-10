import pytest
from a_star_grid import AStarGrid

# Helper to calculate path cost for assertions
def get_path_cost(grid, path):
    if not path:
        return 0
    total = 0
    for r, c in path[1:]: # Exclude start node cost
        total += grid[r][c]
    return total

def test_simple_path_uniform_grid():
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
    
    # Optimal cost for 4-directional movement on 3x3 uniform grid is 4 steps (cost 4)
    # Path length should be 5 nodes (start + 4 moves)
    assert len(path) == 5
    assert astar.calculate_path_cost(path) == 4

def test_path_around_obstacles():
    """Test 2: Path finding around obstacles (walls)."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Start (0,0) to End (2,3)
    # Must go around the wall in the middle row
    path = astar.find_path((0, 0), (2, 3))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)
    
    # Verify no wall cells are in the path
    for r, c in path:
        assert grid[r][c] != 0

def test_weighted_grid_optimality():
    """Test 3: Weighted grid where path prefers lower-cost cells over shorter distance."""
    # Grid:
    # S 1 1
    # 1 10 1
    # 1 1 1
    # Direct path via middle (cost 10) is shorter in steps but expensive.
    # Path going around (cost 1+1+1+1+1) is longer in steps but cheaper.
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    
    # Calculate cost
    cost = astar.calculate_path_cost(path)
    
    # Path via middle: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) cost: 1+1+1+1 = 4? 
    # Wait, let's trace:
    # Option A (Top-Right): (0,0)->(0,1)->(0,2)->(1,2)->(2,2). Costs: 1+1+1+1 = 4.
    # Option B (Bottom-Left): (0,0)->(1,0)->(2,0)->(2,1)->(2,2). Costs: 1+1+1+1 = 4.
    # Option C (Through middle): (0,0)->(1,0)->(1,1)->(1,2)->(2,2). Costs: 1+10+1+1 = 13.
    # Option D (Through middle 2): (0,0)->(0,1)->(1,1)->(2,1)->(2,2). Costs: 1+10+1+1 = 13.
    
    # The algorithm should pick cost 4.
    assert cost == 4
    
    # Verify the path does NOT go through (1,1) which has cost 10
    assert (1, 1) not in path

def test_no_path_exists():
    """Test 4: No path exists because the destination is fully blocked."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    # Start (0,0), End (2,2). The column 1 is blocked, and row 2 col 1 is blocked.
    # Actually, let's make it fully blocked.
    grid_blocked = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar_blocked = AStarGrid(grid_blocked)
    path = astar_blocked.find_path((0, 0), (0, 2))
    
    assert path is None

def test_start_equals_end():
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
    assert astar.calculate_path_cost(path) == 0

def test_invalid_coordinates():
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
        
    # Test start out of bounds (negative col)
    with pytest.raises(ValueError):
        astar.find_path((0, -1), (0, 0))

def test_start_or_end_is_wall():
    """Additional Test: Start or End is a wall (0 cost)."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Start is wall
    path = astar.find_path((0, 0), (1, 1))
    assert path is None
    
    # End is wall
    path = astar.find_path((1, 0), (0, 0))
    assert path is None
