import pytest
from a_star import AStarGrid

def test_simple_path():
    """Test simple path on uniform grid."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)
    path = astar.find_path(start, end)
    
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert len(path) == 5  # Manhattan distance + 1
    assert sum(grid[r][c] for r, c in path) == 5  # Total cost should be 5

def test_path_around_obstacles():
    """Test path around obstacles."""
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)
    path = astar.find_path(start, end)
    
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert (0, 1) not in path  # Should go around the wall
    assert (2, 1) not in path
    assert sum(grid[r][c] for r, c in path) == 6  # Path: (0,0)->(1,0)->(1,1)->(1,2)->(2,2)

def test_weighted_grid():
    """Test path on weighted grid (prefers lower-cost cells)."""
    grid = [
        [1, 9, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)
    path = astar.find_path(start, end)
    
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert (0, 1) not in path  # Should avoid high-cost cell
    assert (1, 1) not in path
    assert sum(grid[r][c] for r, c in path) == 5  # Path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)

def test_no_path_exists():
    """Test when no path exists (fully blocked)."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)
    path = astar.find_path(start, end)
    
    assert path is None

def test_start_equals_end():
    """Test when start equals end."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (1, 1)
    end = (1, 1)
    path = astar.find_path(start, end)
    
    assert path is not None
    assert path == [start]

def test_invalid_coordinates():
    """Test invalid coordinates (out of bounds)."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))  # Start out of bounds
    
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (3, 0))  # End out of bounds

def test_wall_at_start_or_end():
    """Test when start or end is a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Start is wall
    assert astar.find_path((1, 1), (2, 2)) is None
    
    # End is wall
    assert astar.find_path((0, 0), (1, 1)) is None
