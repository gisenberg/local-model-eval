import pytest
from typing import List, Tuple, Optional
from astar_grid import AStarGrid

def test_simple_path_uniform_grid():
    """Test finding a path on a uniform grid where all cells have the same cost."""
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
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    assert len(path) == 5
    assert path[0] == start and path[-1] == end

def test_path_around_obstacles():
    """Test finding a path that must go around obstacles (walls)."""
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)

    path = astar.find_path(start, end)

    assert path is not None
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)]
    assert len(path) == 5
    assert path[0] == start and path[-1] == end

def test_weighted_grid_path_preference():
    """Test that the path prefers lower-cost cells in a weighted grid."""
    grid = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)

    path = astar.find_path(start, end)

    assert path is not None
    # The optimal path should go around the high-cost cell (1,1)
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert len(path) == 5
    assert path[0] == start and path[-1] == end

    # Calculate total cost of the path
    total_cost = sum(grid[row][col] for row, col in path)
    # The minimal possible cost would be 4 (going through the high-cost cell)
    # But since we prefer lower-cost cells, the actual minimal cost is 5
    assert total_cost == 5

def test_no_path_exists_fully_blocked():
    """Test that returns None when no path exists (fully blocked grid)."""
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)

    path = astar.find_path(start, end)

    assert path is None

def test_start_equals_end():
    """Test that returns [start] when start and end are the same."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (1, 1)
    end = (1, 1)

    path = astar.find_path(start, end)

    assert path == [(1, 1)]

def test_invalid_coordinates():
    """Test that raises ValueError for out-of-bounds coordinates."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)

    # Test start out of bounds
    with pytest.raises(ValueError):
        astar.find_path((3, 3), (1, 1))

    # Test end out of bounds
    with pytest.raises(ValueError):
        astar.find_path((1, 1), (3, 3))

    # Test both out of bounds
    with pytest.raises(ValueError):
        astar.find_path((3, 3), (4, 4))

    # Test negative coordinates
    with pytest.raises(ValueError):
        astar.find_path((-1, -1), (1, 1))
