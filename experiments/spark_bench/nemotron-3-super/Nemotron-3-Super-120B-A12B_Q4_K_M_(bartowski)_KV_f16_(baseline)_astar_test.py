import pytest
from typing import List, Tuple
from astar_grid import AStarGrid


def _path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Sum of movement costs for all cells in *path* except the start cell."""
    if not path:
        return 0
    total = 0
    for cell in path[1:]:  # skip start – you don't pay to be there initially
        r, c = cell
        total += grid[r][c]
    return total


def test_simple_path_uniform_grid():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    # path must start and end correctly
    assert path[0] == start and path[-1] == end
    # each step moves to a neighbour (4‑dir)
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    # optimal cost = Manhattan distance * cell cost (4 moves * 1)
    assert _path_cost(grid, path) == 4


def test_path_around_obstacles():
    grid = [
        [1, 1, 1],
        [1, 0, 1],   # wall at (1,1)
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert path[0] == start and path[-1] == end
    # ensure we never step on the wall
    assert all(grid[r][c] != 0 for r, c in path)
    # optimal detour costs 6 (two extra steps around the wall)
    assert _path_cost(grid, path) == 6


def test_weighted_grid_prefers_lower_cost():
    # high cost corridor (value 5) vs low‑cost detour (value 1)
    grid = [
        [1, 5, 1, 1],
        [1, 5, 1, 5],
        [1, 1, 1, 5],
        [1, 5, 5, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (3, 3)
    path = astar.find_path(start, end)
    assert path is not None
    assert path[0] == start and path[-1] == end
    # The cheapest route goes right‑then‑down along the top/left border
    expected_path = [(0, 0), (0, 1), (0, 2), (0, 3),
                     (1, 3), (2, 3), (3, 3)]
    # verify that the found path matches the known optimal one
    assert path == expected_path
    # compute its cost: sum of entered cells (skip start)
    assert _path_cost(grid, path) == 1 + 1 + 1 + 5 + 1 + 1  # =10


def test_no_path_exists():
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is None


def test_start_equals_end():
    grid = [
        [2, 2],
        [2, 2],
    ]
    astar = AStarGrid(grid)
    start = end = (0, 1)
    path = astar.find_path(start, end)
    assert path == [start]


def test_invalid_coordinates_raises():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))   # row out of bounds
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))   # col out of bounds
