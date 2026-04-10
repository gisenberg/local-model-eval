import pytest
from typing import List, Tuple, Optional
from astar import AStarGrid

class TestAStarGrid:
    def test_simple_path(self):
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (1, 1))
        assert path == [(0, 0), (0, 1), (1, 1)]

    def test_start_equals_end(self):
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (0, 0))
        assert path == [(0, 0)]

    def test_no_path_due_to_wall(self):
        grid = [[1, 0], [1, 1]]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (1, 1))
        assert path is None

    def test_out_of_bounds_start(self):
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        with pytest.raises(ValueError):
            astar.find_path((2, 0), (0, 0))

    def test_optimal_cost_path(self):
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        assert path is not None
        total_cost = sum(self.grid[r][c] for r, c in path[1:])
        assert total_cost == 17

    def test_diagonal_movement_not_allowed(self):
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (1, 1))
        assert len(path) == 3
