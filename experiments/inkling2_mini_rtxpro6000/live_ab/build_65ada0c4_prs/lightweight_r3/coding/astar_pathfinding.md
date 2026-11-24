Here's a complete implementation plus the test suite.

## `astar.py`

```python
"""A* pathfinding on a weighted 2D grid (4-directional movement)."""

import heapq
import itertools
from typing import Dict, List, Optional, Set, Tuple

Coord = Tuple[int, int]  # (row, col)


class AStarGrid:
    """A* pathfinder for a weighted 2D grid.

    Grid values are the cost of *entering* a cell:
        0      -> impassable wall
        > 0    -> cost to enter that cell
    Movement is 4-directional (up, down, left, right).
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid and validate its structure.

        Args:
            grid: Non-empty rectangular list of lists of non-negative ints.

        Raises:
            ValueError: If the grid is empty, non-rectangular, or contains
                        invalid cell values.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be a non-empty 2D list")

        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                raise ValueError("Grid must be rectangular")
            for cell in row:
                if not isinstance(cell, int) or cell < 0:
                    raise ValueError("Grid cells must be non-negative integers (0 = wall)")

        self.grid: List[List[int]] = grid
        self.rows: int = len(grid)
        self.cols: int = width

    def find_path(
        self, start: Coord, end: Coord
    ) -> Optional[List[Coord]]:
        """Find the lowest-cost path from start to end.

        Args:
            start: (row, col) coordinate of the start cell.
            end: (row, col) coordinate of the target cell.

        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if no path exists or either endpoint is a wall.
            If start == end, returns [start] (provided it is not a wall).

        Raises:
            ValueError: If start or end is out of bounds.
        """
        self._check_bounds(start)
        self._check_bounds(end)

        sr, sc = start
        er, ec = end

        # Wall endpoints can never be reached / walked on.
        if self.grid[sr][sc] == 0 or self.grid[er][ec] == 0:
            return None

        if start == end:
            return [start]

        open_heap: List[Tuple[int, int, int, Coord]] = []
        counter = itertools.count()  # tiebreaker so heap never compares cells
        g_score: Dict[Coord, int] = {start: 0}
        came_from: Dict[Coord, Coord] = {}
        closed: Set[Coord] = set()

        heapq.heappush(
            open_heap,
            (self._heuristic(start, end), 0, next(counter), start),
        )

        while open_heap:
            _, g, _, current = heapq.heappop(open_heap)

            if current == end:
                return self._reconstruct_path(came_from, start, end)

            if current in closed:
                continue  # stale heap entry
            if g > g_score.get(current, float("inf")):
                continue  # a cheaper path was already recorded
            closed.add(current)

            for neighbor in self._neighbors(current):
                tentative = g + self.grid[neighbor[0]][neighbor[1]]
                if tentative < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = current
                    f_next = tentative + self._heuristic(neighbor, end)
                    heapq.heappush(
                        open_heap, (f_next, tentative, next(counter), neighbor)
                    )

        return None

    def _neighbors(self, cell: Coord) -> List[Coord]:
        """Return the in-bounds, non-wall neighbors of a cell (4-directional)."""
        r, c = cell
        neighbors: List[Coord] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] > 0:
                neighbors.append((nr, nc))
        return neighbors

    @staticmethod
    def _heuristic(a: Coord, b: Coord) -> int:
        """Manhattan distance between two cells.

        Admissible (never overestimates) because every traversable cell
        costs at least 1, so each Manhattan step can save at most 1.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _check_bounds(self, cell: Coord) -> None:
        """Raise ValueError if a coordinate is outside the grid."""
        r, c = cell
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(
                f"Coordinate {cell} is out of bounds for a {self.rows}x{self.cols} grid"
            )

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Coord, Coord], start: Coord, end: Coord
    ) -> List[Coord]:
        """Walk back from end to start and return the full path (start -> end)."""
        path = [end]
        current = end
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
```

## `test_astar.py`

```python
"""Pytest suite for AStarGrid."""

import pytest

from astar import AStarGrid


def total_cost(path, grid):
    """Cost of a path = sum of enter-costs of every cell after the start."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(path, grid, start, end):
    """Assert the path connects start->end, stays on cells, and steps are 4-adjacent."""
    assert path[0] == start
    assert path[-1] == end
    for r, c in path:
        assert grid[r][c] > 0, "path passes through a wall"
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "non-adjacent step"


def test_simple_path_uniform_grid():
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    start, end = (0, 0), (2, 2)

    path = AStarGrid(grid).find_path(start, end)

    assert path is not None
    assert_valid_path(path, grid, start, end)
    # Optimal: 4 steps * cost 1 = 4 (matches Manhattan lower bound).
    assert total_cost(path, grid) == 4


def test_path_around_obstacles():
    grid = [[1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]]
    start, end = (0, 0), (0, 4)

    path = AStarGrid(grid).find_path(start, end)

    assert path is not None
    assert_valid_path(path, grid, start, end)
    # Optimal detour via row 1: (1,0)->(1,1)->(1,2)->(1,3)->(1,4)->(0,4) = 6.
    assert total_cost(path, grid) == 6


def test_weighted_grid_prefers_lower_cost_cells():
    grid = [[1, 10, 10],
            [1, 1, 10],
            [10, 1, 1]]
    start, end = (0, 0), (2, 2)

    path = AStarGrid(grid).find_path(start, end)

    assert path is not None
    assert_valid_path(path, grid, start, end)
    # Optimal route: (1,0)->(1,1)->(2,1)->(2,2) = 1+1+1+1 = 4.
    # (Direct row-0 route would cost 31, so this proves preference for cheap cells.)
    assert total_cost(path, grid) == 4


def test_no_path_exists():
    grid = [[1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]]
    start, end = (0, 0), (0, 2)

    path = AStarGrid(grid).find_path(start, end)

    assert path is None


def test_start_equals_end():
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]

    path = AStarGrid(grid).find_path((1, 1), (1, 1))

    assert path == [(1, 1)]


def test_invalid_coordinates_raise_value_error():
    astar = AStarGrid([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

    with pytest.raises(ValueError):
        astar.find_path((3, 0), (0, 0))    # start row out of bounds
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (-1, 2))   # end row out of bounds
    with pytest.raises(ValueError):
        astar.find_path((0, 3), (2, 2))    # start column out of bounds
```

## Notes

- **Correctness**: Manhattan distance is admissible here because every traversable cell costs ≥ 1, so no step can reduce the remaining cost by more than 1. This makes A* return a minimum-cost path (with a consistent heuristic, the first time `end` is popped is already optimal).
- **Precedence rules**: bounds check → wall check → `start == end`. So `start == end` on a wall cell returns `None`, and an out-of-bounds coordinate always raises `ValueError`.
- **Cost definition**: the path cost is the sum of enter-costs of cells after the start (the start cell is free, matching the `g`-score initialization of 0). The expected costs in the tests are hand-verified optimal values (in tests 1 and 3 they match the Manhattan lower bound, so no cheaper path can exist).
- **Complexity**: roughly O((V + E) log V), where V is the number of cells.

Run with:

```bash
pytest test_astar.py -v
```