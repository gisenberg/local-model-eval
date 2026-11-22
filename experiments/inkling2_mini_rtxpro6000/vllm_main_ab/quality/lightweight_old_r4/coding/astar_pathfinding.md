Here's a complete implementation with the tests, in two files:

## `a_star.py`

```python
"""A* pathfinding on a weighted 2D grid."""

import heapq
from typing import Dict, Iterator, List, Optional, Set, Tuple

Coord = Tuple[int, int]


class AStarGrid:
    """A* shortest-path finder over a weighted 2D grid.

    Grid values: 0 = impassable wall, positive int = cost to enter that cell.
    Movement is 4-directional (up/down/left/right).
    Heuristic: Manhattan distance (admissible because every cell costs >= 1).
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Store and validate the grid.

        Args:
            grid: Rectangular 2D list of non-negative ints (0 = wall).

        Raises:
            ValueError: If the grid is empty, ragged, or contains invalid values.
        """
        if not isinstance(grid, list) or len(grid) == 0 or not isinstance(grid[0], list) or len(grid[0]) == 0:
            raise ValueError("grid must be a non-empty rectangular 2D list")
        cols = len(grid[0])
        for row in grid:
            if not isinstance(row, list) or len(row) != cols:
                raise ValueError("grid must be rectangular")
            for value in row:
                if not isinstance(value, int) or value < 0:
                    raise ValueError("grid values must be non-negative integers (0 = wall)")
        self.grid: List[List[int]] = [list(row) for row in grid]
        self.rows: int = len(self.grid)
        self.cols: int = len(self.grid[0])

    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest path from start to end.

        Args:
            start: (row, col) start cell.
            end: (row, col) target cell.

        Returns:
            The list of (row, col) coordinates from start to end inclusive,
            or None if no path exists. Path cost = sum of cell costs for all
            cells after the start cell (the start cell contributes 0).

        Raises:
            ValueError: If start or end is out of bounds.
        """
        r_start, c_start = start
        r_end, c_end = end
        start_node: Coord = (r_start, c_start)
        end_node: Coord = (r_end, c_end)

        # Bounds check first (out-of-bounds coordinates are always an error).
        if not (0 <= r_start < self.rows and 0 <= c_start < self.cols):
            raise ValueError(f"start {start} is out of bounds")
        if not (0 <= r_end < self.rows and 0 <= c_end < self.cols):
            raise ValueError(f"end {end} is out of bounds")

        # Wall endpoints cannot be part of any valid path.
        if self.grid[r_start][c_start] == 0 or self.grid[r_end][c_end] == 0:
            return None

        # Trivial case: no movement required.
        if start_node == end_node:
            return [start_node]

        # A* search. Heap entries: (f_cost, unique_tiebreak, node).
        tiebreak = 0
        open_heap: List[Tuple[float, int, Coord]] = [
            (self._heuristic(start_node, end_node), tiebreak, start_node)
        ]
        tiebreak += 1
        g_score: Dict[Coord, float] = {start_node: 0}
        came_from: Dict[Coord, Coord] = {}
        closed: Set[Coord] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)

            if current == end_node:
                return self._reconstruct(came_from, start_node, end_node)

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_cost = tentative_g + self._heuristic(neighbor, end_node)
                    heapq.heappush(open_heap, (f_cost, tiebreak, neighbor))
                    tiebreak += 1

        return None

    def _neighbors(self, cell: Coord) -> Iterator[Coord]:
        """Yield in-bounds, passable 4-directional neighbors of cell."""
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] > 0:
                yield (nr, nc)

    @staticmethod
    def _heuristic(a: Coord, b: Coord) -> int:
        """Manhattan distance between two cells (admissible, costs >= 1)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _reconstruct(came_from: Dict[Coord, Coord], start: Coord, end: Coord) -> List[Coord]:
        """Walk came_from backwards from end to start and return the path."""
        path = [end]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path
```

## `test_a_star.py`

```python
"""Pytest tests for A* pathfinding."""

from typing import List, Tuple

import pytest

from a_star import AStarGrid

Coord = Tuple[int, int]


def total_cost(grid: List[List[int]], path: List[Coord]) -> int:
    """Sum of entry costs for every cell after the start cell."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(
    grid: List[List[int]], path: List[Coord], start: Coord, end: Coord
) -> None:
    """Assert endpoints, bounds, walls, and 4-directional adjacency."""
    rows, cols = len(grid), len(grid[0])
    assert path[0] == start
    assert path[-1] == end
    for r, c in path:
        assert 0 <= r < rows
        assert 0 <= c < cols
        assert grid[r][c] > 0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_simple_path_uniform_grid() -> None:
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (2, 2))
    assert total_cost(grid, path) == 4  # Manhattan distance, optimal


def test_path_around_obstacles() -> None:
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (0, 2))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (0, 2))
    assert total_cost(grid, path) == 6  # must detour along the bottom row


def test_weighted_grid_prefers_cheaper_route() -> None:
    grid = [
        [1, 50, 1],
        [1, 50, 1],
        [1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (0, 2))
    assert path is not None
    assert_valid_path(grid, path, (0, 0), (0, 2))
    assert total_cost(grid, path) == 6  # bottom row beats the 50-cost middle column
    assert (2, 1) in path


def test_no_path_exists_fully_blocked() -> None:
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    assert AStarGrid(grid).find_path((0, 0), (0, 2)) is None


def test_start_equals_end() -> None:
    grid = [[1, 1], [1, 1]]
    path = AStarGrid(grid).find_path((1, 1), (1, 1))
    assert path == [(1, 1)]
    assert_valid_path(grid, path, (1, 1), (1, 1))
    assert total_cost(grid, path) == 0


def test_invalid_coordinates_raise_value_error() -> None:
    astar = AStarGrid([[1, 1, 1], [1, 1, 1]])
    with pytest.raises(ValueError):
        astar.find_path((2, 0), (0, 0))   # start row out of bounds
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 3))   # end col out of bounds
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))  # negative coordinate
```

Run with: `pytest test_a_star.py -v`

## Design notes

- **Optimality**: Manhattan distance is an admissible heuristic here since every cell costs ≥ 1 (at least 1 per step), so A* with a min-heap (`heapq`) is guaranteed to return a minimum-cost path. The `(f_cost, tiebreak, node)` heap guarantees deterministic ordering even when `f` ties.
- **Cost convention**: total path cost = sum of entry costs for cells *after* the start (start contributes 0). A single-cell path costs 0.
- **Edge-case precedence**: bounds (raise `ValueError`) → wall endpoints (`None`) → `start == end` (`[start]`). If you'd rather return `[start]` even for a wall cell, move the `start == end` check above the wall check.
- **Grid validation** in `__init__` rejects empty/ragged grids and negative or non-integer values, and keeps a defensive copy of the input so external mutation can't corrupt the search.
- The weighted test uses a uniquely optimal route (through the bottom row), so the cost assertion of 6 confirms optimality, not just reachability.