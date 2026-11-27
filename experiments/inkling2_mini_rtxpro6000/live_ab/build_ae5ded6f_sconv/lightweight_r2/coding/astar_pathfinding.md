Here's a complete implementation with a pytest suite.

## `astar.py`

```python
"""A* pathfinding on a weighted 2D grid with 4-directional movement."""

import heapq
from typing import Dict, List, Optional, Set, Tuple

Position = Tuple[int, int]


class AStarGrid:
    """A* pathfinding over a weighted 2D grid.

    Grid values represent the cost of *entering* the cell:
        0            -> impassable wall
        positive int -> cost to enter that cell
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Store a defensive copy of the grid and validate it.

        Args:
            grid: 2D list of non-negative ints (0 = wall, positive = entry cost).

        Raises:
            ValueError: if the grid is empty, ragged, or contains invalid values.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must not be empty")
        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                raise ValueError("Grid must be rectangular")
            for value in row:
                if not isinstance(value, int) or value < 0:
                    raise ValueError("Grid values must be non-negative integers")
        self._grid: List[List[int]] = [list(row) for row in grid]
        self._rows: int = len(self._grid)
        self._cols: int = len(self._grid[0])

    def find_path(self, start: Position, end: Position) -> Optional[List[Position]]:
        """Find the minimum-cost path from start to end.

        Args:
            start: (row, col) start cell (inclusive in the returned path).
            end:   (row, col) goal cell (inclusive in the returned path).

        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if no path exists.

        Raises:
            ValueError: if start or end is out of bounds or malformed.
        """
        self._validate_position(start, "start")
        self._validate_position(end, "end")

        if start == end:
            return [start]

        if self._grid[start[0]][start[1]] == 0 or self._grid[end[0]][end[1]] == 0:
            return None

        def heuristic(cell: Position) -> int:
            # Manhattan distance; admissible because every step costs >= 1.
            return abs(cell[0] - end[0]) + abs(cell[1] - end[1])

        counter = 0  # tie-breaker so heap entries never compare cells
        open_heap: List[Tuple[int, int, Position]] = [(heuristic(start), counter, start)]
        g_score: Dict[Position, int] = {start: 0}
        parent: Dict[Position, Position] = {}
        closed: Set[Position] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == end:
                return self._reconstruct_path(parent, start, end)
            closed.add(current)

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = current[0] + dr, current[1] + dc
                if not (0 <= nr < self._rows and 0 <= nc < self._cols):
                    continue
                move_cost = self._grid[nr][nc]
                if move_cost == 0:
                    continue
                neighbor = (nr, nc)
                if neighbor in closed:
                    continue
                tentative_g = g_score[current] + move_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (tentative_g + heuristic(neighbor), counter, neighbor),
                    )

        return None

    def path_cost(self, path: List[Position]) -> int:
        """Return the total entry cost of a path.

        The start cell is free; each subsequent cell costs its grid value.
        """
        return sum(self._grid[r][c] for r, c in path[1:])

    def _validate_position(self, pos: Position, name: str) -> None:
        """Raise ValueError if pos is malformed or out of bounds."""
        try:
            row, col = pos
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a (row, col) tuple, got {pos!r}") from None
        if not isinstance(row, int) or not isinstance(col, int):
            raise ValueError(f"{name} coordinates must be integers, got {pos!r}")
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise ValueError(
                f"{name} {pos} is out of bounds for a {self._rows}x{self._cols} grid"
            )

    def _reconstruct_path(
        self, parent: Dict[Position, Position], start: Position, end: Position
    ) -> List[Position]:
        """Walk parent pointers from end back to start and return the path."""
        path = [end]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return path
```

## `test_astar.py`

```python
"""Pytest suite for AStarGrid."""

import pytest

from astar import AStarGrid


def path_cost(grid, path):
    """Total cost of a path: sum of entry costs of cells after the start."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid, start, end, path):
    rows, cols = len(grid), len(grid[0])
    assert path[0] == start
    assert path[-1] == end
    for r, c in path:
        assert 0 <= r < rows
        assert 0 <= c < cols
        assert grid[r][c] > 0  # path never goes through a wall
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # 4-directional, unit steps


def test_simple_path_uniform_grid():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    start, end = (0, 0), (2, 2)

    path = AStarGrid(grid).find_path(start, end)

    assert path is not None
    assert_valid_path(grid, start, end, path)
    assert len(path) == 5  # Manhattan distance = 4 steps
    assert path_cost(grid, path) == 4  # 4 steps x cost 1


def test_path_around_obstacles():
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
    ]
    start, end = (0, 0), (2, 3)

    path = AStarGrid(grid).find_path(start, end)

    assert path is not None
    assert_valid_path(grid, start, end, path)
    assert path_cost(grid, path) == 5  # Manhattan = 5, detour costs nothing extra


def test_weighted_grid_prefers_cheap_cells():
    grid = [
        [1, 1, 100],
        [100, 1, 1],
        [1, 1, 1],
    ]
    start, end = (0, 0), (2, 2)

    path = AStarGrid(grid).find_path(start, end)

    assert path is not None
    assert_valid_path(grid, start, end, path)
    assert path_cost(grid, path) == 4  # avoids the two 100-cost cells
    assert all(grid[r][c] < 100 for r, c in path)


def test_no_path_exists_when_fully_blocked():
    grid = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
    ]
    start, end = (0, 0), (2, 2)

    assert AStarGrid(grid).find_path(start, end) is None


def test_start_equals_end():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    start = end = (1, 1)

    path = AStarGrid(grid).find_path(start, end)

    assert path == [start]
    assert path_cost(grid, path) == 0


def test_invalid_coordinates_raise_value_error():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    solver = AStarGrid(grid)

    with pytest.raises(ValueError):
        solver.find_path((3, 0), (0, 0))  # start row out of bounds
    with pytest.raises(ValueError):
        solver.find_path((0, 0), (0, 3))  # end col out of bounds
    with pytest.raises(ValueError):
        solver.find_path((-1, 0), (0, 0))  # negative coordinate
    with pytest.raises(ValueError):
        solver.find_path((0, 0), (1.5, 0))  # non-integer coordinate
```

## How it works

- **Cost convention**: entering a cell costs its grid value; the start cell is free. Total path cost = sum of entry costs over all cells after the start (this is what `path_cost` and the tests use).
- **Heuristic**: Manhattan distance. Since every step costs ≥ 1, it's admissible — and in fact consistent, so the first time `end` is popped from the heap, the path is provably optimal.
- **Search**: `heapq` open set keyed by `f = g + heuristic` (with a counter tie-breaker), a `g_score` map for lazy deletion, and a `closed` set. The Manhattan heuristic being consistent means closed nodes are never re-opened.
- **Edge cases** (in this precedence): out-of-bounds/malformed coordinates → `ValueError`; `start == end` → `[start]`; start or end on a wall → `None`; otherwise the search result.

Run the tests with:

```bash
python -m pytest test_astar.py -v
```

One note: with ties between equal-cost paths, the returned path is whichever A* pops first — the tests assert validity and optimal total cost rather than a specific route, so they pass regardless of tie-breaking.