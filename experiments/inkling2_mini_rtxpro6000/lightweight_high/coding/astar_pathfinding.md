Here's a complete implementation. The key design point: since every passable cell costs ≥ 1, Manhattan distance is an admissible (and consistent) heuristic, so A* with a closed set is guaranteed optimal.

## `astar.py`

```python
"""A* pathfinding on a weighted 2D grid.

Grid cells store the movement cost of *entering* the cell:
0 = impassable wall, positive int = cost to enter.
"""

from __future__ import annotations

import heapq
from typing import Dict, Iterator, List, Optional, Set, Tuple

Coord = Tuple[int, int]


class AStarGrid:
    """A* solver over a weighted grid (4-directional movement)."""

    def __init__(self, grid: List[List[int]]) -> None:
        """Validate and store the grid.

        Args:
            grid: 2D list of ints. Cell value = cost to enter; 0 = wall.
                  Must be rectangular and non-empty.

        Raises:
            ValueError: If the grid is empty, ragged, or contains a
                        non-integer or negative cell value.
        """
        if not isinstance(grid, list) or not grid:
            raise ValueError("Grid must be a non-empty list of rows.")
        first = grid[0]
        if not isinstance(first, list) or not first:
            raise ValueError("Grid rows must be non-empty lists.")
        self._cols = len(first)
        for row in grid:
            if not isinstance(row, list) or len(row) != self._cols:
                raise ValueError("Grid must be rectangular.")
            for cell in row:
                if isinstance(cell, bool) or not isinstance(cell, int):
                    raise ValueError("Cell costs must be integers.")
                if cell < 0:
                    raise ValueError("Cell costs must be non-negative (0 is a wall).")
        self._grid: List[List[int]] = [list(row) for row in grid]
        self._rows = len(grid)

    @property
    def rows(self) -> int:
        """Number of rows in the grid."""
        return self._rows

    @property
    def cols(self) -> int:
        """Number of columns in the grid."""
        return self._cols

    def find_path(self, start: Coord, end: Coord) -> Optional[List[Coord]]:
        """Find the lowest-cost 4-directional path from start to end.

        Args:
            start: (row, col) of the starting cell.
            end: (row, col) of the destination cell.

        Returns:
            List of (row, col) coordinates from start to end inclusive, or
            None if either cell is a wall or no path exists.

        Raises:
            ValueError: If start or end is malformed or out of bounds.

        Notes:
            If start == end on a passable cell, returns [start].
            Wall checks take precedence over the start == end shortcut.
        """
        self._validate_coord(start, "start")
        self._validate_coord(end, "end")

        if self._grid[start[0]][start[1]] == 0 or self._grid[end[0]][end[1]] == 0:
            return None

        if start == end:
            return [start]

        open_heap: List[Tuple[int, int, Coord]] = []
        counter = 0
        heapq.heappush(open_heap, (self._heuristic(start, end), counter, start))
        counter += 1

        g_score: Dict[Coord, int] = {start: 0}
        parent: Dict[Coord, Coord] = {}
        closed: Set[Coord] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)

            if current == end:
                return self._reconstruct_path(parent, start, end)

            for neighbor in self._neighbors(current):
                if neighbor in closed:
                    continue
                tentative_g = g_score[current] + self._grid[neighbor[0]][neighbor[1]]
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    heapq.heappush(
                        open_heap,
                        (tentative_g + self._heuristic(neighbor, end), counter, neighbor),
                    )
                    counter += 1

        return None

    def path_cost(self, path: List[Coord]) -> int:
        """Return the total movement cost of a path.

        The start cell is free; the cost of each subsequent cell
        (the cost of entering it) is summed.
        """
        return sum(self._grid[r][c] for r, c in path[1:])

    def _validate_coord(self, coord: Coord, name: str) -> None:
        """Raise ValueError if coord is malformed or out of bounds."""
        try:
            row, col = coord
        except Exception as exc:
            raise ValueError(f"{name} must be a (row, col) pair.") from exc
        if (
            isinstance(row, bool) or isinstance(col, bool)
            or not isinstance(row, int) or not isinstance(col, int)
        ):
            raise ValueError(f"{name} must contain integer row and column.")
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise ValueError(f"{name} {coord!r} is out of bounds.")

    def _neighbors(self, cell: Coord) -> Iterator[Coord]:
        """Yield the passable cells directly above/below/left/right of cell."""
        row, col = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < self._rows and 0 <= nc < self._cols and self._grid[nr][nc] != 0:
                yield (nr, nc)

    def _heuristic(self, a: Coord, b: Coord) -> int:
        """Manhattan distance; admissible since every cell costs >= 1."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(
        self, parent: Dict[Coord, Coord], start: Coord, end: Coord
    ) -> List[Coord]:
        """Walk the parent chain from end back to start."""
        path = [end]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return path
```

## `test_astar.py`

```python
"""Tests for the A* grid pathfinding implementation."""

import pytest

from astar import AStarGrid


def assert_valid_path(grid, path, start, end):
    """Assert structural validity of a path through grid."""
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert len(path) == len(set(path)), "path visits a cell twice"
    for r, c in path:
        assert 0 <= r < len(grid) and 0 <= c < len(grid[0])
        assert grid[r][c] != 0, f"path enters wall at {(r, c)}"
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, f"non-adjacent {(r1, c1)} -> {(r2, c2)}"


def total_cost(grid, path):
    return sum(grid[r][c] for r, c in path[1:])


def test_simple_path_uniform_grid():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    start, end = (0, 0), (2, 2)
    path = AStarGrid(grid).find_path(start, end)
    assert_valid_path(grid, path, start, end)
    # Optimal: 2 down + 2 right = 4 moves, cost 1 each
    assert total_cost(grid, path) == 4
    assert len(path) == 5


def test_path_around_obstacles():
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    start, end = (0, 0), (2, 4)
    path = AStarGrid(grid).find_path(start, end)
    assert_valid_path(grid, path, start, end)
    # Optimal detour around the wall block: 6 moves
    assert total_cost(grid, path) == 6


def test_weighted_grid_prefers_low_cost_cells():
    grid = [
        [1, 9, 9, 1],
        [1, 9, 9, 1],
        [1, 1, 1, 1],
    ]
    start, end = (0, 0), (0, 3)
    path = AStarGrid(grid).find_path(start, end)
    assert_valid_path(grid, path, start, end)
    # Straight line costs 9+9+1 = 19; the low-cost detour costs 7
    assert total_cost(grid, path) == 7
    assert (0, 1) not in path and (0, 2) not in path


def test_no_path_exists():
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
    ]
    start, end = (0, 0), (2, 2)
    assert AStarGrid(grid).find_path(start, end) is None


def test_start_equals_end():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    start = (1, 1)
    path = AStarGrid(grid).find_path(start, start)
    assert path == [(1, 1)]
    assert total_cost(grid, path) == 0


def test_invalid_coordinates():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (3, 0))      # row out of bounds
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (2, 2))     # negative row
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 3))      # column out of bounds
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1.5, 1))    # non-integer coordinate


def test_wall_start_or_end_returns_none():
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 1), (2, 2)) is None  # wall start
    assert astar.find_path((2, 2), (0, 1)) is None  # wall end
```

Run with:

```bash
pytest test_astar.py -v
```

## Notes on the design

- **Admissibility/consistency**: every passable cell costs ≥ 1, and each move changes Manhattan distance by at most 1, so the heuristic never overestimates and is consistent — the first time a node is popped from the heap it holds the optimal `g` value, which is what guarantees optimality with the closed set.
- **Cost convention**: the start cell is free; the total cost of a path is the sum of the cells entered after start (documented in `path_cost` and the tests).
- **Precedence**: bounds are validated first (`ValueError`), then walls (`None`), then the `start == end` shortcut — so a wall start/end returns `None` even if `start == end`.
- **Heap**: entries are `(f, counter, coord)`; the monotonic `counter` guarantees a deterministic tie-break without comparing coordinates.
- The seventh test (`test_wall_start_or_end_returns_none`) covers the wall edge case explicitly alongside the six requested categories.