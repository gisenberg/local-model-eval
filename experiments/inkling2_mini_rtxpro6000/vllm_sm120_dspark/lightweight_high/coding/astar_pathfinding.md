# A* Pathfinding Implementation

## `astar.py`

```python
"""A* pathfinding on a weighted 2D grid (4-directional movement)."""

import heapq
from typing import Dict, List, Optional, Tuple

Coord = Tuple[int, int]


class AStarGrid:
    """A* pathfinding over a weighted 2D grid.

    Grid values:
        0             -> impassable wall
        positive int  -> cost to enter that cell

    Movement is 4-directional (up, down, left, right).
    Heuristic is Manhattan distance, which is admissible (and consistent)
    because every passable cell costs >= 1.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Validate the grid and store a defensive copy.

        Raises:
            ValueError: if the grid is empty, non-rectangular, or contains
                invalid values.
        """
        if not isinstance(grid, (list, tuple)) or len(grid) == 0:
            raise ValueError("grid must be a non-empty list of rows")
        if not isinstance(grid[0], (list, tuple)) or len(grid[0]) == 0:
            raise ValueError("grid must contain at least one column")
        cols = len(grid[0])
        for row in grid:
            if not isinstance(row, (list, tuple)) or len(row) != cols:
                raise ValueError("grid must be rectangular")
            for v in row:
                if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    raise ValueError("grid values must be non-negative integers")
        self.grid: List[List[int]] = [list(row) for row in grid]
        self.rows: int = len(self.grid)
        self.cols: int = cols

    def find_path(self, start: Coord, end: Coord) -> Optional[List[Coord]]:
        """Find the minimum-cost path from start to end (inclusive).

        Args:
            start: (row, col) start coordinate.
            end: (row, col) end coordinate.

        Returns:
            List of (row, col) coordinates from start to end inclusive, or
            None if no path exists (or start/end is a wall).

        Raises:
            ValueError: if start or end is malformed or out of bounds.
        """
        self._validate_coord(start, "start")
        self._validate_coord(end, "end")

        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        if start == end:
            return [start]

        open_heap: List[Tuple[int, int, Coord]] = []
        counter = 0  # tie-breaker so heap entries are always unique
        g_score: Dict[Coord, int] = {start: 0}
        came_from: Dict[Coord, Coord] = {}
        heapq.heappush(open_heap, (self.heuristic(start, end), counter, start))

        while open_heap:
            f, _, current = heapq.heappop(open_heap)

            if current == end:
                return self._reconstruct_path(came_from, start, end)
            # Stale heap entry (a better path to `current` was found since).
            if f > g_score[current] + self.heuristic(current, end):
                continue

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (tentative_g + self.heuristic(neighbor, end), counter, neighbor),
                    )

        return None

    def heuristic(self, a: Coord, b: Coord) -> int:
        """Return the Manhattan distance between two coordinates."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, node: Coord) -> List[Coord]:
        """Return the passable 4-connected neighbors of `node`."""
        r, c = node
        neighbors: List[Coord] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] > 0:
                neighbors.append((nr, nc))
        return neighbors

    def _validate_coord(self, coord: Coord, name: str) -> None:
        """Raise ValueError if `coord` is malformed or out of bounds."""
        if (
            not isinstance(coord, (tuple, list))
            or len(coord) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in coord)
        ):
            raise ValueError(f"{name} must be a (row, col) tuple of integers, got {coord!r}")
        r, c = coord
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(
                f"{name} {coord} is out of bounds for a {self.rows}x{self.cols} grid"
            )

    def _reconstruct_path(
        self, came_from: Dict[Coord, Coord], start: Coord, end: Coord
    ) -> List[Coord]:
        """Rebuild the path from start to end by walking came_from backwards."""
        path = [end]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path
```

## `test_astar.py`

```python
"""Tests for AStarGrid."""
import pytest

from astar import AStarGrid


def assert_valid_path(path, grid, start, end):
    """Assert the path is valid: endpoints match, 4-connected, passable cells."""
    rows, cols = len(grid), len(grid[0])
    assert path[0] == start
    assert path[-1] == end
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # 4-directional move only
    for r, c in path:
        assert 0 <= r < rows and 0 <= c < cols
        assert grid[r][c] > 0  # never enters a wall


def path_cost(path, grid):
    """Total cost of entering each cell in the path (start cell excluded)."""
    return sum(grid[r][c] for r, c in path[1:])


def test_simple_path_uniform_grid():
    grid = [[1] * 5 for _ in range(5)]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (4, 4))
    assert path is not None
    assert_valid_path(path, grid, (0, 0), (4, 4))
    # Manhattan distance = 8 steps at cost 1 each -> optimal.
    assert path_cost(path, grid) == 8
    assert len(path) == 9


def test_path_around_obstacles():
    grid = [
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 4))
    assert path is not None
    assert_valid_path(path, grid, (0, 0), (0, 4))
    # Wall column 2 blocks rows 0-3; the only crossing point is row 4,
    # so the path must go down, across, and back up: 4 + 4 + 4 = 12 steps.
    assert path_cost(path, grid) == 12
    assert any(r == 4 for r, _ in path)  # must detour through row 4


def test_weighted_grid_prefers_cheaper_cells():
    grid = [
        [1, 5, 1],
        [1, 1, 1],
        [1, 5, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    assert_valid_path(path, grid, (0, 0), (0, 2))
    # Cheap detour via row 1 costs 1+1+1+1 = 4; the direct row costs 5+1 = 6.
    assert path_cost(path, grid) == 4
    assert (1, 1) in path
    assert (0, 1) not in path  # avoids the expensive cell


def test_no_path_exists():
    grid = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None


def test_start_equals_end():
    grid = [[1] * 3 for _ in range(3)]
    astar = AStarGrid(grid)
    assert astar.find_path((1, 1), (1, 1)) == [(1, 1)]


def test_invalid_coordinates():
    grid = [[1] * 3 for _ in range(3)]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (2, 2))   # negative row
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (3, 0))    # row out of range
    with pytest.raises(ValueError):
        astar.find_path((0, 2), (1, 5))    # column out of range
    with pytest.raises(ValueError):
        astar.find_path((0, 0, 0), (2, 2))  # malformed coordinate


def test_wall_start_or_end_returns_none():
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 1), (2, 2)) is None  # start is a wall
    assert astar.find_path((0, 0), (0, 1)) is None  # end is a wall
```

## Running the tests

```bash
pytest test_astar.py -v
```

## Notes on the design

- **Optimality**: Manhattan distance is admissible (and consistent) here because every passable cell costs ≥ 1, so A* with a min-heap returns a minimum-cost path. The heuristic deliberately ignores cell weights, so paths are optimal but not necessarily the shortest in *step count*.
- **Stale entries**: the heap may contain superseded entries; they're detected via `f > g_score[current] + heuristic(current, end)` and skipped, which avoids an explicit closed set while keeping the complexity near O(E log E).
- **Edge cases** are handled in order: bounds validation (raises `ValueError`) → wall check (returns `None`) → `start == end` (returns `[start]`). So a wall at the start/end cell returns `None` even when `start == end`.
- **Tie-breaking**: a counter is appended to heap entries so tuple comparison never falls through to coordinates.
- The `counter`-keyed `g_score`/`came_from` maps guarantee the reconstructed path's total cost equals the optimal `g_score[end]`, which is what the tests assert.