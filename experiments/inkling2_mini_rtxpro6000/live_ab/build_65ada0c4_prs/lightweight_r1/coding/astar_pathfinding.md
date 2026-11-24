# A* Pathfinding on a Weighted 2D Grid

Here's a complete implementation in two files — `a_star.py` and `test_a_star.py`.

## `a_star.py`

```python
"""A* pathfinding on a weighted 2D grid (4-directional movement)."""

import heapq
import math
from typing import Dict, Iterator, List, Optional, Set, Tuple

Point = Tuple[int, int]


class AStarGrid:
    """Finds shortest paths on a weighted 2D grid.

    Grid values: 0 = impassable wall, positive int = cost of entering that cell.
    Movement is 4-directional (up/down/left/right); Manhattan distance is used
    as the heuristic. Manhattan is admissible here because every cell costs
    at least 1 to enter, so the step-count heuristic never overestimates the
    true remaining cost.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid.

        Args:
            grid: 2D list where each value is 0 (wall) or a positive int (cost).

        Raises:
            ValueError: If the grid is empty, ragged, or contains invalid values.
        """
        if not isinstance(grid, list) or len(grid) == 0:
            raise ValueError("Grid must be a non-empty list of rows.")

        self.rows = len(grid)
        self.cols = len(grid[0])
        if self.cols == 0:
            raise ValueError("Grid must have at least one column.")

        for row in grid:
            if not isinstance(row, list) or len(row) != self.cols:
                raise ValueError("Grid must be rectangular (all rows same length).")

        for r in range(self.rows):
            for c in range(self.cols):
                value = grid[r][c]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("Grid values must be integers (0 = wall, positive = cost).")
                if value < 0:
                    raise ValueError("Grid values must be >= 0 (0 = wall, positive = cost).")

        self.grid = grid

    def find_path(self, start: Point, end: Point) -> Optional[List[Point]]:
        """Find the shortest path from start to end.

        Args:
            start: (row, col) start cell.
            end: (row, col) destination cell.

        Returns:
            Path as a list of (row, col) coordinates from start to end inclusive,
            or None if no path exists (wall endpoints or blocked grid).

        Raises:
            ValueError: If start or end is out of bounds or malformed.
        """
        start = self._validate_point(start, "start")
        end = self._validate_point(end, "end")

        # Endpoints must be passable cells.
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Trivial case: no movement needed.
        if start == end:
            return [start]

        open_heap: List[Tuple[float, int, Point]] = []
        counter = 0
        heapq.heappush(open_heap, (float(self._heuristic(start, end)), counter, start))

        g_score: Dict[Point, int] = {start: 0}
        f_score: Dict[Point, float] = {start: float(self._heuristic(start, end))}
        came_from: Dict[Point, Point] = {}
        closed: Set[Point] = set()

        while open_heap:
            f, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if f > f_score.get(current, math.inf):
                continue  # stale heap entry
            closed.add(current)

            if current == end:
                return self._reconstruct_path(came_from, start, end)

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                if tentative_g < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, neighbor))

        return None  # open set exhausted: no path exists

    def _heuristic(self, a: Point, b: Point) -> int:
        """Manhattan distance between two points (admissible lower bound on remaining cost)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, node: Point) -> Iterator[Point]:
        """Yield passable (in-bounds, non-wall) 4-directional neighbors of node."""
        r, c = node
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] > 0:
                yield (nr, nc)

    def _reconstruct_path(self, came_from: Dict[Point, Point], start: Point, end: Point) -> List[Point]:
        """Rebuild the path from start to end by walking came_from backwards."""
        path = [end]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def _validate_point(self, point: Point, name: str) -> Point:
        """Validate that a point is a well-formed, in-bounds (row, col) pair.

        Raises:
            ValueError: If the point is malformed or out of bounds.
        """
        try:
            r, c = point
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a (row, col) tuple, got {point!r}.") from None

        if isinstance(r, bool) or isinstance(c, bool) or not isinstance(r, int) or not isinstance(c, int):
            raise ValueError(f"{name} must be (row, col) integers, got {point!r}.")
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"{name} {point} is out of bounds for a {self.rows}x{self.cols} grid.")
        return (r, c)
```

## `test_a_star.py`

```python
"""Tests for the A* grid pathfinding implementation."""

import pytest

from a_star import AStarGrid


def total_cost(grid, path):
    """Total movement cost along a path (cost of entering each cell after start)."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid, start, end, path):
    """Assert the path is well-formed: in bounds, adjacent, wall-free, start/end correct."""
    rows, cols = len(grid), len(grid[0])
    assert path[0] == start
    assert path[-1] == end
    assert grid[start[0]][start[1]] > 0
    for r, c in path:
        assert 0 <= r < rows and 0 <= c < cols
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "Moves must be 4-directional and adjacent"
        assert grid[r2][c2] > 0, "Path must not enter wall cells"


def test_simple_path_on_uniform_grid():
    grid = [[1] * 5 for _ in range(5)]
    astar = AStarGrid(grid)
    start, end = (0, 0), (4, 4)
    path = astar.find_path(start, end)
    assert path is not None
    assert_valid_path(grid, start, end, path)
    assert len(path) == 9
    # Optimal: Manhattan distance = 8 steps at min cost 1 each
    assert total_cost(grid, path) == 8


def test_path_around_obstacles():
    grid = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (4, 0), (0, 4)
    path = astar.find_path(start, end)
    assert path is not None
    assert_valid_path(grid, start, end, path)
    # Must cross column 2 at the only gap (row 0 or 1)
    assert any(c == 2 and r in (0, 1) for r, c in path)
    # Optimal: Manhattan lower bound = 8 steps at cost 1 each
    assert total_cost(grid, path) == 8


def test_weighted_grid_prefers_lower_cost_cells():
    grid = [
        [1, 5, 1],
        [1, 5, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert_valid_path(grid, start, end, path)
    # Optimal: descend column 0 then cross the bottom row (cost 4, not 8+ via the cost-5 cells)
    assert total_cost(grid, path) == 4
    assert all(grid[r][c] != 5 for r, c in path)


def test_no_path_exists_fully_blocked():
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None


def test_start_equals_end():
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]
    assert total_cost(grid, path) == 0


def test_invalid_coordinates_raise_value_error():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (2, 2))   # start above grid
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (3, 0))    # end below grid
    with pytest.raises(ValueError):
        astar.find_path((0, 3), (2, 2))    # start right of grid
    with pytest.raises(ValueError):
        astar.find_path((0.5, 0), (2, 2))  # non-integer coordinate


def test_wall_endpoint_returns_none():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None
    assert astar.find_path((1, 0), (0, 0)) is None
    assert astar.find_path((0, 0), (0, 0)) is None  # wall endpoint wins over start == end
```

## Design notes

- **Optimality**: Manhattan distance is admissible (every cell costs ≥ 1, so the step-count heuristic never overestimates remaining cost) and consistent (moving one step changes Manhattan by exactly ±1 ≤ minimum edge cost), so the first time `end` is popped from the heap, its cost is provably minimal. Stale heap entries are skipped via the `f_score` check and a `closed` set.
- **Tie-breaking**: a monotonically increasing counter is stored alongside `f` in the heap so entries never need to be compared by coordinates.
- **Edge-case precedence**: bounds/malformed checks (`ValueError`) → wall endpoints (`None`) → `start == end` (`[start]`). This means `start == end` on a wall cell returns `None`, which is consistent with "wall endpoints → None".
- **Cost definition**: total path cost = sum of the cost of *entering* each cell after the start.

Run the tests with:

```bash
python -m pytest test_a_star.py -v
```

This should report all 7 tests passing (the 6 requested plus one extra test for the wall-endpoint edge case).