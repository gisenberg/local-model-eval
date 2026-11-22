Here's a complete implementation. First the module:

**`astar_grid.py`**

```python
"""A* pathfinding on a weighted 2D grid.

Grid values are movement costs: 0 = impassable wall, positive int = cost to
enter that cell. Movement is 4-directional (up, down, left, right) and the
heuristic is Manhattan distance to the goal, which is admissible and
consistent for this setting (all costs >= 1).
"""

import heapq
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]


class AStarGrid:
    """A* pathfinder on a weighted 2D grid (4-directional movement).

    Attributes:
        grid: The (copied) cost grid; 0 = wall, positive int = entry cost.
        rows: Number of rows.
        cols: Number of columns.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Store the grid and validate its shape and contents.

        Args:
            grid: 2D list of non-negative ints (0 = impassable wall).

        Raises:
            ValueError: If the grid is empty, non-rectangular, or contains a
                negative or non-integer value.
        """
        if not grid or not grid[0]:
            raise ValueError("grid must be a non-empty list of non-empty rows")

        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                raise ValueError("grid must be rectangular (all rows equal length)")
            for cell in row:
                if not isinstance(cell, int) or cell < 0:
                    raise ValueError("grid values must be non-negative integers")

        self.grid: List[List[int]] = [list(row) for row in grid]
        self.rows: int = len(self.grid)
        self.cols: int = width

    def find_path(self, start: Point, end: Point) -> Optional[List[Point]]:
        """Find the minimum-cost path from ``start`` to ``end``.

        Args:
            start: (row, col) start cell.
            end: (row, col) goal cell.

        Returns:
            Shortest path as a list of (row, col) coordinates, inclusive of
            ``start`` and ``end``, or ``None`` if no path exists.

        Raises:
            ValueError: If ``start`` or ``end`` is out of bounds.

        Notes:
            A wall start or end yields ``None`` (even if start == end).
            An in-bounds, passable start == end yields ``[start]``.
        """
        # Bounds checking first — coordinates cannot be validated otherwise.
        for name, (r, c) in (("start", start), ("end", end)):
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(
                    f"{name} coordinate ({r}, {c}) is out of bounds "
                    f"for a {self.rows}x{self.cols} grid"
                )

        # Walls are impassable, so a wall start/end cannot be traversed.
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Degenerate case: already at the goal.
        if start == end:
            return [start]

        # A* open set: min-heap of (f_score, g_score, cell).
        open_heap: List[Tuple[int, int, Point]] = []
        g_score: Dict[Point, int] = {start: 0}
        parent: Dict[Point, Point] = {}
        closed: Set[Point] = set()

        heapq.heappush(open_heap, (self._heuristic(start, end), 0, start))

        while open_heap:
            _, g, current = heapq.heappop(open_heap)

            # Skip stale heap entries and already-expanded cells.
            if current in closed or g > g_score.get(current, float("inf")):
                continue
            closed.add(current)

            if current == end:
                # Reconstruct path by walking parents back to start.
                path = [current]
                node = current
                while node != start:
                    node = parent[node]
                    path.append(node)
                path.reverse()
                return path

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = current[0] + dr, current[1] + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                cost = self.grid[nr][nc]
                if cost == 0:
                    continue
                tentative_g = g + cost
                if tentative_g < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = tentative_g
                    parent[(nr, nc)] = current
                    f_score = tentative_g + self._heuristic((nr, nc), end)
                    heapq.heappush(open_heap, (f_score, tentative_g, (nr, nc)))

        return None

    @staticmethod
    def _heuristic(a: Point, b: Point) -> int:
        """Return the Manhattan distance between cells ``a`` and ``b``."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


if __name__ == "__main__":
    demo_grid = [
        [1, 2, 1, 0, 1],
        [1, 2, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 2, 1],
    ]
    demo_path = AStarGrid(demo_grid).find_path((0, 0), (4, 4))
    if demo_path:
        demo_cost = sum(demo_grid[r][c] for r, c in demo_path[1:])
        print("Path:", demo_path)
        print("Total cost:", demo_cost)
```

**`test_astar_grid.py`**

```python
"""Tests for AStarGrid."""

from typing import List, Optional, Tuple

import pytest

from astar_grid import AStarGrid

Point = Tuple[int, int]


def _total_cost(grid: List[List[int]], path: List[Point]) -> int:
    """Sum of entry costs along the path (cost to enter each cell after start)."""
    return sum(grid[r][c] for r, c in path[1:])


def _assert_path_valid(
    grid: List[List[int]], start: Point, end: Point, path: Optional[List[Point]]
) -> None:
    """Assert the path is an adjacent, wall-free walk from start to end."""
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert grid[start[0]][start[1]] > 0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "steps must be 4-directional"
        assert grid[r2][c2] > 0, "path must not enter walls"


def test_simple_path_on_uniform_grid() -> None:
    grid = [[1, 1, 1] for _ in range(3)]
    start, end = (0, 0), (2, 2)
    path = AStarGrid(grid).find_path(start, end)
    _assert_path_valid(grid, start, end, path)
    assert _total_cost(grid, path) == 4  # 4 steps x cost 1; Manhattan lower bound


def test_path_around_obstacle() -> None:
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    start, end = (0, 0), (2, 2)
    path = AStarGrid(grid).find_path(start, end)
    _assert_path_valid(grid, start, end, path)
    assert (1, 1) not in path  # must go around the wall
    assert _total_cost(grid, path) == 4  # Manhattan lower bound is 4, achieved


def test_weighted_grid_prefers_lower_cost_cells() -> None:
    grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    start, end = (0, 0), (2, 2)
    path = AStarGrid(grid).find_path(start, end)
    _assert_path_valid(grid, start, end, path)
    # Optimal: (0,0)->(0,1)->(0,2)->(1,2)->(2,2): 2+3+6+9 = 20
    assert _total_cost(grid, path) == 20


def test_no_path_exists_fully_blocked() -> None:
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is None


def test_start_equals_end() -> None:
    grid = [[1, 1, 1] for _ in range(3)]
    start = (1, 1)
    path = AStarGrid(grid).find_path(start, start)
    assert path == [start]
    assert _total_cost(grid, path) == 0


@pytest.mark.parametrize(
    "start,end",
    [
        ((-1, 0), (0, 0)),   # start row negative
        ((0, -1), (0, 0)),   # start col negative
        ((3, 0), (0, 0)),    # start row too large
        ((0, 3), (0, 0)),    # start col too large
        ((0, 0), (2, 4)),    # end col too large
        ((0, 0), (-1, 0)),   # end row negative
    ],
)
def test_invalid_coordinates_raise_value_error(start: Point, end: Point) -> None:
    grid = [[1, 1, 1] for _ in range(3)]
    with pytest.raises(ValueError):
        AStarGrid(grid).find_path(start, end)


def test_start_or_end_on_wall_returns_none() -> None:
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 1), (2, 2)) is None  # start is a wall
    assert astar.find_path((0, 0), (0, 1)) is None  # end is a wall
```

**Run the tests:**

```bash
python -m pytest test_astar_grid.py -v
```

**Design notes**

- **Check order**: bounds (raises `ValueError`) → wall (`None`) → `start == end` (`[start]`). A wall `start == end` cell returns `None`, consistent with "walls are impassable."
- **Optimality**: Manhattan distance is admissible and consistent for 4-directional movement with costs ≥ 1, so the first time `end` is popped from the heap, its path is guaranteed minimal — no re-opening of closed cells is needed (stale heap entries are skipped defensively).
- **Open set**: `heapq` min-heap keyed on `(f_score, g_score, cell)` with lazy deletion (`g > g_score[cell]` skip), which is standard heap-based A* and avoids storing "open" membership in a separate set.
- **Optimality in tests**: the expected costs (4, 4, 20) are either Manhattan lower bounds or hand-verified minima over all candidate paths, so an exact match is a genuine optimality check, not just a sanity check.