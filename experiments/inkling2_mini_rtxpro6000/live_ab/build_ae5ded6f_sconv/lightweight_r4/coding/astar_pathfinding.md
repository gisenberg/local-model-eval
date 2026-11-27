Here's a complete implementation with tests. I split it into two files: `astar.py` (the implementation) and `test_astar.py` (the tests).

## `astar.py`

```python
"""A* pathfinding on a weighted 2D grid."""

import heapq
from typing import Dict, Iterator, List, Optional, Set, Tuple


class AStarGrid:
    """Finds shortest paths on a weighted 2D grid using A* search."""

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid.

        Args:
            grid: 2D list of non-negative ints. 0 means impassable wall;
                  any positive int is the cost to enter that cell.

        Raises:
            ValueError: If the grid is empty, non-rectangular, or contains
                invalid (negative or non-int) values.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must contain at least one row and one column")
        width = len(grid[0])
        if any(len(row) != width for row in grid):
            raise ValueError("Grid must be rectangular (all rows equal length)")
        for row in grid:
            for value in row:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        "Grid values must be non-negative integers (0 = wall)"
                    )
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest path from start to end, inclusive.

        Args:
            start: (row, col) of the start cell.
            end: (row, col) of the goal cell.

        Returns:
            List of (row, col) coordinates from start to end inclusive, or
            None if no path exists or either endpoint is a wall.

        Raises:
            ValueError: If start or end is malformed or out of bounds.
        """
        self._check_coordinate("start", start)
        self._check_coordinate("end", end)

        # Wall endpoints (even when start == end) → no path.
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        if start == end:
            return [start]

        counter = 0
        open_heap: List[Tuple[float, int, Tuple[int, int]]] = [
            (self._heuristic(start, end), counter, start)
        ]
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed: Set[Tuple[int, int]] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            # Skip stale entries already finalized via a better path.
            if current in closed:
                continue
            closed.add(current)

            if current == end:
                return self._reconstruct_path(came_from, start, end)

            for neighbor in self._passable_neighbors(current):
                if neighbor in closed:
                    continue
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_heap, (f_score, counter, neighbor))

        return None

    def _check_coordinate(self, name: str, coord: Tuple[int, int]) -> None:
        """Validate that a coordinate is a (row, col) pair inside the grid.

        Raises:
            ValueError: If coord is malformed or out of bounds.
        """
        try:
            row, col = coord
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} coordinate must be a (row, col) tuple, got {coord!r}"
            ) from None
        if (
            isinstance(row, bool)
            or isinstance(col, bool)
            or not isinstance(row, int)
            or not isinstance(col, int)
        ):
            raise ValueError(
                f"{name} coordinate must contain ints, got {coord!r}"
            )
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"{name} coordinate {coord!r} is out of bounds "
                f"for a {self.rows}x{self.cols} grid"
            )

    def _passable_neighbors(
        self, node: Tuple[int, int]
    ) -> Iterator[Tuple[int, int]]:
        """Yield the 4-directional passable (non-wall) neighbors of node."""
        row, col = node
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < self.rows
                and 0 <= nc < self.cols
                and self.grid[nr][nc] != 0
            ):
                yield (nr, nc)

    def _heuristic(self, node: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Manhattan distance from node to end.

        Admissible (and consistent) because every passable cell costs >= 1.
        """
        return abs(node[0] - end[0]) + abs(node[1] - end[1])

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Walk the came_from chain backwards from end to start."""
        path = [end]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path
```

## `test_astar.py`

```python
"""Pytest tests for AStarGrid."""

import pytest

from astar import AStarGrid


def path_cost(grid, path):
    """Total cost of entering each cell after the start cell."""
    return sum(grid[r][c] for r, c in path[1:])


def assert_valid_path(grid, start, end, path):
    """Assert the path connects start to end with valid 4-directional steps."""
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    for r, c in path:
        assert grid[r][c] > 0  # every cell is passable
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # single 4-directional step


def test_simple_path_uniform_grid():
    grid = [[1] * 5 for _ in range(5)]
    start, end = (0, 0), (4, 4)
    path = AStarGrid(grid).find_path(start, end)

    assert_valid_path(grid, start, end, path)
    assert path_cost(grid, path) == 8  # equals Manhattan distance


def test_path_around_obstacles():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    start, end = (0, 0), (0, 2)
    path = AStarGrid(grid).find_path(start, end)

    assert_valid_path(grid, start, end, path)
    # Forced detour down to row 2 and back: 6 steps, 6 cells entered
    assert path_cost(grid, path) == 6


def test_weighted_grid_prefers_cheaper_cells():
    grid = [
        [1, 5, 1],
        [1, 5, 1],
        [1, 1, 1],
    ]
    start, end = (0, 0), (2, 2)
    path = AStarGrid(grid).find_path(start, end)

    assert_valid_path(grid, start, end, path)
    # Straight-down column 0 then across: 1+1+1+1 = 4 (provably minimal)
    assert path_cost(grid, path) == 4
    assert all(grid[r][c] != 5 for r, c in path[1:])  # avoids expensive column


def test_no_path_exists():
    grid = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
    ]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is None


def test_start_equals_end():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    path = AStarGrid(grid).find_path((1, 1), (1, 1))
    assert path == [(1, 1)]
    assert path_cost(grid, path) == 0


def test_out_of_bounds_coordinates_raise_value_error():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (2, 2))  # start above grid
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (3, 0))  # end below grid
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 99))  # end far right


def test_wall_endpoints_return_none():
    grid = [
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 1), (2, 2)) is None  # start is a wall
    assert astar.find_path((0, 0), (1, 2)) is None  # end is a wall
    assert astar.find_path((0, 1), (0, 1)) is None  # same wall cell, both ends
```

That's the 6 required tests (uniform, obstacles, weighted, no-path, start==end, invalid coords) plus one bonus test covering the wall-endpoint edge case, which the spec also requires handling.

## Design notes

- **Heuristic**: Manhattan distance is admissible here because every passable cell costs ≥ 1, so the heuristic never overestimates the remaining true cost. It's also *consistent* (moving toward the goal decreases it by exactly 1 while costing ≥ 1), which guarantees the first time we pop a node it's optimal — so the `closed` set is safe and stale heap entries can be skipped.
- **Open set**: `heapq` with entries `(f_score, counter, node)`. The counter tiebreaker guarantees heap comparison never reaches the `node` tuple, and keeps the heap well-ordered.
- **Edge-case precedence**: bounds check → wall check → start == end. This means a wall cell where `start == end` returns `None` (wall rule) rather than `[start]` (trivially-optimal rule), matching the spec's "start or end is a wall → None".
- **Optimality**: asserted in tests by comparing the path's total cost (sum of cell costs after the start) against hand-verified minima — e.g. 8 for the uniform grid, 6 for the obstacle detour (exceeds Manhattan, proving it routed around the wall), and 4 for the weighted grid (beats any route through the expensive column, which costs ≥ 8).
- **Complexity**: O(N log N) where N is the number of reachable cells (heap operations), with O(N) space for `g_score`, `came_from`, and `closed`.

## Running the tests

```bash
pip install pytest
pytest test_astar.py -v
```

Expected: 7 passed. Note that exact optimal *paths* may differ across valid solutions (the tests deliberately assert total cost and validity rather than one specific route, except for the trivial start == end case), which is the right way to test a pathfinding algorithm.