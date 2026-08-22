

```python
"""A* pathfinding on a weighted 2D grid."""

import heapq
from typing import List, Tuple, Optional, Dict, Set


class AStarGrid:
    """Weighted 2D grid with A* pathfinding support.

    Grid values represent movement cost: 0 = impassable wall,
    positive int = cost to enter that cell.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid.

        Args:
            grid: 2D list where 0 is a wall and positive ints are entry costs.
        """
        self.grid: List[List[int]] = grid
        self.rows: int = len(grid)
        self.cols: int = len(grid[0]) if self.rows > 0 else 0

    def _in_bounds(self, row: int, col: int) -> bool:
        """Check whether (row, col) is within the grid dimensions."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two grid coordinates."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the minimum-cost path from start to end using A*.

        Args:
            start: (row, col) of the starting cell.
            end: (row, col) of the target cell.

        Returns:
            List of (row, col) tuples from start to end inclusive,
            or None if no valid path exists.

        Raises:
            ValueError: If start or end is outside grid bounds.
        """
        # --- boundary checks ---
        if not self._in_bounds(*start):
            raise ValueError(f"Start {start} is out of grid bounds ({self.rows}x{self.cols})")
        if not self._in_bounds(*end):
            raise ValueError(f"End {end} is out of grid bounds ({self.rows}x{self.cols})")

        # --- wall checks ---
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # --- trivial path ---
        if start == end:
            return [start]

        # --- A* search ---
        directions: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # g_cost: best known cost from start to each node
        g_cost: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()

        # Heap entries: (f_cost, tiebreaker, node)
        # tiebreaker ensures we never compare two tuples of nodes
        counter: int = 0
        open_heap: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (self._heuristic(start, end), counter, start))

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == end:
                # Reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dr, dc in directions:
                neighbor: Tuple[int, int] = (current[0] + dr, current[1] + dc)

                if not self._in_bounds(*neighbor):
                    continue
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue
                if neighbor in closed_set:
                    continue

                move_cost: int = self.grid[neighbor[0]][neighbor[1]]
                tentative_g: int = g_cost[current] + move_cost

                if tentative_g < g_cost.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_g
                    f: int = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, neighbor))

        return None  # open set exhausted — no path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest


def _path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Total cost of entering each cell after the start."""
    return sum(grid[r][c] for r, c in path[1:])


def _assert_valid_path(
    grid: List[List[int]],
    path: List[Tuple[int, int]],
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> None:
    """Shared helper: verify path is contiguous, in-bounds, and wall-free."""
    assert path[0] == start
    assert path[-1] == end
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        # Adjacent (4-directional)
        assert abs(r0 - r1) + abs(c0 - c1) == 1
    for r, c in path:
        assert 0 <= r < len(grid) and 0 <= c < len(grid[0])
        assert grid[r][c] != 0, f"Path passes through wall at ({r},{c})"


def test_simple_uniform_grid() -> None:
    """Straightforward shortest path on a cost-1 grid."""
    grid = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    a_star = AStarGrid(grid)
    start, end = (0, 0), (2, 2)

    path = a_star.find_path(start, end)

    assert path is not None
    _assert_valid_path(grid, path, start, end)
    # Manhattan distance is 4, so 4 moves → 5 cells, cost = 4
    assert len(path) == 5
    assert _path_cost(grid, path) == 4


def test_path_around_obstacles() -> None:
    """Forced detour around a wall."""
    grid = [
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    a_star = AStarGrid(grid)
    start, end = (0, 0), (0, 4)

    path = a_star.find_path(start, end)

    assert path is not None
    _assert_valid_path(grid, path, start, end)
    # Must go around the wall at (1,2).
    # Optimal: (0,0)→(1,0)→(2,0)→(2,1)→(2,2)→(2,3)→(1,3)→(0,3)→(0,4)
    # That's 8 moves, cost = 8.
    assert _path_cost(grid, path) == 8
    # Verify the wall cell is not on the path
    assert (1, 2) not in path


def test_weighted_grid_prefers_lower_cost() -> None:
    """A longer path through cheap cells beats a short path through expensive ones."""
    grid = [
        [1, 10, 10, 10, 1],
        [1,  1,  1,  1, 1],
        [1,  1,  1,  1, 1],
    ]
    a_star = AStarGrid(grid)
    start, end = (0, 0), (0, 4)

    path = a_star.find_path(start, end)

    assert path is not None
    _assert_valid_path(grid, path, start, end)
    # Going along row 0 costs 10+10+10+1 = 31
    # Going through row 1 costs 1+1+1+1+1+1 = 6  ← optimal
    assert _path_cost(grid, path) == 6
    # The path should dip into row 1
    assert any(r == 1 for r, c in path)


def test_no_path_exists() -> None:
    """Fully blocked — no path should be found."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    a_star = AStarGrid(grid)
    start, end = (0, 0), (0, 2)

    path = a_star.find_path(start, end)

    assert path is None


def test_start_equals_end() -> None:
    """When start == end, return a single-element path."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
    ]
    a_star = AStarGrid(grid)
    start = (1, 1)

    path = a_star.find_path(start, start)

    assert path == [start]
    assert len(path) == 1


def test_invalid_coordinates() -> None:
    """Out-of-bounds start or end must raise ValueError."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    a_star = AStarGrid(grid)

    with pytest.raises(ValueError, match="out of grid bounds"):
        a_star.find_path((-1, 0), (0, 0))

    with pytest.raises(ValueError, match="out of grid bounds"):
        a_star.find_path((0, 0), (3, 0))

    with pytest.raises(ValueError, match="out of grid bounds"):
        a_star.find_path((0, 0), (0, 99))


def test_start_or_end_is_wall() -> None:
    """If start or end is a wall cell, return None."""
    grid = [
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
    ]
    a_star = AStarGrid(grid)

    assert a_star.find_path((0, 0), (1, 1)) is None  # start is wall
    assert a_star.find_path((1, 1), (2, 2)) is None  # end is wall
```

### How it works

| Component | Detail |
|---|---|
| **Open set** | `heapq` min-heap of `(f, counter, node)` tuples. The integer `counter` acts as a tiebreaker so Python never has to compare two coordinate tuples lexicographically. |
| **g-cost** | `g_cost[node]` stores the cheapest known cost from `start` to `node`. The cost of *entering* a neighbour cell is `grid[r][c]`. |
| **Heuristic** | Manhattan distance `|Δrow| + |Δcol|` — admissible and consistent on a 4-directional grid, so A* is guaranteed optimal. |
| **Closed set** | A node is expanded at most once. Because the heuristic is consistent, the first time a node is popped it already has its final g-cost. |
| **Path reconstruction** | `came_from` dictionary is walked backwards from `end` to `start`, then reversed. |

### Running the tests

```bash
pytest test_astar.py -v
```

All six tests (plus the bonus wall test) exercise the core requirements: optimality on uniform grids, obstacle avoidance, weighted-cost preference, unreachable targets, the `start == end` shortcut, and `ValueError` on out-of-bounds input.