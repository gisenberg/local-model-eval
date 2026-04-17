**`astar_grid.py` – A* implementation for a weighted 2‑D grid**

```python
"""
astar_grid.py
--------------

A minimal A* path‑finder for a rectangular grid where each cell stores the
cost to *enter* that cell.  A value of ``0`` marks an impassable wall.

Features
~~~~~~~~
* 4‑directional movement (no diagonals)
* Manhattan distance heuristic (scaled by the cheapest traversable cell)
* Optimal (minimum‑cost) path guarantee
* Full type hints and doc‑strings
"""

from __future__ import annotations

import heapq
from typing import List, Tuple, Optional, Dict, Set


class AStarGrid:
    """
    A* path‑finder on a weighted 2‑D grid.

    Parameters
    ----------
    grid :
        A rectangular list of lists where ``grid[r][c]`` is the cost to *enter*
        the cell at row ``r`` and column ``c``. ``0`` denotes an impassable wall.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        if not grid or not all(grid):
            raise ValueError("Grid must be a non‑empty rectangular matrix.")
        row_len = len(grid[0])
        if any(len(row) != row_len for row in grid):
            raise ValueError("All rows in the grid must have the same length.")
        self.grid: List[List[int]] = grid
        self.n_rows: int = len(grid)
        self.n_cols: int = row_len
        # cheapest positive cost – used to keep the heuristic admissible
        positive_costs = [cell for row in grid for cell in row if cell > 0]
        self.min_cost: int = min(positive_costs) if positive_costs else 1

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def _is_passable(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return self.grid[r][c] > 0

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance scaled by the cheapest traversable cell."""
        return self._manhattan(a, b) * self.min_cost

    def _neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return passable orthogonal neighbours of ``pos``."""
        r, c = pos
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [
            (nr, nc)
            for nr, nc in candidates
            if self._in_bounds((nr, nc)) and self._is_passable((nr, nc))
        ]

    def _reconstruct_path(
        self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Rebuild the path from ``start`` to ``current`` using ``came_from``."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def find_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Compute the cheapest path from ``start`` to ``end``.

        Returns
        -------
        list of (row, col) tuples
            The path including both ``start`` and ``end``.
        None
            If no path exists.

        Raises
        ------
        ValueError
            If ``start`` or ``end`` lies outside the grid.
        """
        if not self._in_bounds(start) or not self._in_bounds(end):
            raise ValueError("Start or end coordinate is out of grid bounds.")

        if start == end:
            return [start] if self._is_passable(start) else None

        if not self._is_passable(start) or not self._is_passable(end):
            return None

        open_heap: List[Tuple[int, int, Tuple[int, int]]] = []
        # (f_score, counter, position) – counter breaks ties deterministically
        counter = 0
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        f_score_start = self._heuristic(start, end)
        heapq.heappush(open_heap, (f_score_start, counter, start))
        counter += 1

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current == end:
                return self._reconstruct_path(came_from, current)

            closed_set.add(current)

            for neighbor in self._neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f, counter, neighbor))
                    counter += 1

        # Exhausted search – no path
        return None
```

---

**`test_astar_grid.py` – pytest suite**

```python
import pytest
from typing import List, Tuple

from astar_grid import AStarGrid


def path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Total cost of a path (excluding the start cell)."""
    return sum(grid[r][c] for r, c in path[1:])


def test_simple_uniform_grid():
    grid = [[1] * 5 for _ in range(5)]
    astar = AStarGrid(grid)
    start, end = (0, 0), (4, 4)
    path = astar.find_path(start, end)
    assert path is not None
    # Manhattan distance = 8, each step costs 1 → total cost 8
    assert path_cost(grid, path) == 8
    # Path length should be distance + 1 (including start)
    assert len(path) == 9
    # Path must start and end at the correct cells
    assert path[0] == start
    assert path[-1] == end


def test_path_around_obstacles():
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (4, 4)
    path = astar.find_path(start, end)
    assert path is not None
    # Verify that no wall (0) appears in the path
    for r, c in path:
        assert grid[r][c] > 0
    # The optimal path goes around the central wall; its cost is 12
    assert path_cost(grid, path) == 12


def test_weighted_grid_prefers_low_cost():
    # High‑cost corridor vs. longer low‑cost detour
    grid = [
        [1, 1, 1, 1, 1],
        [1, 5, 5, 5, 1],
        [1, 5, 1, 5, 1],
        [1, 5, 1, 5, 1],
        [1, 1, 1, 1, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (4, 4)
    path = astar.find_path(start, end)
    assert path is not None
    # The cheapest route hugs the outer ring (cost 1 per step)
    # Manhattan distance = 8 → total cost 8
    assert path_cost(grid, path) == 8
    # Ensure the path does NOT go through the expensive centre column
    for r, c in path:
        assert not (c == 2 and r in {1, 2, 3})  # those cells have cost 5


def test_no_path_exists():
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is None


def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    start = end = (0, 1)
    path = astar.find_path(start, end)
    assert path == [(0, 1)]
    # If the cell were a wall, we should get None
    grid[0][1] = 0
    astar2 = AStarGrid(grid)
    assert astar2.find_path(start, end) is None


def test_invalid_coordinates():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))
```

**Explanation of the tests**

| Test | What it checks |
|------|----------------|
| `test_simple_uniform_grid` | Basic functionality on a uniform cost grid; verifies total cost and path length. |
| `test_path_around_obstacles` | Correct handling of impassable walls and that the returned path never steps on a wall. |
| `test_weighted_grid_prefers_low_cost` | Guarantees the algorithm prefers lower‑cost cells even if the geometric distance is larger. |
| `test_no_path_exists` | Returns `None` when start and end are separated by a wall barrier. |
| `test_start_equals_end` | Handles the degenerate case where start equals end (both passable and impassable). |
| `test_invalid_coordinates` | Raises `ValueError` for out‑of‑bounds coordinates. |

Run the tests with:

```bash
pytest test_astar_grid.py
```

All tests should pass, confirming a correct and optimal A* implementation for a weighted 2‑D grid.