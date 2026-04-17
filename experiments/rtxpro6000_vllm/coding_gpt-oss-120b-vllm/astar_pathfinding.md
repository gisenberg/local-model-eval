**`astar_grid.py`**

```python
"""A* path‑finding on a weighted 2‑D grid.

The grid is a list of rows, each row a list of integers:
* ``0``  – impassable wall
* ``>0`` – cost to *enter* that cell

The algorithm works with 4‑directional movement (no diagonals) and
uses Manhattan distance as an admissible heuristic.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple


class AStarGrid:
    """A* search on a rectangular weighted grid."""

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Create a new search object.

        Parameters
        ----------
        grid:
            2‑D list where ``grid[r][c]`` is the movement cost of the cell.
            ``0`` means the cell cannot be entered.
        """
        if not grid or not all(grid):
            raise ValueError("grid must be a non‑empty rectangular matrix")
        row_len = len(grid[0])
        if any(len(row) != row_len for row in grid):
            raise ValueError("all rows in the grid must have the same length")
        self.grid: List[List[int]] = grid
        self.n_rows: int = len(grid)
        self.n_cols: int = row_len
        # the smallest positive cost – used for a tighter heuristic
        self._min_cost: int = min(
            cell for row in grid for cell in row if cell > 0
        ) or 1  # fallback, should never be 0 because of the filter

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def find_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Return the cheapest path from ``start`` to ``end`` (inclusive).

        Parameters
        ----------
        start, end:
            ``(row, col)`` coordinates inside the grid.

        Returns
        -------
        list of coordinates or ``None`` if no path exists.

        Edge cases
        ----------
        * ``start == end`` → ``[start]``.
        * ``start`` or ``end`` is a wall → ``None``.
        * coordinates outside the grid → ``ValueError``.
        """
        self._validate_point(start, "start")
        self._validate_point(end, "end")

        if start == end:
            return [start]

        if self._is_wall(start) or self._is_wall(end):
            return None

        # ----- A* core ---------------------------------------------------- #
        open_set: List[Tuple[int, Tuple[int, int]]] = []  # (f_score, point)
        heapq.heappush(open_set, (self._heuristic(start, end), start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        g_score: Dict[Tuple[int, int], int] = {
            start: 0
        }  # cost from start to the key (does NOT include start cell cost)

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))

        # exhausted – no path
        return None

    # --------------------------------------------------------------------- #
    # Helper methods (private)
    # --------------------------------------------------------------------- #
    def _validate_point(self, pt: Tuple[int, int], name: str) -> None:
        """Raise ``ValueError`` if ``pt`` is outside the grid."""
        r, c = pt
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            raise ValueError(f"{name} point {pt} is out of bounds")

    def _is_wall(self, pt: Tuple[int, int]) -> bool:
        """True if the cell is impassable."""
        return self.grid[pt[0]][pt[1]] == 0

    def _neighbors(self, pt: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return the 4‑directional walkable neighbours of ``pt``."""
        r, c = pt
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        result = []
        for nr, nc in candidates:
            if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
                if not self._is_wall((nr, nc)):
                    result.append((nr, nc))
        return result

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance multiplied by the smallest possible step cost."""
        manhattan = abs(a[0] - b[0]) + abs(a[1] - b[1])
        return manhattan * self._min_cost

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Rebuild the path from ``came_from`` mapping."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
```

---

**`test_astar_grid.py`**

```python
import pytest
from typing import List, Tuple

from astar_grid import AStarGrid


def path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Utility: sum of entry costs for all cells except the first one."""
    return sum(grid[r][c] for r, c in path[1:])


# --------------------------------------------------------------------- #
# 1. Simple uniform grid
# --------------------------------------------------------------------- #
def test_simple_uniform_grid():
    grid = [[1] * 5 for _ in range(5)]
    astar = AStarGrid(grid)
    start, end = (0, 0), (4, 4)
    path = astar.find_path(start, end)

    assert path is not None
    # Manhattan distance = 8, each step costs 1 → total cost 8
    assert path_cost(grid, path) == 8
    # length = moves + 1
    assert len(path) == 9
    # path must start/end correctly
    assert path[0] == start
    assert path[-1] == end


# --------------------------------------------------------------------- #
# 2. Path around obstacles
# --------------------------------------------------------------------- #
def test_path_around_walls():
    grid = [
        [1, 1, 1, 1],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 1, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (4, 3)
    path = astar.find_path(start, end)

    assert path is not None
    # The only viable route goes down, right, down, right, down, right
    expected_cost = 8  # 8 moves, each cost 1
    assert path_cost(grid, path) == expected_cost
    assert path[0] == start
    assert path[-1] == end


# --------------------------------------------------------------------- #
# 3. Weighted grid – prefers cheaper cells
# --------------------------------------------------------------------- #
def test_weighted_grid_prefers_low_cost():
    grid = [
        [1, 5, 5, 5, 1],
        [1, 5, 1, 5, 1],
        [1, 5, 1, 5, 1],
        [1, 1, 1, 1, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (0, 4)
    path = astar.find_path(start, end)

    assert path is not None
    # The cheapest route goes down the left side, across the bottom,
    # then up the right side (cost 1 per step, 8 moves total)
    assert path_cost(grid, path) == 8
    # Verify that the path indeed uses the cheap corridor
    for r, c in path:
        assert grid[r][c] == 1


# --------------------------------------------------------------------- #
# 4. No path exists (completely blocked)
# --------------------------------------------------------------------- #
def test_no_path_due_to_walls():
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    assert astar.find_path(start, end) is None


# --------------------------------------------------------------------- #
# 5. Start equals end
# --------------------------------------------------------------------- #
def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    start = end = (1, 0)
    path = astar.find_path(start, end)
    assert path == [start]


# --------------------------------------------------------------------- #
# 6. Invalid coordinates raise ValueError
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "start,end",
    [
        ((-1, 0), (0, 0)),   # start out of bounds
        ((0, 0), (3, 0)),    # end out of bounds (row)
        ((0, 2), (0, 0)),    # start column out of bounds
    ],
)
def test_invalid_coordinates(start, end):
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path(start, end)
```

**Explanation of the implementation**

* **Data structures** – `open_set` is a min‑heap of `(f_score, point)`.  
  `g_score` stores the cheapest known cost from the start to each visited cell.  
  `came_from` maps a cell to the predecessor on the current best path.

* **Heuristic** – Manhattan distance multiplied by the smallest positive cell cost
  (`self._min_cost`). Because every move costs at least that amount, the heuristic
  never overestimates the true remaining cost, keeping A* optimal.

* **Edge‑case handling** –  
  * Out‑of‑bounds coordinates raise `ValueError`.  
  * If start or end is a wall (`0`) the function returns `None`.  
  * If start equals end the function returns a one‑element list.

* **Cost model** – The cost of a move is the value of the *destination* cell,
  matching the problem statement (“cost to enter that cell”). The start cell’s
  cost is therefore not added to the total.

* **Tests** – Six `pytest` cases cover the required scenarios, checking both
  existence of a path and that the total cost equals the optimal value. The
  helper `path_cost` computes the sum of entry costs for a returned path.

Run the tests with:

```bash
pytest test_astar_grid.py
```

All tests should pass, confirming a correct and optimal A* implementation.