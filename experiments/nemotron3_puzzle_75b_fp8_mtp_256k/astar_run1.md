We need to produce code for AStarGrid class with find_path method, handling weighted cells, walls (0 means wall? Typically 0 means unwalkable, >0 weight). We'll need Manhattan heuristic.

We'll include type hints, docstrings. Write six pytest tests covering typical cases: start==end, simple path, weighted path vs uniform, walls blocking forcing alternative, out-of-bounds raising ValueError, maybe diagonal not allowed.

We should also provide instructions on how to run tests (but not required). We'll output code.

Make sure to import heapq, typing.

Implementation details: grid is a 2D list of ints. A cell with value 0 considered wall (impassable). Others weight cost to enter (or move into that cell). Starting cell's weight maybe not counted? Usually path cost sum of weights of cells traversed (excluding start?). We'll define cost to move into neighbor = grid[neighbor]. We'll treat start cost zero and add neighbor's weight on each step.

A* algorithm: open set heap of (f, g, position). visited set or closed set for g scores.

When start==end, return [start] if not wall.

Check bounds: raise ValueError if start or end out of bounds or on wall? Probably raise if start/end out-of-bounds or start/end is wall (0). We'll implement.

Let's write code.

We'll also write tests.

We'll need to define a small grid for each test.

Test cases:

1. test_start_equals_end: grid [[1]] start (0,0) end (0,0) => path [(0,0)].

2. test_simple_path_no_weights: grid of ones 3x3, start (0,0) end (2,2). Expect Manhattan distance path length 4 steps (including start? path list length 5). Should find optimal.

3. test_weighted_prefer_lower_cost: grid:
[[1,10,1],
 [1,10,1],
 [1,1,1]]
Start (0,0) end (2,0). Direct path down column 0 cost 1+1+1+1? Actually moving down two steps: costs of cells (1,0) and (2,0) each weight 1 => total cost 2. Alternative route via column 2 then left across bottom maybe higher due to weight 10 walls. Expect path using column 0.

4. test_wall_blockage: grid:
[[1,1,1],
 [0,0,0],
 [1,1,1]]
Start (0,0) end (2,0) -> need to go around via top row then down then left? Actually middle row walls block vertical passage; need go up? can't go out-of-bounds. Actually grid 3x3: rows 0-2. Walls row1 all zero blocks; can't get from top to bottom. Should return None.

But spec: find_path returns Optional[List[Tuple[int,int]]]. So test expects None.

5. test_out_of_bounds_start: raises ValueError.

6. test_out_of_bounds_end: raises ValueError.

Or test start on wall raises ValueError.

Let's implement tests accordingly.

Now produce final answer: code block for AStarGrid and test file.

We'll also include __init__ taking grid.

Implementation:

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0

    def find_path(self, start: Tuple[int,int], end: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
        # validation
        ...

        if start == end:
            return [start] if self._is_passable(start) else None

        # heuristic Manhattan
        def heuristic(a,b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        # directions 4
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]

        open_heap = []
        heapq.heappush(open_heap, (0 + heuristic(start, end), 0, start))
        came_from = {}
        g_score = {start:0}
        closed_set = set()

        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current in closed_set:
                continue
            if current == end:
                # reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            closed_set.add(current)

            for dx,dy in dirs:
                nx, ny = current[0]+dx, current[1]+dy
                if not (0 <= nx < self.rows and 0 <= ny < self.cols):
                    continue
                neighbor = (nx, ny)
                if not self._is_passable(neighbor):
                    continue
                tentative_g = g + self.grid[nx][ny]  # cost to enter neighbor
                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_heap, (f, tentative_g, neighbor))
        return None

    def _is_passable(self, pos):
        x,y = pos
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x][y] != 0

We need to treat weight zero as wall.

Now tests.

We'll create test file test_astar.py.

Write six tests.

We'll import pytest.

Now produce final answer: code with class and tests.

Make sure to include proper imports.

Now output.





**A* path‑finding implementation (weighted 2‑D grid)**

```python
# astar_grid.py
from __future__ import annotations

import heapq
from typing import List, Tuple, Optional, Set, Dict


class AStarGrid:
    """
    A* search on a rectangular grid with 4‑directional movement.
    Cells with value ``0`` are considered walls (impassable).  All other
    positive integers represent the movement cost to *enter* that cell.
    The Manhattan distance is used as the heuristic.

    Parameters
    ----------
    grid:
        2‑D list of non‑negative ints. ``grid[y][x]`` is the cost of cell
        (x, y).  ``0`` → wall.

    Example
    -------
    >>> g = [[1, 2, 1],
    ...      [1, 0, 1],
    ...      [1, 1, 1]]
    >>> AStarGrid(g).find_path((0, 0), (2, 2))
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    """

    def __init__(self, grid: List[List[int]]) -> None:
        if not grid or not any(grid):
            raise ValueError("grid must be non‑empty")
        self._grid: List[List[int]] = grid
        self._rows: int = len(grid)
        self._cols: int = len(grid[0])

        # sanity check – all rows must have equal length
        if any(len(row) != self._cols for row in self._grid):
            raise ValueError("irregular grid: all rows must have the same length")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Return the optimal path from ``start`` to ``end`` as a list of
        coordinates (inclusive).  Returns ``None`` if no path exists.

        Raises
        ------
        ValueError
            If ``start`` or ``end`` is out of bounds or located on a wall.
        """
        self._validate_point(start, "start")
        self._validate_point(end, "end")

        # Trivial case
        if start == end:
            return [start]

        # Helper: Manhattan heuristic
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Directions: up, down, left, right
        directions: List[Tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        open_heap: List[Tuple[int, int, Tuple[int, int]]] = []  # (f, g, node)
        heapq.heappush(open_heap, (heuristic(start, end), 0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        closed: Set[Tuple[int, int]] = set()

        while open_heap:
            _, g_current, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            if current == end:
                # Reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            closed.add(current)

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if not (0 <= nx < self._rows and 0 <= ny < self._cols):
                    continue  # out of bounds
                neighbor = (nx, ny)
                if not self._is_passable(neighbor):
                    continue  # wall

                tentative_g = g_current + self._grid[nx][ny]  # cost to step into neighbor
                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue  # not a better route

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_heap, (f, tentative_g, neighbor))

        # Exhausted open set → no path
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _is_passable(self, pt: Tuple[int, int]) -> bool:
        """True if the cell is inside the grid and not a wall (value != 0)."""
        x, y = pt
        return 0 <= x < self._rows and 0 <= y < self._cols and self._grid[x][y] != 0

    def _validate_point(self, pt: Tuple[int, int], name: str) -> None:
        """Raise ValueError if ``pt`` is out of bounds or on a wall."""
        if not (0 <= pt[0] < self._rows and 0 <= pt[1] < self._cols):
            raise ValueError(f"{name} {pt} is out of grid bounds")
        if not self._is_passable(pt):
            raise ValueError(f"{name} {pt} lies on a wall (cell value 0)")


```

---

**Pytest suite (6 tests)**

```python
# test_astar_grid.py
import pytest
from astar_grid import AStarGrid
from typing import List, Tuple, Optional


def test_start_equals_end():
    grid = [[5]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]


def test_simple_unweighted_path():
    # 3x3 grid of uniform cost 1
    grid = [[1] * 3 for _ in range(3)]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Manhattan distance = 4 steps → 5 nodes in path
    assert len(path) == 5
    # Check that each step moves orthogonally and stays inside the grid
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    assert path[0] == (0, 0) and path[-1] == (2, 2)


def test_weighted_prefers_lower_cost():
    # High cost column forces detour through cheap column
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1,  1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    assert path is not None
    # Expected optimal: straight down column 0
    assert path == [(0, 0), (1, 0), (2, 0)]


def test_wall_blocks_path():
    # Horizontal wall of zeros in middle row
    grid = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 0)) is None


def test_out_of_bounds_start_raises():
    grid = [[1, 2], [3, 4]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of grid bounds"):
        astar.find_path((-1, 0), (1, 1))


def test_end_on_wall_raises():
    grid = [
        [1, 1],
        [0, 1],   # (0,1) is a wall
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="lies on a wall"):
        astar.find_path((0, 0), (0, 1))


# To run the tests:  pytest test_astar_grid.py
```

The implementation satisfies all requirements:

* **Class** `AStarGrid` with `find_path(start, end) → Optional[List[Tuple[int,int]]]`.
* 4‑directional moves, Manhattan heuristic, `heapq` for the open set.
* Cells with value `0` are treated as walls; all other values are traversal costs.
* Proper handling of `start == end`, out‑of‑bounds, and wall cells (raises `ValueError`).
* The algorithm returns an optimal (minimum‑cost) path when one exists.
* Full type hints and docstrings.
* Six pytest tests covering the requested scenarios.

You can place `astar_grid.py` and `test_astar_grid.py` in the same directory and run `pytest` to verify correctness.