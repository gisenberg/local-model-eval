We need to produce Python code implementing AStarGrid class with find_path method, using heapq, Manhattan heuristic, weighted cells, walls (0). Provide type hints and docstring. Also produce six pytest tests covering various scenarios: start==end, wall blocking, out-of-bounds raising ValueError, simple grid, weighted cells where optimal path uses lower weight, path must be optimal (check that sum of weights minimal). Also test that diagonal not allowed (4-directional). Provide tests.

We must output final answer with code.

Let's design:

Class AStarGrid:
- __init__(self, grid: List[List[int]]) where grid cells weight >=0; 0 considered wall (impassable). We'll treat weight >0 as traversable with cost = weight (maybe cost to enter cell). For start and end we can allow weight 0? Probably treat them as traversable regardless? Usually start and end can be any passable cell; if they are 0 we can treat as start even if weight 0? We'll assume they must be >0 else raise ValueError.

- find_path(self, start: Tuple[int,int], end: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]
  - checks start, end within bounds and grid value >0 else raise ValueError
  - if start==end: return [start]
  - Use heapq for open set: elements (f_score, g_score, node)
  - g_score dict default inf
  - came_from dict
  - f = g + heuristic (Manhattan)
  - while open_set: pop; if node==end reconstruct path and return
  - For each neighbor (dx,dy) in [(1,0),(-1,0),(0,1),(0,-1)]:
        - check bounds, not wall (grid[nr][nc]>0)
        - tentative_g = g_score[current] + grid[nr][nc]   (cost to enter neighbor)
        - if tentative_g < g_score[neighbor]: update
        - push to heap

- At end, if not found, return None

Heuristic: Manhattan distance * min_cell_cost? Usually heuristic must be admissible; we can just use Manhattan distance * min_weight (maybe 1) but we have weighted cells; but using Manhattan distance (cost per step = 1) may underestimate actual cost if weights>1? Actually weight is cost to enter cell; moving from one cell to adjacent costs weight of neighbor cell (assuming start cost not counted). If we use Manhattan distance * min_possible_weight (which is >0 minimal weight in grid) it's admissible; but for simplicity we can use Manhattan distance (assuming each step cost >=1). If weights can be fractional less than 1, but they are ints; weight>=1 for traversable? Not specified but weight can be any positive int; we can treat weight>=1. So Manhattan heuristic is admissible.

Implementation details: We'll need to import List, Tuple, Optional, heapq.

Now tests: We'll write using pytest.

Test 1: simple 3x3 grid all ones, start (0,0) end (2,2) path length 5 cells? Actually path of moves: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) length 5 cells, cost 5.

Test 2: start==end returns list with that cell.

Test 3: wall blocking: grid with a wall column of zeros, start left, end right; expect None.

Test 4: out-of-bounds start raises ValueError.

Test 5: weighted cells: grid where moving through low weight cells yields lower total cost than Manhattan shortest path. Eg: 3x3:
[[1,100,1],
 [1,100,1],
 [1,1,1]]
Start (0,0) end (2,2). Manhattan shortest path would go right then down etc costing high due to 100 cells; optimal path goes around via bottom row: cost = 1+1+1+1+1+1 =6? Let's compute: path (0,0)->(1,0)->(2,0)->(2,1)->(2,2) cost: entering (1,0)=1, (2,0)=1, (2,1)=1, (2,2)=1 plus start? We don't count start cost. So total 4. Another path via top row then right then down: (0,0)->(0,1)=100 cost huge. So optimal path is the low weight path. We'll test that find_path returns that path and compute total cost equals expected minimal.

Test 6: diagonal not allowed: ensure path doesn't contain diagonal steps. Could test that for a grid with no walls, the returned path only moves in 4 directions.

We'll also include a test for heuristic correctness? Not needed.

Implementation details: We'll need to reconstruct path: start from end, while current in came_from: append current, set current = came_from[current]; finally append start and reverse.

Edge Cases: If start or end is wall (0) raise ValueError.

Now produce final answer with code and tests in separate code blocks maybe.

Let's craft final.





**Solution**

```python
from __future__ import annotations

import heapq
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    """
    A* path‑finder on a rectangular 2‑D grid.

    Parameters
    ----------
    grid : List[List[int]]
        The navigation map.  A cell value of ``0`` is treated as an
        impassable wall.  Positive integers are traversable and their
        value is the **cost to enter** that cell (the cost of the start
        cell itself is not counted).

    The algorithm uses 4‑directional movement (up, down, left, right)
    and the Manhattan heuristic, which is admissible for non‑negative
    integer costs.

    Methods
    -------
    find_path(start, end)
        Returns the optimal path as a list of ``(row, col)`` tuples
        (including ``start`` and ``end``) or ``None`` if no path exists.
        Raises ``ValueError`` if ``start`` or ``end`` are out of bounds
        or lie on a wall.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        if not grid or not grid[0]:
            raise ValueError("grid must be non‑empty")
        self._grid = grid
        self._rows = len(grid)
        self._cols = len(grid[0])

        # sanity check: all rows must have the same length
        if any(len(row) != self._cols for row in grid):
            raise ValueError("all rows in the grid must have the same length")

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance – admissible for 4‑directional grid."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self._rows and 0 <= c < self._cols

    def _is_passable(self, r: int, c: int) -> bool:
        """A cell is passable if it is inside the grid and not a wall."""
        return self._in_bounds(r, c) and self._grid[r][c] > 0

    def _neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """4‑directional neighbours."""
        r, c = pos
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [n for n in candidates if self._is_passable(*n)]

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find the cheapest path from *start* to *end* using A*.

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            List of coordinates from start to end (inclusive) if a path
            exists; otherwise ``None``.

        Raises
        ------
        ValueError
            If ``start`` or ``end`` are out of bounds or located on a wall.
        """
        # ----- validation -------------------------------------------------
        if not self._in_bounds(*start) or not self._in_bounds(*end):
            raise ValueError("start or end is out of grid bounds")
        if not self._is_passable(*start) or not self._is_passable(*end):
            raise ValueError("start or end is located on a wall (value 0)")

        if start == end:
            return [start]

        # ----- A* core ----------------------------------------------------
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []  # (f, g, node)
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))

        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == end:
                # reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self._neighbors(current):
                tentative_g = current_g + self._grid[neighbor[0]][neighbor[1]]
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))

        # open set exhausted → no path
        return None
```

---

### Pytest test‑suite

```python
import pytest
from typing import List, Tuple

# Assuming the implementation above is saved in a file named `astar_grid.py`
from astar_grid import AStarGrid


def test_simple_unweighted_grid():
    """All cells cost 1 – the shortest Manhattan path should be found."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Manhattan distance is 4 steps → 5 cells in the path
    assert len(path) == 5
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # verify each step is orthogonal
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_start_equals_end():
    """When start == end the function must return a single‑element list."""
    grid = [[1, 2], [3, 4]]
    astar = AStarGrid(grid)
    assert astar.find_path((1, 0), (1, 0)) == [(1, 0)]


def test_wall_blocks_path():
    """A full column of zeros should make the target unreachable."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None


def test_out_of_bounds_raises():
    """Accessing coordinates outside the grid must raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))


def test_weighted_prefers_low_cost_route():
    """
    The algorithm must choose a lower‑total‑cost path even if it is longer
    in terms of step count.
    """
    grid = [
        [1, 100, 1],
        [1, 100, 1],
        [1,   1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Expected optimal route: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)
    expected = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert path == expected
    # Compute total cost (excluding the start cell)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4  # 1+1+1+1


def test_path_contains_only_orthogonal_moves():
    """Ensure the returned path never contains a diagonal step."""
    grid = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    astar = AStarGrid(grid)
    # Use a case where many equivalent shortest paths exist;
    # we just check that whatever path is returned obeys 4‑directionality.
    path = astar.find_path((0, 0), (3, 3))
    assert path is not None
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # Manhattan distance == 1


if __name__ == "__main__":
    # Allow running the file directly for a quick sanity check
    pytest.main([__file__])
```

**Explanation of the implementation**

* The grid stores the cost to *enter* each cell; a value of `0` denotes an impassable wall.
* `find_path` validates the start/end coordinates, handles the trivial case, then runs the classic A* loop:
  * `open_set` is a priority queue (`heapq`) ordered by `f = g + h`.
  * `g_score` holds the cheapest known cost to reach each node.
  * `came_from` enables path reconstruction.
  * Neighbours are limited to the four orthogonal directions.
* The Manhattan heuristic (`|dx|+|dy|`) is admissible because every step costs at least 1 (all traversable cells have a positive integer weight).
* The test suite checks normal operation, start==end, wall blocking, out‑of‑bounds errors, weighted optimality, and that the algorithm never steps diagonally. Each test uses `assert` statements and `pytest.raises` where appropriate.