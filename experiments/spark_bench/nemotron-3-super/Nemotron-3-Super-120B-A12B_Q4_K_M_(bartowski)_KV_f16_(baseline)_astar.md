# Nemotron-3-Super-120B-A12B Q4_K_M (bartowski) — KV f16 (baseline) — A* Pathfinding

## Thinking (3541 chars)

We need to produce code implementing AStarGrid with required methods, type hints, docstrings, using heapq, Manhattan heuristic. Also need to write 6 pytest tests covering cases. Provide code in a single block? Probably separate sections: implementation and tests. Ensure imports: typing List, Tuple, Optional, heapq. Provide docstrings.

Edge cases: start or end out of bounds raise ValueError. start or end is a wall (grid value 0) return None. start == end return [start] (if not wall? Should check if start is wall? Probably if start is wall then no path; but spec says start or end is a wall return None, so if start==end and it's a wall, return None. We'll handle that.

Implementation details: open set heap entries: (f_score, g_score, counter, node). Use counter to break ties. Keep came_from dict. g_score dict default inf. Initialize g_score[start]=0. f_score = g + h.

When exploring neighbors: if neighbor is out of bounds skip; if grid[neighbor]==0 skip (impassable). tentative_g = g_score[current] + grid[neighbor] (cost to enter neighbor). If tentative_g < g_score[neighbor], update.

At end, reconstruct path by backtracking from end to start via came_from, then reverse and include start.

Return None if open set exhausted.

Now tests: need to assert path validity and optimality (total cost). We'll compute total cost as sum of grid values for each cell in path (including start? Usually cost to enter start? Since movement cost to enter a cell, start cost maybe not counted? For path cost we can sum grid[cell] for each cell in path (including start) but if start cost considered? Usually cost to move into start is zero because you start there. But spec: grid values represent movement cost to enter that cell. Starting cell you are already in, so cost to enter start maybe not counted. For simplicity, we can compute total cost as sum of grid[cell] for each cell in path except start. Or we can include start but then uniform grid cost will be off by start cost. We'll decide to compute cost as sum of grid[cell] for each cell in path (including start) and note that for uniform grid cost = len(path)*cost_per_cell. For optimality test we can compare with alternative paths? Simpler: compute cost of found path and assert it's minimal by checking that any alternative path would have >= cost? That's heavy. Instead we can assert that the found path cost equals the known optimal cost for the test case (we can compute manually). For simple uniform grid, we know optimal path is Manhattan distance steps * cost (if cost uniform). For weighted grid we can design a scenario where lower-cost path is known.

We'll write tests:

1. simple path uniform grid 3x3 all cost 1, start (0,0) end (2,2). Expected path length 5 cells (including start) => cost = sum of costs for each cell entered? If we count start cost, cost =5. Manhattan distance steps =4 moves, each move enters a new cell cost 1 => total cost =4 (excluding start). We'll decide to compute cost as sum of grid[cell] for each cell in path (including start) and then assert cost == (len(path))*1. That will hold.

2. path around obstacles: grid with a wall blocking direct path, need to go around.

3. weighted grid: create grid where a high-cost corridor vs low-cost detour.

4. no path exists: fully blocked.

5. start equals end: return [start].

6. invalid coordinates: start out of bounds -> ValueError.

We'll implement tests using pytest.

Now produce final answer with code block for implementation and another for tests. Ensure type hints.

Let's craft.



## Output

**Implementation (`astar_grid.py`)**

```python
from __future__ import annotations

import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* path‑finding on a 2‑D grid with movement costs.

    The grid is a list of lists where:
        * 0   → impassable wall
        * >0  → cost to *enter* that cell
    Movement is allowed in the four cardinal directions.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Store the grid and pre‑compute its dimensions.

        Args:
            grid: 2‑D list of non‑negative integers representing movement cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non‑empty")
        self._grid: List[List[int]] = grid
        self._rows: int = len(grid)
        self._cols: int = len(grid[0])

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _in_bounds(self, cell: Tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self._rows and 0 <= c < self._cols

    def _is_wall(self, cell: Tuple[int, int]) -> bool:
        r, c = cell
        return self._grid[r][c] == 0

    def _neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = cell
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [nb for nb in candidates if self._in_bounds(nb) and not self._is_wall(nb)]

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Return the cheapest path from *start* to *end* (inclusive) using A*,
        or ``None`` if no path exists.

        The total cost of a path is the sum of the movement costs of all
        entered cells (the start cell is **not** charged because you begin
        there).

        Raises:
            ValueError: if *start* or *end* lies outside the grid.
        """
        # ----- validation -------------------------------------------------
        if not self._in_bounds(start) or not self._in_bounds(end):
            raise ValueError("Start or end coordinate is out of bounds")
        if self._is_wall(start) or self._is_wall(end):
            # If start == end and it is a wall we also return None
            return None
        if start == end:
            return [start]

        # ----- A* core ----------------------------------------------------
        open_heap: List[Tuple[int, int, int, Tuple[int, int]]] = []  # (f, g, counter, node)
        counter = 0  # tie‑breaker for heapq
        heapq.heappush(open_heap, (0, 0, counter, start))
        counter += 1

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        f_score: Dict[Tuple[int, int], int] = {
            start: self._manhattan(start, end)
        }

        while open_heap:
            _, g_current, _, current = heapq.heappop(open_heap)

            if current == end:
                # reconstruct path
                path: List[Tuple[int, int]] = [end]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                path.reverse()
                return path

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + self._grid[neighbor[0]][neighbor[1]]
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._manhattan(neighbor, end)
                    f_score[neighbor] = f
                    heapq.heappush(open_heap, (f, tentative_g, counter, neighbor))
                    counter += 1

        # open set exhausted → no path
        return None
```

---

**Test suite (`test_astar_grid.py`)**

```python
import pytest
from typing import List, Tuple
from astar_grid import AStarGrid


def _path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Sum of movement costs for all cells in *path* except the start cell."""
    if not path:
        return 0
    total = 0
    for cell in path[1:]:  # skip start – you don't pay to be there initially
        r, c = cell
        total += grid[r][c]
    return total


def test_simple_path_uniform_grid():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    # path must start and end correctly
    assert path[0] == start and path[-1] == end
    # each step moves to a neighbour (4‑dir)
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    # optimal cost = Manhattan distance * cell cost (4 moves * 1)
    assert _path_cost(grid, path) == 4


def test_path_around_obstacles():
    grid = [
        [1, 1, 1],
        [1, 0, 1],   # wall at (1,1)
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert path[0] == start and path[-1] == end
    # ensure we never step on the wall
    assert all(grid[r][c] != 0 for r, c in path)
    # optimal detour costs 6 (two extra steps around the wall)
    assert _path_cost(grid, path) == 6


def test_weighted_grid_prefers_lower_cost():
    # high cost corridor (value 5) vs low‑cost detour (value 1)
    grid = [
        [1, 5, 1, 1],
        [1, 5, 1, 5],
        [1, 1, 1, 5],
        [1, 5, 5, 1],
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (3, 3)
    path = astar.find_path(start, end)
    assert path is not None
    assert path[0] == start and path[-1] == end
    # The cheapest route goes right‑then‑down along the top/left border
    expected_path = [(0, 0), (0, 1), (0, 2), (0, 3),
                     (1, 3), (2, 3), (3, 3)]
    # verify that the found path matches the known optimal one
    assert path == expected_path
    # compute its cost: sum of entered cells (skip start)
    assert _path_cost(grid, path) == 1 + 1 + 1 + 5 + 1 + 1  # =10


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
    grid = [
        [2, 2],
        [2, 2],
    ]
    astar = AStarGrid(grid)
    start = end = (0, 1)
    path = astar.find_path(start, end)
    assert path == [start]


def test_invalid_coordinates_raises():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))   # row out of bounds
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))   # col out of bounds
```

**How to run the tests**

```bash
pytest test_astar_grid.py
```

The implementation satisfies all requirements:

* `AStarGrid.__init__` stores the grid and checks dimensions.  
* `find_path` validates bounds and walls, handles the trivial case, and runs A* with a min‑heap (`heapq`).  
* Manhattan distance is used as the heuristic.  
* The returned path is optimal (lowest total movement cost).  
* Full type hints and docstrings are provided.  
* The six pytest cases cover uniform grids, obstacles, weighted preferences, blocked scenarios, start‑equals‑end, and invalid coordinates, each asserting both path validity and optimality.
