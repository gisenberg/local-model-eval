```python
from typing import List, Tuple, Optional
import heapq


class AStarGrid:
    """A* pathfinder on a 2D weighted grid."""

    def __init__(self, grid: List[List[int]]):
        """grid[r][c] == 0 is a wall, >0 is cost to enter."""
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0

    # ---- helpers ----
    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _passable(self, r: int, c: int) -> bool:
        return self.grid[r][c] > 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---- public ----
    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Return optimal path inclusive of start/end or None."""
        sr, sc = start
        er, ec = end

        if not self._in_bounds(sr, sc) or not self._in_bounds(er, ec):
            raise ValueError("start or end out of bounds")

        if not self._passable(sr, sc) or not self._passable(er, ec):
            return None

        if start == end:
            return [start]

        open_heap = []
        heapq.heappush(open_heap, (self._heuristic(start, end), 0, sr, sc))
        g_score = {start: 0}
        came_from = {}
        closed = set()

        while open_heap:
            _, g_cur, r, c = heapq.heappop(open_heap)
            cur = (r, c)
            if cur in closed:
                continue
            if cur == end:
                # reconstruct
                path = [end]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                path.reverse()
                return path
            closed.add(cur)

            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not self._in_bounds(nr, nc) or not self._passable(nr, nc):
                    continue
                neigh = (nr, nc)
                tentative_g = g_cur + self.grid[nr][nc]
                if tentative_g < g_score.get(neigh, float("inf")):
                    g_score[neigh] = tentative_g
                    f = tentative_g + self._heuristic(neigh, end)
                    heapq.heappush(open_heap, (f, tentative_g, nr, nc))
                    came_from[neigh] = cur

        return None
```

```python
# tests
import pytest
from astar import AStarGrid

def path_cost(path, grid):
    return sum(grid[r][c] for r, c in path[1:])  # exclude start

def test_simple_path_uniform():
    grid = [[1]*3 for _ in range(3)]
    a = AStarGrid(grid)
    path = a.find_path((0,0),(2,2))
    assert path is not None
    assert path[0]==(0,0) and path[-1]==(2,2)
    assert len(path)==5
    assert path_cost(path, grid)==4

def test_path_around_obstacles():
    grid = [
        [1,1,1],
        [0,0,1],
        [1,1,1]
    ]
    a = AStarGrid(grid)
    path = a.find_path((0,0),(2,0))
    assert path is not None
    assert path[0]==(0,0) and path[-1]==(2,0)
    for r,c in path:
        assert grid[r][c]!=0
    # must go around the wall
    assert (1,0) not in path and (1,1) not in path

def test_weighted_grid_prefers_lower_cost():
    grid = [
        [1,1,1],
        [1,100,1],
        [1,1,1]
    ]
    a = AStarGrid(grid)
    path = a.find_path((0,0),(2,2))
    assert path is not None
    # optimal cost is 4 via perimeter, not via centre
    assert path_cost(path, grid) == 4
    assert (1,1) not in path

def test_no_path_exists():
    grid = [
        [1,0,1],
        [1,0,1],
        [1,0,1]
    ]
    a = AStarGrid(grid)
    assert a.find_path((0,0),(2,2)) is None

def test_start_equals_end():
    grid = [[1,1],[1,1]]
    a = AStarGrid(grid)
    assert a.find_path((1,1),(1,1)) == [(1,1)]

def test_invalid_coordinates():
    grid = [[1,1],[1,1]]
    a = AStarGrid(grid)
    with pytest.raises(ValueError):
        a.find_path((-1,0),(1,1))
    with pytest.raises(ValueError):
        a.find_path((0,0),(5,5))
    # wall -> None, not error
    grid_wall = [[0,1],[1,1]]
    a2 = AStarGrid(grid_wall)
    assert a2.find_path((0,0),(1,1)) is None
```

The implementation uses a min-heap for the open set, Manhattan distance as an admissible heuristic, 4-directional moves, and returns the minimum total entry cost.